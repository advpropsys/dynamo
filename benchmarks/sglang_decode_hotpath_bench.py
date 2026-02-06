"""
Micro-benchmark: Decode handler per-token hot path analysis.

Breaks down EVERY operation that happens on the decode handler's critical path:
- Per-request setup (generate method preamble)
- Per-token processing in _process_token_stream
- Each dict access, dict creation, branch, yield

Compares current implementation vs optimized alternatives.

Usage:
    python benchmarks/sglang_decode_hotpath_bench.py
"""

import asyncio
import statistics
import time


def make_fake_sglang_response(token_idx: int, n_tokens: int, request_id: str = "req-abc123"):
    """Create a fake SGLang engine response dict matching real format."""
    is_last = token_idx == n_tokens - 1
    return {
        "output_ids": [token_idx + 1000],
        "text": "",
        "index": 0,
        "meta_info": {
            "id": request_id,
            "finish_reason": {"type": "stop"} if is_last else None,
            "prompt_tokens": 100,
            "completion_tokens": token_idx + 1,
            "cached_tokens": 0,
        },
    }


async def fake_engine_stream(n_tokens: int, request_id: str = "req-abc123"):
    """Simulate SGLang engine.async_generate with stream=True."""
    for i in range(n_tokens):
        yield make_fake_sglang_response(i, n_tokens, request_id)


# ============================================================
# CURRENT implementation (mirrors decode_handler.py exactly)
# ============================================================
async def current_process_token_stream(stream_source, context_stopped_fn):
    """Exact replica of DecodeWorkerHandler._process_token_stream."""
    request_id_future = asyncio.Future()

    # Simulated cancellation monitor (background task)
    async def _monitor():
        try:
            await request_id_future
            # Would normally await context.async_killed_or_stopped()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise

    cancel_task = asyncio.create_task(_monitor())

    try:
        async for res in stream_source:
            # Extract SGLang request ID from first response
            if not request_id_future.done():
                meta_info = res.get("meta_info", {})
                sglang_request_id = meta_info.get("id")
                if sglang_request_id:
                    request_id_future.set_result(sglang_request_id)

            out = {}
            finish_reason = res["meta_info"]["finish_reason"]
            if finish_reason:
                out["finish_reason"] = finish_reason["type"]

            output_ids = res.get("output_ids", [])
            if not output_ids and not finish_reason:
                if not context_stopped_fn():
                    yield {"finish_reason": "error", "token_ids": []}
                break

            out["token_ids"] = output_ids
            if finish_reason:
                input_tokens = res["meta_info"]["prompt_tokens"]
                completion_tokens = res["meta_info"]["completion_tokens"]
                cached_tokens = res["meta_info"]["cached_tokens"]
                prefill_prompt_tokens_details = None
                if cached_tokens is not None and cached_tokens > 0:
                    prefill_prompt_tokens_details = {"cached_tokens": cached_tokens}
                out["completion_usage"] = {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": input_tokens + completion_tokens,
                    "prompt_tokens_details": prefill_prompt_tokens_details,
                }
            if not context_stopped_fn():
                yield out
    finally:
        cancel_task.cancel()
        try:
            await cancel_task
        except asyncio.CancelledError:
            pass


# ============================================================
# OPTIMIZED implementation
# ============================================================
async def optimized_process_token_stream(stream_source, context_stopped_fn):
    """Optimized version - eliminates overhead while preserving behavior."""
    sglang_request_id = None

    async for res in stream_source:
        meta_info = res["meta_info"]

        # Track request ID (simple variable, no Future)
        if sglang_request_id is None:
            sglang_request_id = meta_info.get("id")

        finish_reason = meta_info["finish_reason"]
        output_ids = res.get("output_ids")

        # Fast path: no output_ids and not finished = error
        if not output_ids and not finish_reason:
            if not context_stopped_fn():
                yield {"finish_reason": "error", "token_ids": []}
            return

        # Build output dict - minimize allocations
        if finish_reason:
            cached_tokens = meta_info["cached_tokens"]
            input_tokens = meta_info["prompt_tokens"]
            completion_tokens = meta_info["completion_tokens"]
            yield {
                "token_ids": output_ids,
                "finish_reason": finish_reason["type"],
                "completion_usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": input_tokens + completion_tokens,
                    "prompt_tokens_details": {"cached_tokens": cached_tokens}
                    if cached_tokens
                    else None,
                },
            }
        elif not context_stopped_fn():
            yield {"token_ids": output_ids}


# ============================================================
# MINIMAL implementation (absolute minimum work)
# ============================================================
async def minimal_process_token_stream(stream_source, context_stopped_fn):
    """Minimal: just forward output_ids with zero overhead."""
    async for res in stream_source:
        meta = res["meta_info"]
        fr = meta["finish_reason"]
        if fr:
            yield {
                "token_ids": res["output_ids"],
                "finish_reason": fr["type"],
                "completion_usage": {
                    "prompt_tokens": meta["prompt_tokens"],
                    "completion_tokens": meta["completion_tokens"],
                    "total_tokens": meta["prompt_tokens"] + meta["completion_tokens"],
                },
            }
        else:
            yield {"token_ids": res["output_ids"]}


# ============================================================
# Per-request setup benchmark (the generate() preamble)
# ============================================================
def bench_generate_preamble(n: int):
    """Measure everything that happens in generate() before engine.async_generate."""
    times = []
    request = {
        "sampling_options": {"temperature": 0.7, "top_p": 0.9, "top_k": 50},
        "stop_conditions": {"max_tokens": 512, "ignore_eos": False},
        "token_ids": [1, 2, 3, 4, 5] * 100,
        "bootstrap_info": {
            "bootstrap_host": "10.0.0.1",
            "bootstrap_port": 8080,
            "bootstrap_room": 123456789,
        },
        "routing": {"dp_rank": 0},
    }
    skip_tokenizer_init = True
    enable_trace = False

    for _ in range(n):
        start = time.perf_counter_ns()

        # 1. _build_sampling_params
        sampling_opts = request.get("sampling_options", {})
        stop_conditions = request.get("stop_conditions", {})
        param_mapping = {
            "temperature": sampling_opts.get("temperature"),
            "top_p": sampling_opts.get("top_p"),
            "top_k": sampling_opts.get("top_k"),
            "max_new_tokens": stop_conditions.get("max_tokens"),
            "ignore_eos": stop_conditions.get("ignore_eos"),
        }
        sampling_params = {k: v for k, v in param_mapping.items() if v is not None}

        # 2. _get_input_param
        if skip_tokenizer_init:
            input_param = {"input_ids": request.get("token_ids")}
        else:
            input_param = {"prompt": "test"}

        # 3. bootstrap_info extraction
        bootstrap_info = request.get("bootstrap_info")

        # 4. trace header
        trace_header = None  # self._get_trace_header(context) if enable_trace else None

        # 5. routing extraction
        routing = request.get("routing") or {}
        dp_rank = routing.get("dp_rank")

        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)
    return times


# ============================================================
# Full per-token breakdown
# ============================================================
def bench_per_token_operations(n_tokens: int, n_iterations: int):
    """Measure each per-token operation independently."""
    res_objs = [make_fake_sglang_response(i, n_tokens) for i in range(n_tokens)]

    results = {}

    # Operation 1: res.get("meta_info", {})
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter_ns()
        for res in res_objs:
            _ = res.get("meta_info", {})
        times.append((time.perf_counter_ns() - start) / n_tokens)
    results["get_meta_info"] = times

    # Operation 2: res["meta_info"]["finish_reason"]
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter_ns()
        for res in res_objs:
            _ = res["meta_info"]["finish_reason"]
        times.append((time.perf_counter_ns() - start) / n_tokens)
    results["index_finish_reason"] = times

    # Operation 3: res.get("output_ids", [])
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter_ns()
        for res in res_objs:
            _ = res.get("output_ids", [])
        times.append((time.perf_counter_ns() - start) / n_tokens)
    results["get_output_ids"] = times

    # Operation 4: out = {} (empty dict creation)
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter_ns()
        for _ in res_objs:
            out = {}
        times.append((time.perf_counter_ns() - start) / n_tokens)
    results["empty_dict_create"] = times

    # Operation 5: out["token_ids"] = output_ids (dict set)
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter_ns()
        for res in res_objs:
            out = {"token_ids": res["output_ids"]}
        times.append((time.perf_counter_ns() - start) / n_tokens)
    results["dict_set_token_ids"] = times

    # Operation 6: request_id_future.done() check
    fut = asyncio.Future()
    fut.set_result("test")
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter_ns()
        for _ in res_objs:
            _ = fut.done()
        times.append((time.perf_counter_ns() - start) / n_tokens)
    results["future_done_check"] = times

    # Operation 7: context.is_stopped() (simulated)
    def is_stopped():
        return False
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter_ns()
        for _ in res_objs:
            _ = is_stopped()
        times.append((time.perf_counter_ns() - start) / n_tokens)
    results["is_stopped_check"] = times

    # Operation 8: Full typical token dict build (non-final)
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter_ns()
        for res in res_objs[:-1]:  # Non-final tokens
            out = {}
            _ = res["meta_info"]["finish_reason"]
            output_ids = res.get("output_ids", [])
            out["token_ids"] = output_ids
        times.append((time.perf_counter_ns() - start) / max(n_tokens - 1, 1))
    results["full_nonfinal_token"] = times

    # Operation 9: Full final token dict build
    final_res = res_objs[-1]
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter_ns()
        out = {}
        fr = final_res["meta_info"]["finish_reason"]
        out["finish_reason"] = fr["type"]
        out["token_ids"] = final_res["output_ids"]
        input_tokens = final_res["meta_info"]["prompt_tokens"]
        completion_tokens = final_res["meta_info"]["completion_tokens"]
        cached_tokens = final_res["meta_info"]["cached_tokens"]
        out["completion_usage"] = {
            "prompt_tokens": input_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": input_tokens + completion_tokens,
            "prompt_tokens_details": None,
        }
        times.append(time.perf_counter_ns() - start)
    results["full_final_token"] = times

    return results


async def bench_full_stream(n_requests: int, tokens_per_request: int, impl: str):
    """End-to-end stream processing benchmark."""
    if impl == "current":
        processor = current_process_token_stream
    elif impl == "optimized":
        processor = optimized_process_token_stream
    else:
        processor = minimal_process_token_stream

    def not_stopped():
        return False

    times = []
    token_counts = []
    for _ in range(n_requests):
        stream = fake_engine_stream(tokens_per_request)
        count = 0
        start = time.perf_counter_ns()
        async for out in processor(stream, not_stopped):
            count += 1
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)
        token_counts.append(count)
    return times, token_counts


def report(name, times_ns, per_unit=""):
    times_us = [t / 1000 for t in times_ns]
    sorted_t = sorted(times_us)
    p50 = statistics.median(times_us)
    p99 = sorted_t[int(len(sorted_t) * 0.99)] if len(sorted_t) >= 20 else sorted_t[-1]
    mean = statistics.mean(times_us)
    suffix = f" per {per_unit}" if per_unit else ""
    print(f"    {name:<30s}  mean={mean:>8.3f}  p50={p50:>8.3f}  p99={p99:>8.3f} us{suffix}")


async def main():
    N_REQUESTS = 5000
    TOKENS = 128

    print(f"=== Decode Handler Hot Path Analysis ===")
    print(f"    n_requests={N_REQUESTS}, tokens={TOKENS}")
    print()

    # 1. Per-request preamble
    print("[1] generate() preamble (before engine.async_generate)")
    report("preamble", bench_generate_preamble(N_REQUESTS), "request")
    print()

    # 2. Per-token operation breakdown
    print(f"[2] Per-token operation breakdown (cost per token, {TOKENS} tokens)")
    token_ops = bench_per_token_operations(TOKENS, 1000)
    for name, times in token_ops.items():
        report(name, times, "token")
    print()

    # 3. Full stream processing comparison
    print(f"[3] Full stream processing ({TOKENS} tokens per request)")
    for impl in ["current", "optimized", "minimal"]:
        times, counts = await bench_full_stream(N_REQUESTS, TOKENS, impl)
        report(f"{impl}", times, "request")
        per_token = [t / c for t, c in zip(times, counts) if c > 0]
        report(f"  per-token", per_token, "token")
    print()

    # 4. Concurrent stream processing
    print(f"[4] Concurrent stream processing (100 concurrent, {TOKENS} tokens each)")
    for impl in ["current", "optimized", "minimal"]:
        if impl == "current":
            processor = current_process_token_stream
        elif impl == "optimized":
            processor = optimized_process_token_stream
        else:
            processor = minimal_process_token_stream

        def not_stopped():
            return False

        async def single_request():
            stream = fake_engine_stream(TOKENS)
            count = 0
            start = time.perf_counter_ns()
            async for _ in processor(stream, not_stopped):
                count += 1
            return time.perf_counter_ns() - start

        tasks = [asyncio.create_task(single_request()) for _ in range(100)]
        all_times = await asyncio.gather(*tasks)
        report(f"{impl}", list(all_times), "request")
    print()


if __name__ == "__main__":
    asyncio.run(main())
