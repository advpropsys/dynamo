"""
Micro-benchmark: Measure per-request overhead in sglang decode/prefill handlers.

This benchmarks the Python-side overhead that is added on TOP of the actual
SGLang engine.async_generate() call. Specifically:
1. asyncio.Future creation + .done() check per token
2. asyncio.create_task() for cancellation monitor per request
3. _build_sampling_params dict comprehension
4. _get_input_param tokenizer lookup
5. _consume_results create_task + immediate await pattern (prefill)
6. Dict construction per yielded token

Usage:
    python benchmarks/sglang_handler_overhead_bench.py
"""

import asyncio
import random
import statistics
import time


async def bench_future_pattern(n_requests: int, tokens_per_request: int):
    """Measure overhead of asyncio.Future create + .done() check per token."""
    times = []
    for _ in range(n_requests):
        start = time.perf_counter_ns()
        fut = asyncio.Future()
        for i in range(tokens_per_request):
            if not fut.done():
                if i == 0:
                    fut.set_result(f"req-{random.randint(0, 2**63-1)}")
            # Simulate the dict access pattern from _process_token_stream
            _ = fut.done()
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)
    return times


async def bench_cancellation_monitor(n_requests: int):
    """Measure overhead of creating + cancelling a background task per request."""
    times = []

    async def _dummy_monitor(fut, event):
        try:
            await fut
            await event.wait()
        except asyncio.CancelledError:
            raise

    for _ in range(n_requests):
        start = time.perf_counter_ns()
        fut = asyncio.Future()
        event = asyncio.Event()

        task = asyncio.create_task(_dummy_monitor(fut, event))
        fut.set_result("test-id")

        # Simulate: generation done, cancel the monitor
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)
    return times


async def bench_consume_results_pattern(n_requests: int, tokens_per_request: int):
    """Measure the _consume_results pattern: create_task wrapping an async for + immediate await."""
    times = []

    async def _fake_engine_stream(n_tokens):
        for i in range(n_tokens):
            yield {
                "meta_info": {
                    "id": f"req-{i}",
                    "finish_reason": {"type": "stop"} if i == n_tokens - 1 else None,
                    "prompt_tokens": 100,
                    "completion_tokens": i + 1,
                    "cached_tokens": 0,
                },
                "output_ids": [i + 1000],
            }

    async def _consume(gen):
        request_id_future = asyncio.Future()
        async for res in gen:
            if not request_id_future.done():
                meta = res.get("meta_info", {})
                rid = meta.get("id")
                if rid:
                    request_id_future.set_result(rid)

    for _ in range(n_requests):
        start = time.perf_counter_ns()
        gen = _fake_engine_stream(tokens_per_request)
        task = asyncio.create_task(_consume(gen))
        await task
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)
    return times


async def bench_token_dict_construction(n_requests: int, tokens_per_request: int):
    """Measure per-token dict construction overhead in _process_token_stream."""
    times = []
    for _ in range(n_requests):
        start = time.perf_counter_ns()
        for tok_idx in range(tokens_per_request):
            is_last = tok_idx == tokens_per_request - 1
            out = {}
            if is_last:
                out["finish_reason"] = "stop"
                out["completion_usage"] = {
                    "prompt_tokens": 100,
                    "completion_tokens": tok_idx + 1,
                    "total_tokens": 100 + tok_idx + 1,
                    "prompt_tokens_details": None,
                }
            out["token_ids"] = [tok_idx + 1000]
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)
    return times


async def bench_sampling_params(n_requests: int):
    """Measure _build_sampling_params overhead."""
    times = []
    request = {
        "sampling_options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
        },
        "stop_conditions": {
            "max_tokens": 512,
            "ignore_eos": False,
        },
    }
    for _ in range(n_requests):
        start = time.perf_counter_ns()
        sampling_opts = request.get("sampling_options", {})
        stop_conditions = request.get("stop_conditions", {})
        param_mapping = {
            "temperature": sampling_opts.get("temperature"),
            "top_p": sampling_opts.get("top_p"),
            "top_k": sampling_opts.get("top_k"),
            "max_new_tokens": stop_conditions.get("max_tokens"),
            "ignore_eos": stop_conditions.get("ignore_eos"),
        }
        _ = {k: v for k, v in param_mapping.items() if v is not None}
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)
    return times


async def bench_concurrent_cancellation_monitors(concurrency: int):
    """Measure overhead when many cancellation monitors are active simultaneously."""

    async def _monitor(fut, event):
        try:
            await fut
            await event.wait()
        except asyncio.CancelledError:
            raise

    futs = [asyncio.Future() for _ in range(concurrency)]
    events = [asyncio.Event() for _ in range(concurrency)]
    tasks = []

    start = time.perf_counter_ns()
    for i in range(concurrency):
        t = asyncio.create_task(_monitor(futs[i], events[i]))
        tasks.append(t)

    # Let them all be scheduled
    await asyncio.sleep(0)

    # Set all futures
    for i, f in enumerate(futs):
        f.set_result(f"req-{i}")

    # Cancel all
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.perf_counter_ns() - start

    return elapsed


def report(name, times_ns, unit="ns"):
    """Print statistics for a benchmark."""
    if not times_ns:
        print(f"  {name}: no data")
        return
    times_us = [t / 1000 for t in times_ns]
    print(f"  {name}:")
    print(f"    mean:  {statistics.mean(times_us):>10.2f} us")
    print(f"    p50:   {statistics.median(times_us):>10.2f} us")
    if len(times_us) >= 20:
        sorted_t = sorted(times_us)
        p99_idx = int(len(sorted_t) * 0.99)
        print(f"    p99:   {sorted_t[p99_idx]:>10.2f} us")
    print(f"    min:   {min(times_us):>10.2f} us")
    print(f"    max:   {max(times_us):>10.2f} us")


async def main():
    N_REQUESTS = 10000
    TOKENS_PER_REQUEST = 128

    print(f"=== SGLang Handler Overhead Micro-Benchmarks ===")
    print(f"    n_requests={N_REQUESTS}, tokens_per_request={TOKENS_PER_REQUEST}")
    print()

    # 1. Future pattern (per-request)
    print("[1] asyncio.Future create + .done() checks (per request)")
    times = await bench_future_pattern(N_REQUESTS, TOKENS_PER_REQUEST)
    report("future_pattern", times)
    print()

    # 2. Cancellation monitor create+cancel (per-request)
    print("[2] Cancellation monitor: create_task + cancel + await (per request)")
    times = await bench_cancellation_monitor(N_REQUESTS)
    report("cancellation_monitor", times)
    print()

    # 3. _consume_results pattern (per-request)
    print("[3] _consume_results: create_task(async for) + await (per request)")
    times = await bench_consume_results_pattern(N_REQUESTS, TOKENS_PER_REQUEST)
    report("consume_results", times)
    print()

    # 4. Token dict construction (per-request, all tokens)
    print("[4] Per-token dict construction (per request, 128 tokens)")
    times = await bench_token_dict_construction(N_REQUESTS, TOKENS_PER_REQUEST)
    report("token_dict", times)
    print()

    # 5. Sampling params construction (per-request)
    print("[5] _build_sampling_params (per request)")
    times = await bench_sampling_params(N_REQUESTS)
    report("sampling_params", times)
    print()

    # 6. Concurrent cancellation monitors (scaling test)
    print("[6] Concurrent cancellation monitors (scaling test)")
    for concurrency in [10, 100, 500, 1000, 5000]:
        elapsed = await bench_concurrent_cancellation_monitors(concurrency)
        print(f"    concurrency={concurrency:>5d}: total={elapsed/1000:.2f} us, per_task={elapsed/1000/concurrency:.2f} us")
    print()


if __name__ == "__main__":
    asyncio.run(main())
