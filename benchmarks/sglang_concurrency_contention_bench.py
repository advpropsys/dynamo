"""
Micro-benchmark: Measure event-loop contention under high concurrency.

Simulates what happens when many decode requests are active simultaneously,
each with their own cancellation monitor task. Measures how event loop
scheduling latency degrades as concurrency increases.

Also measures the overhead of the _consume_results pattern (prefill) under load.

Usage:
    python benchmarks/sglang_concurrency_contention_bench.py
"""

import asyncio
import statistics
import time


async def simulate_decode_request(tokens: int, results: list):
    """Simulate a single decode request with the current handler pattern.

    Replicates the pattern from decode_handler.py:
    - asyncio.Future for request ID
    - _cancellation_monitor context manager (background task)
    - async for over engine stream
    - dict construction per token
    """
    request_id_future = asyncio.Future()

    async def _monitor():
        try:
            await request_id_future
            await asyncio.Event().wait()  # wait forever (cancelled on completion)
        except asyncio.CancelledError:
            raise

    cancel_task = asyncio.create_task(_monitor())

    ttft_start = time.perf_counter_ns()
    first_token_time = None
    itl_times = []
    last_token_time = None

    try:
        for i in range(tokens):
            # Simulate engine yielding a token (tiny await to yield control)
            await asyncio.sleep(0)

            now = time.perf_counter_ns()
            if first_token_time is None:
                first_token_time = now
                if not request_id_future.done():
                    request_id_future.set_result(f"req-{id(request_id_future)}")
            elif last_token_time is not None:
                itl_times.append(now - last_token_time)

            last_token_time = now

            # Simulate dict construction (mirrors _process_token_stream)
            out = {}
            if i == tokens - 1:
                out["finish_reason"] = "stop"
                out["completion_usage"] = {
                    "prompt_tokens": 100,
                    "completion_tokens": i + 1,
                    "total_tokens": 100 + i + 1,
                }
            out["token_ids"] = [i + 1000]
    finally:
        cancel_task.cancel()
        try:
            await cancel_task
        except asyncio.CancelledError:
            pass

    ttft = (first_token_time - ttft_start) if first_token_time else 0
    results.append({
        "ttft_ns": ttft,
        "itl_mean_ns": statistics.mean(itl_times) if itl_times else 0,
        "itl_p99_ns": sorted(itl_times)[int(len(itl_times) * 0.99)] if len(itl_times) >= 10 else 0,
    })


async def simulate_decode_request_optimized(tokens: int, results: list):
    """Simulate decode request WITHOUT the cancellation monitor pattern.

    Eliminates:
    - asyncio.Future for request ID (use simple variable)
    - No background cancellation task
    - Same dict construction
    """
    ttft_start = time.perf_counter_ns()
    first_token_time = None
    itl_times = []
    last_token_time = None
    sglang_request_id = None

    for i in range(tokens):
        await asyncio.sleep(0)

        now = time.perf_counter_ns()
        if first_token_time is None:
            first_token_time = now
            sglang_request_id = f"req-{i}"
        elif last_token_time is not None:
            itl_times.append(now - last_token_time)

        last_token_time = now

        out = {}
        if i == tokens - 1:
            out["finish_reason"] = "stop"
            out["completion_usage"] = {
                "prompt_tokens": 100,
                "completion_tokens": i + 1,
                "total_tokens": 100 + i + 1,
            }
        out["token_ids"] = [i + 1000]

    ttft = (first_token_time - ttft_start) if first_token_time else 0
    results.append({
        "ttft_ns": ttft,
        "itl_mean_ns": statistics.mean(itl_times) if itl_times else 0,
        "itl_p99_ns": sorted(itl_times)[int(len(itl_times) * 0.99)] if len(itl_times) >= 10 else 0,
    })


async def run_concurrent_test(concurrency: int, tokens: int, optimized: bool = False):
    """Run concurrent decode simulations and measure scheduling overhead."""
    results = []
    handler = simulate_decode_request_optimized if optimized else simulate_decode_request
    tasks = [asyncio.create_task(handler(tokens, results)) for _ in range(concurrency)]

    start = time.perf_counter_ns()
    await asyncio.gather(*tasks)
    total_ns = time.perf_counter_ns() - start

    ttft_values = [r["ttft_ns"] for r in results]
    itl_mean_values = [r["itl_mean_ns"] for r in results if r["itl_mean_ns"] > 0]
    itl_p99_values = [r["itl_p99_ns"] for r in results if r["itl_p99_ns"] > 0]

    return {
        "concurrency": concurrency,
        "total_us": total_ns / 1000,
        "ttft_mean_us": statistics.mean(ttft_values) / 1000 if ttft_values else 0,
        "ttft_p99_us": sorted(ttft_values)[int(len(ttft_values) * 0.99)] / 1000 if len(ttft_values) >= 10 else 0,
        "itl_mean_us": statistics.mean(itl_mean_values) / 1000 if itl_mean_values else 0,
        "itl_p99_mean_us": statistics.mean(itl_p99_values) / 1000 if itl_p99_values else 0,
    }


async def main():
    TOKENS = 64
    print(f"=== Event Loop Contention Under Concurrency ===")
    print(f"    tokens_per_request={TOKENS}")
    print()

    concurrencies = [1, 10, 50, 100, 250, 500, 1000]

    print(f"{'':>4} {'Concurrency':>12} | {'TTFT mean':>12} {'TTFT p99':>12} | {'ITL mean':>12} {'ITL p99 mean':>14} | {'Total':>12}")
    print(f"{'':>4} {'':>12} | {'(us)':>12} {'(us)':>12} | {'(us)':>12} {'(us)':>14} | {'(us)':>12}")
    print("-" * 100)

    # Current pattern (with cancellation monitor)
    print("  CURRENT (with cancellation monitor per request):")
    for c in concurrencies:
        r = await run_concurrent_test(c, TOKENS, optimized=False)
        print(f"{'':>4} {r['concurrency']:>12d} | {r['ttft_mean_us']:>12.2f} {r['ttft_p99_us']:>12.2f} | {r['itl_mean_us']:>12.2f} {r['itl_p99_mean_us']:>14.2f} | {r['total_us']:>12.0f}")

    print()
    print("  OPTIMIZED (no cancellation monitor task):")
    for c in concurrencies:
        r = await run_concurrent_test(c, TOKENS, optimized=True)
        print(f"{'':>4} {r['concurrency']:>12d} | {r['ttft_mean_us']:>12.2f} {r['ttft_p99_us']:>12.2f} | {r['itl_mean_us']:>12.2f} {r['itl_p99_mean_us']:>14.2f} | {r['total_us']:>12.0f}")

    print()
    print("  Note: These measure pure Python event loop overhead only.")
    print("  Real-world impact is multiplied by GIL contention + Rust bridge overhead.")


if __name__ == "__main__":
    asyncio.run(main())
