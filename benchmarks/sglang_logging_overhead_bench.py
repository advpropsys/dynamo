"""
Micro-benchmark: Measure logging overhead on the hot path.

The decode_handler and prefill_handler both call logging.debug() on every
request and sometimes per-token. Even when debug is disabled, the string
formatting and function call overhead can be significant at high throughput.

Usage:
    python benchmarks/sglang_logging_overhead_bench.py
"""

import logging
import statistics
import time

N = 100000


def bench_debug_disabled():
    """logging.debug() when level is INFO (should be fast but still has overhead)."""
    logger = logging.getLogger("bench.disabled")
    logger.setLevel(logging.INFO)
    times = []
    for i in range(N):
        start = time.perf_counter_ns()
        logger.debug(f"New Request ID: req-{i}")
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)
    return times


def bench_debug_disabled_fstring_eager():
    """f-string is evaluated even if debug is disabled."""
    logger = logging.getLogger("bench.fstring")
    logger.setLevel(logging.INFO)
    bootstrap_info = {
        "bootstrap_host": "10.0.0.1",
        "bootstrap_port": 8080,
        "bootstrap_room": 123456789012345,
    }
    times = []
    for i in range(N):
        start = time.perf_counter_ns()
        logger.debug(
            f"Using bootstrap_info: "
            f"host={bootstrap_info['bootstrap_host']}, "
            f"port={bootstrap_info['bootstrap_port']}, "
            f"room={bootstrap_info['bootstrap_room']}"
        )
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)
    return times


def bench_debug_with_guard():
    """Using isEnabledFor guard to avoid f-string evaluation."""
    logger = logging.getLogger("bench.guarded")
    logger.setLevel(logging.INFO)
    bootstrap_info = {
        "bootstrap_host": "10.0.0.1",
        "bootstrap_port": 8080,
        "bootstrap_room": 123456789012345,
    }
    times = []
    for i in range(N):
        start = time.perf_counter_ns()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Using bootstrap_info: "
                f"host={bootstrap_info['bootstrap_host']}, "
                f"port={bootstrap_info['bootstrap_port']}, "
                f"room={bootstrap_info['bootstrap_room']}"
            )
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)
    return times


def bench_no_logging():
    """Baseline: no logging at all."""
    times = []
    for i in range(N):
        start = time.perf_counter_ns()
        # nothing
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)
    return times


def report(name, times_ns):
    times_us = [t / 1000 for t in times_ns]
    sorted_t = sorted(times_us)
    p99_idx = int(len(sorted_t) * 0.99)
    print(f"  {name}:")
    print(f"    mean: {statistics.mean(times_us):>8.3f} us")
    print(f"    p50:  {statistics.median(times_us):>8.3f} us")
    print(f"    p99:  {sorted_t[p99_idx]:>8.3f} us")


def main():
    print(f"=== Logging Overhead on Hot Path (N={N}) ===")
    print()

    print("[1] Baseline (no logging call)")
    report("baseline", bench_no_logging())
    print()

    print("[2] logging.debug(f'...') with level=INFO (disabled)")
    report("debug_disabled", bench_debug_disabled())
    print()

    print("[3] logging.debug(f'complex format') with level=INFO")
    report("debug_fstring_eager", bench_debug_disabled_fstring_eager())
    print()

    print("[4] isEnabledFor guard + logging.debug(f'...')")
    report("debug_guarded", bench_debug_with_guard())
    print()

    print("Conclusion: Even disabled debug calls with f-strings have measurable overhead.")
    print("At 1000+ requests/sec with multiple debug calls per request, this adds up.")


if __name__ == "__main__":
    main()
