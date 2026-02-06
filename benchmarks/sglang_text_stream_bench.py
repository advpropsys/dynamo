"""
Micro-benchmark: _process_text_stream per-token overhead.

The text stream path (OpenAI format) has additional per-token costs:
- time.time() syscall per token
- String slicing (delta = text[count:])
- Nested dict construction (choice_data, response)
- config.server_args.served_model_name attribute access

Usage:
    python benchmarks/sglang_text_stream_bench.py
"""

import statistics
import time

N_TOKENS = 128
N_ITER = 5000


def make_text_response(token_idx, n_tokens, cumulative_text):
    """Fake SGLang response in text mode (cumulative text)."""
    is_last = token_idx == n_tokens - 1
    return {
        "text": cumulative_text,
        "index": 0,
        "meta_info": {
            "id": "req-abc123",
            "finish_reason": {"type": "stop"} if is_last else None,
        },
    }


def bench_text_stream_current():
    """Current _process_text_stream per-token logic."""
    # Pre-build fake responses with cumulative text
    words = ["Hello", " world", " this", " is", " a", " test", " of", " the"]
    responses = []
    cum_text = ""
    for i in range(N_TOKENS):
        cum_text += words[i % len(words)]
        responses.append(make_text_response(i, N_TOKENS, cum_text))

    served_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    times = []

    for _ in range(N_ITER):
        count = 0
        start = time.perf_counter_ns()
        for res in responses:
            index = res.get("index", 0)
            text = res.get("text", "")
            finish_reason = res["meta_info"]["finish_reason"]
            finish_reason_type = finish_reason["type"] if finish_reason else None
            next_count = len(text)
            delta = text[count:]

            choice_data = {
                "index": index,
                "delta": {"role": "assistant", "content": delta},
                "finish_reason": finish_reason_type,
            }
            response = {
                "id": res["meta_info"]["id"],
                "created": int(time.time()),  # <-- syscall per token!
                "choices": [choice_data],
                "model": served_model_name,
                "object": "chat.completion.chunk",
            }
            count = next_count
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)
    return times


def bench_text_stream_optimized():
    """Optimized: cache time.time(), minimize dict nesting."""
    words = ["Hello", " world", " this", " is", " a", " test", " of", " the"]
    responses = []
    cum_text = ""
    for i in range(N_TOKENS):
        cum_text += words[i % len(words)]
        responses.append(make_text_response(i, N_TOKENS, cum_text))

    served_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    times = []

    for _ in range(N_ITER):
        count = 0
        created_time = int(time.time())  # Cache once per request, not per token
        start = time.perf_counter_ns()
        for res in responses:
            text = res.get("text", "")
            meta = res["meta_info"]
            fr = meta["finish_reason"]
            delta = text[count:]
            count = len(text)

            response = {
                "id": meta["id"],
                "created": created_time,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": delta},
                    "finish_reason": fr["type"] if fr else None,
                }],
                "model": served_model_name,
                "object": "chat.completion.chunk",
            }
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)
    return times


def bench_time_time():
    """Measure time.time() vs cached time."""
    times_syscall = []
    times_cached = []

    for _ in range(N_ITER):
        start = time.perf_counter_ns()
        for _ in range(N_TOKENS):
            _ = int(time.time())
        elapsed = time.perf_counter_ns() - start
        times_syscall.append(elapsed / N_TOKENS)

    cached = int(time.time())
    for _ in range(N_ITER):
        start = time.perf_counter_ns()
        for _ in range(N_TOKENS):
            _ = cached
        elapsed = time.perf_counter_ns() - start
        times_cached.append(elapsed / N_TOKENS)

    return times_syscall, times_cached


def bench_string_slice():
    """Measure cumulative text slicing overhead."""
    words = ["Hello", " world", " this", " is", " a", " test", " of", " the"]
    cum_text = ""
    texts = []
    for i in range(N_TOKENS):
        cum_text += words[i % len(words)]
        texts.append(cum_text)

    times = []
    for _ in range(N_ITER):
        count = 0
        start = time.perf_counter_ns()
        for text in texts:
            delta = text[count:]
            count = len(text)
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed / N_TOKENS)
    return times


def report(name, times_ns, unit="us"):
    times_us = [t / 1000 for t in times_ns]
    sorted_t = sorted(times_us)
    p50 = statistics.median(times_us)
    p99 = sorted_t[int(len(sorted_t) * 0.99)] if len(sorted_t) >= 20 else sorted_t[-1]
    mean = statistics.mean(times_us)
    print(f"    {name:<35s}  mean={mean:>8.3f}  p50={p50:>8.3f}  p99={p99:>8.3f} us")


def main():
    print(f"=== _process_text_stream Hot Path Analysis ===")
    print(f"    n_tokens={N_TOKENS}, n_iterations={N_ITER}")
    print()

    # 1. time.time() per-token cost
    print("[1] time.time() per token vs cached")
    syscall_times, cached_times = bench_time_time()
    report("int(time.time()) per token", syscall_times)
    report("cached_time per token", cached_times)
    print()

    # 2. String slicing (cumulative text)
    print("[2] Cumulative text slicing (delta = text[count:])")
    report("string_slice per token", bench_string_slice())
    print()

    # 3. Full text stream comparison
    print(f"[3] Full text stream processing ({N_TOKENS} tokens per request)")
    times_current = bench_text_stream_current()
    times_optimized = bench_text_stream_optimized()
    report("current (per request)", times_current)
    report("  per token", [t / N_TOKENS for t in times_current])
    report("optimized (per request)", times_optimized)
    report("  per token", [t / N_TOKENS for t in times_optimized])

    speedup = statistics.mean(times_current) / statistics.mean(times_optimized)
    print(f"\n    Speedup: {speedup:.2f}x")
    print()


if __name__ == "__main__":
    main()
