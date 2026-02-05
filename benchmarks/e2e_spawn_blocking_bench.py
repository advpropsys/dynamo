"""
E2E benchmark: TTFT with sync tokenization vs spawn_blocking
Sends concurrent HTTP chat/completions requests to dynamo frontend,
measures time-to-first-token (TTFT) at varying concurrencies and ISLs.

Usage:
  # 1. Start sglang worker (in another terminal):
  #    PYTHONPATH=components/src .venv/bin/python -m dynamo.sglang \
  #      --model Qwen/Qwen3-0.6B --enable-metrics
  #
  # 2. Start frontend (in another terminal):
  #    PYTHONPATH=components/src .venv/bin/python -m dynamo.frontend \
  #      --http-port 8080
  #
  # 3. Run this benchmark:
  #    .venv/bin/python benchmarks/e2e_spawn_blocking_bench.py \
  #      --label sync   # or --label spawn_blocking
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp

BASE_URL = "http://localhost:8080"
MODEL = "Qwen/Qwen3-0.6B"
ISLS = [128, 512, 1024, 2048, 4096, 8192]
CONCURRENCIES = [1, 4, 8, 16, 32, 64]
SAMPLES_PER_CONFIG = 100
MAX_TOKENS = 1  # We only care about TTFT, not full generation
OUTPUT_DIR = Path(__file__).parent


def generate_prompt(isl: int) -> str:
    """Generate a prompt that will tokenize to approximately `isl` tokens.
    ~4 chars per token is a rough average for English text."""
    base = "The quick brown fox jumps over the lazy dog. "
    repeats = max(1, (isl * 4) // len(base))
    return base * repeats


async def measure_ttft(session: aiohttp.ClientSession, prompt: str) -> float:
    """Send a streaming request and measure time to first token."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "temperature": 0.0,
    }

    t0 = time.perf_counter_ns()
    ttft = None

    try:
        async with session.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if decoded.startswith("data:") and ttft is None:
                    data = decoded[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if chunk.get("choices", [{}])[0].get("delta", {}).get(
                            "content"
                        ) is not None:
                            ttft = (time.perf_counter_ns() - t0) / 1e6  # ms
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        print(f"  Request error: {e}")
        return -1.0

    return ttft if ttft is not None else (time.perf_counter_ns() - t0) / 1e6


async def bench_config(
    isl: int, concurrency: int, samples: int
) -> list[float]:
    """Run `samples` requests at given concurrency, return TTFT list."""
    prompt = generate_prompt(isl)
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    results = []

    async with aiohttp.ClientSession(connector=connector) as session:
        async def task():
            async with sem:
                return await measure_ttft(session, prompt)

        tasks = [asyncio.create_task(task()) for _ in range(samples)]
        for t in asyncio.as_completed(tasks):
            lat = await t
            if lat > 0:
                results.append(lat)

    return results


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = int(p / 100.0 * (len(s) - 1) + 0.5)
    return s[min(idx, len(s) - 1)]


async def run(label: str):
    print(f"E2E TTFT benchmark — label={label}")
    print(f"ISLs: {ISLS}")
    print(f"Concurrencies: {CONCURRENCIES}")
    print(f"Samples per config: {SAMPLES_PER_CONFIG}")

    # Warmup — hit every ISL at concurrency to stabilize JIT, caches, thread pools
    print("\nWarmup (30 requests across ISLs)...")
    connector = aiohttp.TCPConnector(limit=20)
    async with aiohttp.ClientSession(connector=connector) as session:
        for isl in ISLS:
            prompt = generate_prompt(isl)
            for _ in range(5):
                await measure_ttft(session, prompt)

    jsonl_path = OUTPUT_DIR / f"e2e_ttft_{label}.jsonl"
    lines = []

    for conc in CONCURRENCIES:
        print(f"\n=== concurrency={conc} ===")
        for isl in ISLS:
            lats = await bench_config(isl, conc, SAMPLES_PER_CONFIG)
            if not lats:
                print(f"  ISL {isl:>6}: all requests failed")
                continue

            p50 = percentile(lats, 50)
            p99 = percentile(lats, 99)
            mean = sum(lats) / len(lats)

            print(
                f"  ISL {isl:>6}: p50={p50:.1f}ms  p99={p99:.1f}ms  "
                f"mean={mean:.1f}ms  n={len(lats)}"
            )

            row = {
                "label": label,
                "isl": isl,
                "concurrency": conc,
                "num_samples": len(lats),
                "ttft_p50_ms": round(p50, 2),
                "ttft_p99_ms": round(p99, 2),
                "ttft_mean_ms": round(mean, 2),
            }
            lines.append(json.dumps(row))

    with open(jsonl_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults written to {jsonl_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        required=True,
        help="Label for this run (e.g. 'sync' or 'spawn_blocking')",
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()

    _set_globals(args.model, args.base_url)
    asyncio.run(run(args.label))


def _set_globals(model, base_url):
    global MODEL, BASE_URL
    MODEL = model
    BASE_URL = base_url


if __name__ == "__main__":
    main()
