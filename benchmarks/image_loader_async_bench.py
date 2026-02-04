# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark: Thread offloading for PIL image operations in async context.

Compares:
1. OLD: Image.open in thread, convert inline (blocks event loop)
2. NEW: Both open + convert in thread (fully non-blocking)

Measures both raw operation time and async event loop blocking impact.
"""

import asyncio
import io
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Tuple

from PIL import Image
import numpy as np


@dataclass
class BenchResult:
    name: str
    image_size: str
    iterations: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    event_loop_blocked_ms: float  # Time the event loop was blocked


def generate_test_image(width: int, height: int) -> bytes:
    """Generate a random RGB image as JPEG bytes."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ============================================================================
# Implementation variants to benchmark
# ============================================================================

def _open_only(image_data: io.BytesIO) -> Image.Image:
    """Just open the image (old approach - open in thread)."""
    return Image.open(image_data)


def _open_and_convert(image_data: io.BytesIO) -> Image.Image:
    """Open and convert to RGB (new approach - all in thread)."""
    img = Image.open(image_data)
    return img.convert("RGB")


async def approach_old(image_bytes: bytes, executor: ThreadPoolExecutor) -> Image.Image:
    """OLD: Open in thread, convert inline (blocks event loop during convert)."""
    loop = asyncio.get_running_loop()
    image_data = io.BytesIO(image_bytes)

    # Open in thread - non-blocking
    img = await loop.run_in_executor(executor, _open_only, image_data)

    # Convert inline - BLOCKS event loop
    return img.convert("RGB")


async def approach_new(image_bytes: bytes, executor: ThreadPoolExecutor) -> Image.Image:
    """NEW: Both open and convert in thread (fully non-blocking)."""
    loop = asyncio.get_running_loop()
    image_data = io.BytesIO(image_bytes)

    # Both operations in thread - non-blocking
    return await loop.run_in_executor(executor, _open_and_convert, image_data)


async def approach_sync_baseline(image_bytes: bytes, executor: ThreadPoolExecutor) -> Image.Image:
    """BASELINE: Everything inline (fully blocks event loop)."""
    image_data = io.BytesIO(image_bytes)
    img = Image.open(image_data)
    return img.convert("RGB")


# ============================================================================
# Event loop blocking measurement
# ============================================================================

async def measure_event_loop_blocking(
    func: Callable,
    image_bytes: bytes,
    executor: ThreadPoolExecutor,
    check_interval_ms: float = 0.5,
) -> Tuple[Image.Image, float]:
    """
    Measure how long the event loop is blocked during an operation.

    Spawns a background task that tries to run every check_interval_ms.
    The max gap between runs indicates event loop blocking.
    """
    gaps = []
    stop_flag = asyncio.Event()

    async def monitor():
        last_time = time.perf_counter()
        while not stop_flag.is_set():
            await asyncio.sleep(check_interval_ms / 1000)
            now = time.perf_counter()
            gap_ms = (now - last_time) * 1000 - check_interval_ms
            if gap_ms > 0:
                gaps.append(gap_ms)
            last_time = now

    monitor_task = asyncio.create_task(monitor())

    try:
        result = await func(image_bytes, executor)
    finally:
        stop_flag.set()
        await monitor_task

    max_block = max(gaps) if gaps else 0
    return result, max_block


# ============================================================================
# Benchmark runner
# ============================================================================

async def run_benchmark(
    name: str,
    func: Callable,
    image_bytes: bytes,
    image_size_str: str,
    executor: ThreadPoolExecutor,
    iterations: int = 50,
    warmup: int = 5,
) -> BenchResult:
    """Run benchmark for a single approach."""

    # Warmup
    for _ in range(warmup):
        await func(image_bytes, executor)

    # Benchmark iterations
    times_ms = []
    block_times_ms = []

    for _ in range(iterations):
        start = time.perf_counter()
        _, block_time = await measure_event_loop_blocking(func, image_bytes, executor)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed_ms)
        block_times_ms.append(block_time)

    return BenchResult(
        name=name,
        image_size=image_size_str,
        iterations=iterations,
        mean_ms=statistics.mean(times_ms),
        std_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0,
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        event_loop_blocked_ms=statistics.mean(block_times_ms),
    )


async def main():
    # Test image sizes: small, medium, large, very large
    image_configs = [
        ("256x256 (~20KB)", 256, 256),
        ("512x512 (~80KB)", 512, 512),
        ("1024x1024 (~300KB)", 1024, 1024),
        ("2048x2048 (~1.2MB)", 2048, 2048),
        ("4096x4096 (~5MB)", 4096, 4096),
    ]

    approaches = [
        ("SYNC (baseline)", approach_sync_baseline),
        ("OLD (open in thread, convert inline)", approach_old),
        ("NEW (open+convert in thread)", approach_new),
    ]

    executor = ThreadPoolExecutor(max_workers=4)

    print("=" * 90)
    print("PIL Image Loading Async Benchmark")
    print("=" * 90)
    print(f"Iterations per test: 50 (after 5 warmup)")
    print()

    for size_name, width, height in image_configs:
        print(f"\n{'='*90}")
        print(f"Image Size: {size_name}")
        print(f"{'='*90}")

        # Generate test image
        image_bytes = generate_test_image(width, height)
        actual_size_kb = len(image_bytes) / 1024
        print(f"Actual JPEG size: {actual_size_kb:.1f} KB")
        print()

        print(f"{'Approach':<45} {'Mean (ms)':>10} {'Std':>8} {'Min':>8} {'Max':>8} {'EL Block':>10}")
        print("-" * 90)

        results = []
        for name, func in approaches:
            result = await run_benchmark(
                name=name,
                func=func,
                image_bytes=image_bytes,
                image_size_str=size_name,
                executor=executor,
                iterations=50,
                warmup=5,
            )
            results.append(result)
            print(
                f"{result.name:<45} "
                f"{result.mean_ms:>10.2f} "
                f"{result.std_ms:>8.2f} "
                f"{result.min_ms:>8.2f} "
                f"{result.max_ms:>8.2f} "
                f"{result.event_loop_blocked_ms:>10.2f}"
            )

        # Analysis
        old_result = results[1]
        new_result = results[2]
        overhead_ms = new_result.mean_ms - old_result.mean_ms
        block_reduction_ms = old_result.event_loop_blocked_ms - new_result.event_loop_blocked_ms

        print()
        print(f"Analysis:")
        print(f"  Thread offload overhead: {overhead_ms:+.2f} ms")
        print(f"  Event loop blocking reduction: {block_reduction_ms:+.2f} ms")
        if block_reduction_ms > abs(overhead_ms):
            print(f"  ✅ NEW approach is better (reduces blocking more than overhead)")
        elif overhead_ms < 0:
            print(f"  ✅ NEW approach is faster overall")
        else:
            print(f"  ⚠️  OLD approach has lower latency, but blocks event loop more")

    executor.shutdown(wait=True)

    print()
    print("=" * 90)
    print("Conclusion:")
    print("=" * 90)
    print("""
For high-concurrency serving:
- Event loop blocking directly impacts tail latency for OTHER requests
- Even small blocking (1-5ms) can cascade under high load
- Thread offload overhead is typically <1ms
- NEW approach (full thread offload) is recommended for production
""")


if __name__ == "__main__":
    asyncio.run(main())
