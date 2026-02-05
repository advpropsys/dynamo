"""Plot e2e_ttft_sync.jsonl vs e2e_ttft_spawn_blocking.jsonl → comparison PNG"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR = Path(__file__).parent
OUTPUT = DIR / "e2e_ttft_comparison.png"


def load(label):
    path = DIR / f"e2e_ttft_{label}.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    sync = load("sync")
    sb = load("spawn_blocking")

    concurrencies = sorted(set(r["concurrency"] for r in sync))
    isls = sorted(set(r["isl"] for r in sync))

    fig, axes = plt.subplots(len(concurrencies), 2, figsize=(16, 5 * len(concurrencies)),
                             squeeze=False)

    for ci, conc in enumerate(concurrencies):
        s_rows = sorted([r for r in sync if r["concurrency"] == conc], key=lambda r: r["isl"])
        sb_rows = sorted([r for r in sb if r["concurrency"] == conc], key=lambda r: r["isl"])

        x = np.arange(len(s_rows))
        w = 0.35
        labels = [str(r["isl"]) for r in s_rows]

        # P50
        ax = axes[ci][0]
        ax.bar(x - w/2, [r["ttft_p50_ms"] for r in s_rows], w,
               label="sync (blocks runtime)", color="#e74c3c", alpha=0.85)
        ax.bar(x + w/2, [r["ttft_p50_ms"] for r in sb_rows], w,
               label="spawn_blocking", color="#2ecc71", alpha=0.85)
        ax.set_ylabel("TTFT (ms)")
        ax.set_title(f"P50 TTFT — concurrency={conc}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # P99
        ax = axes[ci][1]
        ax.bar(x - w/2, [r["ttft_p99_ms"] for r in s_rows], w,
               label="sync (blocks runtime)", color="#e74c3c", alpha=0.85)
        ax.bar(x + w/2, [r["ttft_p99_ms"] for r in sb_rows], w,
               label="spawn_blocking", color="#2ecc71", alpha=0.85)
        ax.set_ylabel("TTFT (ms)")
        ax.set_title(f"P99 TTFT — concurrency={conc}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    axes[-1][0].set_xlabel("Input Sequence Length (tokens)")
    axes[-1][1].set_xlabel("Input Sequence Length (tokens)")

    fig.suptitle(
        "E2E TTFT: sync tokenization vs spawn_blocking\n"
        "Qwen/Qwen3-0.6B + sglang + dynamo frontend",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=150)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
