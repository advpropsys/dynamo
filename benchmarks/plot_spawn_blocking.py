"""Plot tokenizer_spawn_blocking_results.jsonl → PNG"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).parent / "tokenizer_spawn_blocking_results.jsonl"
OUTPUT = Path(__file__).parent / "tokenizer_spawn_blocking_bench.png"


def load():
    rows = []
    with open(RESULTS) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    rows = load()
    concurrencies = sorted(set(r["concurrency"] for r in rows))
    isls = sorted(set(r["isl"] for r in rows))

    fig, axes = plt.subplots(len(concurrencies), 2, figsize=(16, 5 * len(concurrencies)),
                             squeeze=False)

    for ci, conc in enumerate(concurrencies):
        crows = [r for r in rows if r["concurrency"] == conc]
        crows.sort(key=lambda r: r["isl"])

        x = np.arange(len(crows))
        w = 0.35
        labels = [str(r["isl"]) for r in crows]

        sync_p50 = [r["sync_p50_ms"] for r in crows]
        sb_p50 = [r["spawn_blocking_p50_ms"] for r in crows]
        sync_p99 = [r["sync_p99_ms"] for r in crows]
        sb_p99 = [r["spawn_blocking_p99_ms"] for r in crows]

        ax = axes[ci][0]
        ax.bar(x - w / 2, sync_p50, w, label="sync (blocks runtime)", color="#e74c3c", alpha=0.85)
        ax.bar(x + w / 2, sb_p50, w, label="spawn_blocking", color="#2ecc71", alpha=0.85)
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"P50 — concurrency={conc}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        ax = axes[ci][1]
        ax.bar(x - w / 2, sync_p99, w, label="sync (blocks runtime)", color="#e74c3c", alpha=0.85)
        ax.bar(x + w / 2, sb_p99, w, label="spawn_blocking", color="#2ecc71", alpha=0.85)
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"P99 — concurrency={conc}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    axes[-1][0].set_xlabel("Input Sequence Length (tokens)")
    axes[-1][1].set_xlabel("Input Sequence Length (tokens)")

    fig.suptitle(
        "Tokenizer Encode: sync vs spawn_blocking\n"
        f"Arc<dyn Tokenizer> — {crows[0].get('num_samples', '?')} samples per config",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=150)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
