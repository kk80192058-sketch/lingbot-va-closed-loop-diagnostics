#!/usr/bin/env python3
"""Plot the checked-in LingBot-VA horizon-sweep manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/rollout_metrics.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/rollout_scaling.png"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: row["prediction_horizon_chunks"])
    x = [row["prediction_horizon_chunks"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(x, [row["wall_seconds"] for row in rows], "o-", color="#2563eb")
    axes[0].set(title="End-to-end rollout time", xlabel="Prediction horizon (chunks)", ylabel="Seconds")
    axes[1].plot(x, [row["video_seconds"] for row in rows], "o-", color="#16a34a")
    axes[1].set(title="Generated video length", xlabel="Prediction horizon (chunks)", ylabel="Seconds")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("LingBot-VA official I2VA rollout scaling (3×H20)", fontweight="bold")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
