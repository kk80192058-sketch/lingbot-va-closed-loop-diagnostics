#!/usr/bin/env python3
"""Aggregate rollout records into a shareable closed-loop diagnostic figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = ["success", "latency_ms", "lpips", "ssim", "state_error_cm"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("No episode records found.")
    frame = pd.DataFrame(rows)
    missing = {"condition", "success"} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    available = [metric for metric in METRICS if metric in frame.columns]
    summary = frame.groupby("condition", sort=False)[available].mean(numeric_only=True)
    summary["episodes"] = frame.groupby("condition", sort=False).size()
    summary.to_csv(args.output_dir / "condition_summary.csv")

    plot_metrics = [m for m in ["success", "latency_ms", "lpips", "state_error_cm"] if m in summary]
    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(4.5 * len(plot_metrics), 4))
    if len(plot_metrics) == 1:
        axes = [axes]
    for axis, metric in zip(axes, plot_metrics):
        values = summary[metric]
        axis.bar(values.index, values.values, color="#2563eb")
        axis.set_title(metric.replace("_", " "))
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("LingBot-VA closed-loop rollout diagnostics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.output_dir / "closed_loop_diagnostics.png", dpi=180, bbox_inches="tight")
    print(summary.to_string())


if __name__ == "__main__":
    main()
