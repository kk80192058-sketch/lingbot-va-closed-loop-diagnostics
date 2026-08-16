#!/usr/bin/env python3
"""Build a checked, shareable manifest from completed I2VA worker outputs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def video_metadata(path: Path) -> dict[str, float | int]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames", "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    return {
        "generated_frames": int(payload["streams"][0]["nb_read_frames"]),
        "video_seconds": round(float(payload["format"]["duration"]), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--logs-root", type=Path,
        help="Worker-log directory; defaults to <results-root>/logs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-name", default="NVIDIA H20 96GB")
    args = parser.parse_args()

    logs_root = args.logs_root or (args.results_root / "logs")
    rows = []
    for result_dir in args.results_root.glob("horizon_*"):
        match = re.fullmatch(r"horizon_(\d+)", result_dir.name)
        video = result_dir / "demo.mp4"
        log = logs_root / f"{result_dir.name}.log"
        if not match or not video.is_file() or not log.is_file():
            continue
        timing = re.search(r"WALL_SECONDS=(\d+)", log.read_text(errors="replace"))
        if not timing:
            raise ValueError(f"No WALL_SECONDS record in {log}")
        rows.append({
            "condition": result_dir.name,
            "prediction_horizon_chunks": int(match.group(1)),
            **video_metadata(video),
            "wall_seconds": int(timing.group(1)),
            "gpu": args.gpu_name,
            "mode": "official LingBot-VA robotwin_i2av",
        })
    if not rows:
        raise ValueError("No complete horizon_* results found")
    rows.sort(key=lambda row: row["prediction_horizon_chunks"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
    print(f"Wrote {len(rows)} records to {args.output}")


if __name__ == "__main__":
    main()
