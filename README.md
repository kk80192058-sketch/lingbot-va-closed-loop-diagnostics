# Closed-Loop Diagnostics for LingBot-VA

An evaluation harness for measuring how observation-feedback frequency affects a video-action world model's prediction drift, latency, and long-horizon robot-manipulation success.

## Why this project

LingBot-VA jointly predicts future video latents and robot actions. A generated rollout is useful only when it remains aligned with the world after the robot acts. This project evaluates that practical question by varying the frequency at which real observations are fed back into a closed-loop rollout.

## Experiment

```text
real observation ──> LingBot-VA server ──> imagined video + action chunk
       ^                                               │
       └──── feed back every N control steps ──────────┘
```

We run the same task with feedback intervals `N ∈ {1, 2, 4, open_loop}` and record:

- task success rate;
- rollout video error (LPIPS / SSIM when available);
- simulator state error (end-effector and object position when exposed);
- action-chunk latency.

## Hardware and software

- 3 × NVIDIA H20 96GB GPUs with NVLink
- Ubuntu 22.04, CUDA 12.8, PyTorch 2.9.1
- LingBot-VA official checkpoint and RobotTwin-2.0

The three GPUs are used for **independent rollout conditions**, not tensor-parallel single episodes: this maximizes completed experiments per rented GPU-hour.

## Reproduce

Install LingBot-VA and RobotTwin using their official instructions, then run one condition per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_condition.sh every_step
CUDA_VISIBLE_DEVICES=1 bash scripts/run_condition.sh every_2_steps
CUDA_VISIBLE_DEVICES=2 bash scripts/run_condition.sh every_4_steps
```

Each condition must append episode records to `artifacts/metrics.jsonl`. Summarize the results with:

```bash
python scripts/summarize.py --input artifacts/metrics.jsonl --output-dir artifacts/summary
```

## Expected artifact layout

```text
artifacts/
├── metrics.jsonl
├── raw/
│   └── <condition>/episode_<id>/
│       ├── observed.mp4
│       ├── imagined.mp4
│       └── metadata.json
└── summary/
    ├── condition_summary.csv
    └── closed_loop_diagnostics.png
```

## Status

Infrastructure setup and official-checkpoint inference are in progress. Results, videos, and the final table will be added after the first evaluation run.

## References

- [LingBot-VA official repository](https://github.com/Robbyant/lingbot-va)
- [Causal World Modeling for Robot Control](https://arxiv.org/abs/2601.21998)
