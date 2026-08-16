# LingBot-VA 世界动作模型：并行 Rollout 诊断实验

这是一个基于官方 **LingBot-VA** 的可复现实验项目：给定三路机器人相机观测和自然语言指令，使用官方 **Image-to-Video-Action（I2VA）** 流程，同时预测未来视频与机器人动作；随后比较不同预测长度下的生成视频、端到端时延与吞吐。

项目使用 3 张 **NVIDIA H20 96GB** 并行运行独立 Rollout，所有视频、指标和复现脚本均已提交。

## 为什么做这个项目

LingBot-VA 同时建模“未来世界会如何变化”和“机器人应该执行什么动作”。对具身智能而言，单纯生成视频并不够：模型需要在有限延迟内给出与视觉预测一致的动作序列。

本项目完成了世界动作模型的第一步验证：

```text
三路真实相机观测 + 语言指令
              │
              ▼
       LingBot-VA（I2VA）
              │
              ├── 预测未来视觉序列（World）
              └── 预测机器人动作块（Action）
```

实验固定输入观测和指令，仅改变预测长度 `H ∈ {4, 7, 10}`，用来观察长时序预测带来的时延与视频延展。

## 已完成的三卡实验

三个 Worker 均使用官方 `robotwin_i2av` 配置、官方 LingBot-VA RobotTwin Checkpoint、相同三路相机输入和相同指令。配置每个 Chunk 前进 2 个 Latent Frame；经过 VAE 时序解码后，得到如下视频帧数。

| GPU | 预测长度（Chunk） | 生成帧数 | 生成视频时长 | 端到端耗时 |
| --- | ---: | ---: | ---: | ---: |
| NVIDIA H20 #0 | 4 | 29 | 2.9 秒 | 58 秒 |
| NVIDIA H20 #1 | 7 | 53 | 5.3 秒 | 82 秒 |
| NVIDIA H20 #2 | 10 | 77 | 7.7 秒 | 103 秒 |

端到端耗时包含 Checkpoint 冷启动加载和最终视频解码，因此反映的是部署体验，而不是只统计 Denoising 的理想化速度。

![预测长度与端到端时延](artifacts/rollout_scaling.png)

| H = 4 | H = 7 | H = 10 |
| --- | --- | --- |
| ![4 Chunk Rollout](artifacts/preview/horizon_4.gif) | ![7 Chunk Rollout](artifacts/preview/horizon_7.gif) | ![10 Chunk Rollout](artifacts/preview/horizon_10.gif) |

原始 MP4、GIF 预览和结构化指标都在 [`artifacts/`](artifacts/) 目录中。

## 实验结论

- 3 个独立 Rollout 同时运行，比把单个短序列强行做张量并行更适合这种单机三卡配置。
- 预测长度从 4 增加到 10 Chunk 时，视频从 2.9 秒延展至 7.7 秒；端到端耗时从 58 秒增长至 103 秒。
- 本次结果验证了 LingBot-VA 的“视觉未来 + 动作”联合 Rollout 可以在真实 H20 环境稳定运行，并给出了长时序预测的部署成本。

## 环境

- 3 × **NVIDIA H20 96GB**（NVLink）
- Ubuntu 22.04
- **CUDA 12.8**
- **PyTorch 2.9.1**
- 官方 **LingBot-VA RobotTwin Checkpoint**

## 复现三卡 Horizon Sweep

1. 按照 [LingBot-VA 官方仓库](https://github.com/Robbyant/lingbot-va) 安装环境并下载官方 RobotTwin Checkpoint。
2. 将官方配置文件 `wan_va/configs/va_robotwin_cfg.py` 中的 `wan22_pretrained_model_name_or_path` 指向本地 Checkpoint。
3. 运行：

```bash
LINGBOT_VA_ROOT=/data/lingbot-va \
MODEL_ROOT=/data/models/lingbot-va-posttrain-robotwin \
PYTHON_BIN=/path/to/python \
bash scripts/run_horizon_sweep.sh
```

默认会将视频和日志输出到 `artifacts/generated/`。完成后，从原始 MP4 和 Worker 日志自动生成可校验指标：

```bash
python scripts/make_rollout_manifest.py \
  --results-root artifacts/generated \
  --output artifacts/rollout_metrics.jsonl
```

若日志与视频目录分离，额外传入 `--logs-root /path/to/logs`。

## 后续可扩展方向：RobotTwin 闭环评估

当前已完成的是官方 I2VA 世界动作 Rollout。下一步可接入 **RobotTwin** Episode Client：每执行 `N` 个控制步回传一次真实观测，比较 `N ∈ {1, 2, 4, open_loop}` 的任务成功率、LPIPS/SSIM、状态误差与 Action Latency。

仓库已提供 `scripts/summarize.py`，可汇总这一闭环实验产生的 `artifacts/metrics.jsonl`。本仓库没有伪造未运行的成功率或误差指标。

## 目录说明

```text
artifacts/
├── raw/                         # 三段原始 MP4
├── preview/                     # GitHub 可直接预览的 GIF
├── rollout_metrics.jsonl        # 经原始视频与日志核验的指标
└── rollout_scaling.png          # 预测长度—时延/视频长度图
scripts/
├── run_horizon_sweep.sh         # 三张 GPU 并发执行官方 I2VA
├── make_rollout_manifest.py     # 从原始结果重建指标
├── plot_rollout_scaling.py      # 绘制趋势图
└── summarize.py                 # RobotTwin 闭环实验的结果汇总器
```

## 参考资料

- [LingBot-VA 官方仓库](https://github.com/Robbyant/lingbot-va)
- [Causal World Modeling for Robot Control](https://arxiv.org/abs/2601.21998)
