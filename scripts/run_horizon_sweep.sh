#!/usr/bin/env bash
# Run the official LingBot-VA I2VA entry point on independent GPUs.
#
# Example:
#   LINGBOT_VA_ROOT=/data/lingbot-va \
#   MODEL_ROOT=/data/models/lingbot-va-posttrain-robotwin \
#   bash scripts/run_horizon_sweep.sh
set -euo pipefail

repo_root="${LINGBOT_VA_ROOT:?set LINGBOT_VA_ROOT to the official LingBot-VA checkout}"
model_root="${MODEL_ROOT:?set MODEL_ROOT to the downloaded LingBot-VA RobotTwin checkpoint}"
output_root="${OUTPUT_ROOT:-$PWD/artifacts/generated}"
python_bin="${PYTHON_BIN:-python}"
horizons="${HORIZONS:-4 7 10}"
gpus="${GPUS:-0 1 2}"

read -r -a horizon_array <<< "$horizons"
read -r -a gpu_array <<< "$gpus"
if [[ ${#horizon_array[@]} -ne ${#gpu_array[@]} ]]; then
  echo "HORIZONS and GPUS must contain the same number of entries" >&2
  exit 2
fi

# The upstream config contains the model location. Keeping this explicit makes
# the one local, deployment-specific edit visible and easy to undo.
config="$repo_root/wan_va/configs/va_robotwin_cfg.py"
if ! grep -Fq "$model_root" "$config"; then
  echo "Set wan22_pretrained_model_name_or_path in $config to: $model_root" >&2
  exit 2
fi

mkdir -p "$output_root" "$output_root/logs"
pids=()
names=()
for index in "${!horizon_array[@]}"; do
  gpu="${gpu_array[$index]}"
  horizon="${horizon_array[$index]}"
  name="horizon_$horizon"
  result_dir="$output_root/$name"
  log="$output_root/logs/$name.log"
  mkdir -p "$result_dir"
  (
    started="$(date +%s)"
    cd "$repo_root"
    CUDA_VISIBLE_DEVICES="$gpu" LINGBOT_NUM_CHUNKS="$horizon" TOKENIZERS_PARALLELISM=false \
      "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=1 \
      -m wan_va.wan_va_server --config-name robotwin_i2av --save_root "$result_dir"
    printf 'WALL_SECONDS=%s\n' "$(( $(date +%s) - started ))"
  ) >"$log" 2>&1 &
  pids+=("$!")
  names+=("$name")
done

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "${names[$index]} failed; see $output_root/logs/${names[$index]}.log" >&2
    status=1
  fi
done
exit "$status"
