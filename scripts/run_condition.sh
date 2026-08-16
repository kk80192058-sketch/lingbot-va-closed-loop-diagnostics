#!/usr/bin/env bash
set -euo pipefail

condition="${1:?usage: run_condition.sh <every_step|every_2_steps|every_4_steps|open_loop>}"
repo_root="${LINGBOT_VA_ROOT:?set LINGBOT_VA_ROOT to the official LingBot-VA checkout}"
results_root="${RESULTS_ROOT:-$PWD/artifacts/raw}"

mkdir -p "$results_root/$condition"

# Keep this wrapper intentionally small: it owns experiment metadata and delegates
# model launch to the official repository, so upstream updates remain easy to adopt.
export LINGBOT_DIAGNOSTIC_CONDITION="$condition"
export LINGBOT_DIAGNOSTIC_OUTPUT="$results_root/$condition"

cd "$repo_root"
bash evaluation/robotwin/launch_server.sh
