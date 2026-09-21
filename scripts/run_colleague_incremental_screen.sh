#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 CODE_ROOT REAL_ROOT BASE_CHECKPOINT START_CHECKPOINT MODEL_ROOT DATA_MANIFEST SPLIT_MANIFEST OUT_ROOT" >&2
  exit 2
fi

code_root=$1
real_root=$2
base_checkpoint=$3
start_checkpoint=$4
model_root=$5
data_manifest=$6
split_manifest=$7
out_root=$8

exec python -u -B "$code_root/tools/colleague_80pt/run_incremental_screen.py" \
  --real-root "$real_root" \
  --base-checkpoint "$base_checkpoint" \
  --start-checkpoint "$start_checkpoint" \
  --model-root "$model_root" \
  --data-manifest "$data_manifest" \
  --split-manifest "$split_manifest" \
  --out-root "$out_root"
