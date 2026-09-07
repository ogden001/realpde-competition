#!/usr/bin/env bash
set -euo pipefail

: "${RW_SOURCE_ROOT:?set RW_SOURCE_ROOT to the synced source directory}"
: "${RW_DATA_ROOT:?set RW_DATA_ROOT to the remote RealPDE_data directory}"
: "${RW_KIT_ROOT:?set RW_KIT_ROOT to the official Track 1 v9 kit directory}"
: "${RW_OUTPUT_ROOT:?set RW_OUTPUT_ROOT to a fresh remote output directory}"
: "${RW_MANIFEST:?set RW_MANIFEST to the frozen 50/16 manifest}"
: "${RW_CHECKPOINT:?set RW_CHECKPOINT to the shared initial checkpoint}"

IMAGE="${RW_IMAGE:-realpde-pytorch-h5py:0831}"
MAX_TRAIN_SECONDS="${RW_MAX_TRAIN_SECONDS:-21600}"
SEED="${RW_SEED:-20260901}"
EVAL_UPDATES="${RW_EVAL_UPDATES:-1000,2000,3000,5000,7500}"
COMMON=(
  --manifest "$RW_MANIFEST"
  --kit-root /kit
  --checkpoint "$RW_CHECKPOINT"
  --batch-size 8
  --workers 2
  --lr 1e-5
  --seed "$SEED"
  --updates 7500
  --max-train-seconds "$MAX_TRAIN_SECONDS"
  --eval-updates "$EVAL_UPDATES"
)

if [[ -e "$RW_OUTPUT_ROOT" ]] && [[ -n "$(find "$RW_OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "RW_OUTPUT_ROOT must be fresh/empty: $RW_OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$RW_OUTPUT_ROOT"
STATUS="$RW_OUTPUT_ROOT/status"
printf '%s\n' RUNNING > "$STATUS"

on_error() {
  local code=$?
  printf 'FAILED %s\n' "$code" > "$STATUS"
  exit "$code"
}
trap on_error ERR

run_arm() {
  local mode="$1"
  local out_dir="$2"
  shift 2
  docker run --rm --gpus all --shm-size=8g \
    -v "$RW_SOURCE_ROOT:/task:ro" \
    -v "$RW_DATA_ROOT:$RW_DATA_ROOT:ro" \
    -v "$RW_DATA_ROOT:/data:ro" \
    -v "$RW_KIT_ROOT:/kit:ro" \
    -v "$RW_OUTPUT_ROOT:/out" \
    -w /task "$IMAGE" \
    python realpde_b1_p0a_n2.py "${COMMON[@]}" \
    --out-dir "/out/$out_dir" --train-window-mode "$mode" "$@"
}

run_arm fixed RW-00_fixed
run_arm random_phase RW-01_random_phase --early-gate-summary /out/RW-00_fixed/summary.json

docker run --rm \
  -v "$RW_SOURCE_ROOT:/task:ro" \
  -v "$RW_OUTPUT_ROOT:/out" \
  -w /task "$IMAGE" \
  python realpde_random_phase_report.py \
  --rw00-dir /out/RW-00_fixed \
  --rw01-dir /out/RW-01_random_phase \
  --out-dir /out/report

printf '%s\n' COMPLETE > "$STATUS"
trap - ERR
