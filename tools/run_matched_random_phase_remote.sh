#!/usr/bin/env bash
set -euo pipefail

: "${RW_SOURCE_ROOT:?set RW_SOURCE_ROOT to the synced tools directory}"
: "${RW_DATA_ROOT:?set RW_DATA_ROOT to the remote RealPDE_data directory}"
: "${RW_KIT_ROOT:?set RW_KIT_ROOT to the official Track 1 v9 kit directory}"
: "${RW_OUTPUT_ROOT:?set RW_OUTPUT_ROOT to a fresh remote output directory}"
: "${RW_MANIFEST:?set RW_MANIFEST to the frozen 50/16 manifest}"
: "${RW_CHECKPOINT:?set RW_CHECKPOINT to the shared initial checkpoint}"
: "${RW_EXECUTION_COMMIT:?set RW_EXECUTION_COMMIT to the frozen execution commit}"

IMAGE="${RW_IMAGE:-realpde-pytorch-h5py:0831}"
MAX_TRAIN_SECONDS="${RW_MAX_TRAIN_SECONDS:-21600}"
COMMON=(
  --manifest "$RW_MANIFEST" --kit-root /kit --checkpoint "$RW_CHECKPOINT"
  --batch-size 8 --workers 2 --lr 1e-5 --seed 20260901 --updates 3000
  --max-train-seconds "$MAX_TRAIN_SECONDS" --eval-updates 1000,2000,3000
  --train-window-mode random_phase --execution-commit "$RW_EXECUTION_COMMIT"
)

if [[ -e "$RW_OUTPUT_ROOT" ]] && [[ -n "$(find "$RW_OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "RW_OUTPUT_ROOT must be fresh/empty: $RW_OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$RW_OUTPUT_ROOT"
printf '%s\n' RUNNING > "$RW_OUTPUT_ROOT/status"
on_error() { local code=$?; printf 'FAILED %s\n' "$code" > "$RW_OUTPUT_ROOT/status"; exit "$code"; }
trap on_error ERR

run_arm() {
  local out_dir="$1"; shift
  docker run --rm --gpus all --shm-size=8g \
    -v "$RW_SOURCE_ROOT:/task:ro" -v "$RW_DATA_ROOT:$RW_DATA_ROOT:ro" -v "$RW_DATA_ROOT:/data:ro" \
    -v "$RW_KIT_ROOT:/kit:ro" -v "$RW_OUTPUT_ROOT:/out" -w /task "$IMAGE" \
    python realpde_b1_p0a_n2.py "${COMMON[@]}" --out-dir "/out/$out_dir" "$@" 2>&1 | tee "$RW_OUTPUT_ROOT/$out_dir.train.log"
}

run_arm RW-MA_phase0 --random-phase-forced-phase 0
run_arm RW-MB_random_phase --early-gate-summary /out/RW-MA_phase0/summary.json --early-gate-update 1000 --early-gate-mode matched_phase_severe

docker run --rm -v "$RW_SOURCE_ROOT:/task:ro" -v "$RW_OUTPUT_ROOT:/out" -w /task "$IMAGE" \
  python realpde_matched_phase_report.py --rw-ma-dir /out/RW-MA_phase0 --rw-mb-dir /out/RW-MB_random_phase --out-dir /out/report

for run_dir in RW-MA_phase0 RW-MB_random_phase; do
  docker run --rm -v "$RW_SOURCE_ROOT:/task:ro" -v "$RW_OUTPUT_ROOT:/out" -w /task "$IMAGE" \
    python realpde_runtime_context.py manifest --run-dir "/out/$run_dir" >/dev/null
  docker run --rm -v "$RW_SOURCE_ROOT:/task:ro" -v "$RW_OUTPUT_ROOT:/out" -w /task "$IMAGE" \
    python build_training_review_log.py --input "/out/$run_dir.train.log" --output "/out/$run_dir.training.review.log" >/dev/null
done

printf '%s\n' COMPLETE > "$RW_OUTPUT_ROOT/status"
trap - ERR
