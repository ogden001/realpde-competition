#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
: "${DATA_ROOT:?Set DATA_ROOT to the released real-PIV H5 directory}"
: "${MANIFEST:?Set MANIFEST to the frozen 50/16/16 ID manifest}"
: "${KIT_ROOT:?Set KIT_ROOT to the Track 1 starting-kit v9 root}"
: "${RESUME_CHECKPOINT:?Set RESUME_CHECKPOINT to the P0-A+N2 validation update-18860 resume checkpoint}"
: "${OUT_ROOT:?Set OUT_ROOT to a new artifact directory}"

if [[ -e "$OUT_ROOT" ]]; then
  echo "OUT_ROOT already exists: $OUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUT_ROOT"

status_file="$OUT_ROOT/launcher_status.json"
on_error() {
  local code=$?
  printf '{"state":"FAILED","exit_code":%d}\n' "$code" > "$status_file"
  exit "$code"
}
trap on_error ERR
printf '{"state":"RUNNING"}\n' > "$status_file"

COMMON=(
  --data-root "$DATA_ROOT"
  --manifest "$MANIFEST"
  --kit-root "$KIT_ROOT"
  --resume-checkpoint "$RESUME_CHECKPOINT"
  --expected-start-update 18860
  --max-gpu-gib 12.0
  --micro-batch 4
  --accumulate 2
  --workers 2
  --save-interval 100
  --seed 20260901
)

"$PYTHON_BIN" "$REPO_ROOT/tools/sota_lr_screen.py" preflight \
  --manifest "$MANIFEST" \
  --kit-root "$KIT_ROOT" \
  --resume-checkpoint "$RESUME_CHECKPOINT" \
  --out "$OUT_ROOT/preflight.json"

"$PYTHON_BIN" "$REPO_ROOT/tools/realpde_p0a_n2_full.py" \
  "${COMMON[@]}" \
  --out-dir "$OUT_ROOT/smoke_decay_lr5e6" \
  --updates 18862 \
  --lr 5e-6 \
  --max-train-seconds 900 \
  --max-windows-per-trajectory 1 \
  > "$OUT_ROOT/smoke_decay_lr5e6.log" 2>&1
"$PYTHON_BIN" "$REPO_ROOT/tools/sota_lr_screen.py" check-run \
  --run-dir "$OUT_ROOT/smoke_decay_lr5e6" \
  --expected-update 18862 \
  --expected-lr 5e-6

"$PYTHON_BIN" "$REPO_ROOT/tools/realpde_p0a_n2_full.py" \
  "${COMMON[@]}" \
  --out-dir "$OUT_ROOT/control_lr1e5" \
  --updates 22960 \
  --lr 1e-5 \
  --max-train-seconds 5400 \
  > "$OUT_ROOT/control_lr1e5.log" 2>&1
"$PYTHON_BIN" "$REPO_ROOT/tools/sota_lr_screen.py" check-run \
  --run-dir "$OUT_ROOT/control_lr1e5" \
  --expected-update 22960 \
  --expected-lr 1e-5

"$PYTHON_BIN" "$REPO_ROOT/tools/realpde_p0a_n2_full.py" \
  "${COMMON[@]}" \
  --out-dir "$OUT_ROOT/decay_lr5e6" \
  --updates 22960 \
  --lr 5e-6 \
  --max-train-seconds 5400 \
  > "$OUT_ROOT/decay_lr5e6.log" 2>&1
"$PYTHON_BIN" "$REPO_ROOT/tools/sota_lr_screen.py" check-run \
  --run-dir "$OUT_ROOT/decay_lr5e6" \
  --expected-update 22960 \
  --expected-lr 5e-6

"$PYTHON_BIN" "$REPO_ROOT/tools/sota_lr_screen.py" summarize \
  --control-dir "$OUT_ROOT/control_lr1e5" \
  --decay-dir "$OUT_ROOT/decay_lr5e6" \
  --out "$OUT_ROOT/lr_ab_summary.json"

printf '{"state":"REVIEW_REQUIRED","summary":"lr_ab_summary.json"}\n' > "$status_file"
