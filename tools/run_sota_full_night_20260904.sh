#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?set DATA_ROOT to the 82 released real-PIV H5 directory}"
: "${KIT_ROOT:?set KIT_ROOT to the official Track 1 starting-kit root}"
: "${RESUME_CHECKPOINT:?set RESUME_CHECKPOINT to the exact full-data update-15300 model_last.pth}"
: "${OUT_ROOT:?set OUT_ROOT to a new artifact root outside Git}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SMOKE_DIR="$OUT_ROOT/smoke_full_15300"
RUN_DIR="$OUT_ROOT/full_15300_to_43260"
PREFLIGHT_JSON="$OUT_ROOT/preflight.json"
SUMMARY_JSON="$OUT_ROOT/full_night_summary.json"
STATUS_JSON="$OUT_ROOT/launcher_status.json"

mkdir -p "$OUT_ROOT"
if [[ -e "$SMOKE_DIR" || -e "$RUN_DIR" ]]; then
  echo "refusing to reuse existing smoke/run directory under OUT_ROOT" >&2
  exit 2
fi

write_status() {
  local state="$1"
  local detail="$2"
  python - "$STATUS_JSON" "$state" "$detail" <<'PY'
import json, sys, time
from pathlib import Path
path, state, detail = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
path.write_text(json.dumps({"state": state, "detail": detail, "updated_at": time.time()}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

on_error() {
  local rc=$?
  write_status "FAILED" "launcher exited with code ${rc}"
  exit "$rc"
}
trap on_error ERR

python tools/sota_full_night.py preflight \
  --data-root "$DATA_ROOT" \
  --kit-root "$KIT_ROOT" \
  --resume-checkpoint "$RESUME_CHECKPOINT" \
  --output "$PREFLIGHT_JSON"

write_status "SMOKE" "running 2-update full-continuation smoke"
python tools/realpde_p0a_n2_full.py \
  --all-released-data \
  --data-root "$DATA_ROOT" \
  --kit-root "$KIT_ROOT" \
  --resume-checkpoint "$RESUME_CHECKPOINT" \
  --expected-start-update 15300 \
  --out-dir "$SMOKE_DIR" \
  --updates 15302 \
  --milestone-updates 15302 \
  --lr 1e-5 \
  --max-train-seconds 300 \
  --max-gpu-gib 12 \
  --micro-batch 4 \
  --accumulate 2 \
  --workers 2 \
  --save-interval 1 \
  --seed 20260901 \
  --max-trajectories 2

write_status "RUNNING" "full-data P0-A+N2 continuation 15300 -> 43260, hard cap 21300s"
python tools/realpde_p0a_n2_full.py \
  --all-released-data \
  --data-root "$DATA_ROOT" \
  --kit-root "$KIT_ROOT" \
  --resume-checkpoint "$RESUME_CHECKPOINT" \
  --expected-start-update 15300 \
  --expected-trajectories 82 \
  --expected-windows 3383 \
  --out-dir "$RUN_DIR" \
  --updates 43260 \
  --milestone-updates 31100 36500 37850 40560 43260 \
  --lr 1e-5 \
  --max-train-seconds 21300 \
  --max-gpu-gib 12 \
  --micro-batch 4 \
  --accumulate 2 \
  --workers 2 \
  --save-interval 100 \
  --seed 20260901 \
  --allow-time-cap

python tools/sota_full_night.py summarize \
  --run-dir "$RUN_DIR" \
  --output "$SUMMARY_JSON"

write_status "REVIEW_REQUIRED" "see $SUMMARY_JSON"
trap - ERR
