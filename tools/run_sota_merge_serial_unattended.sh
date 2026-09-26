#!/usr/bin/env bash
set -euo pipefail

# Detached launcher for REALPDE serial SOTA-merge comparison.
#
# Usage:
#   bash tools/run_sota_merge_serial_unattended.sh \
#     DATA_ROOT TRAIN_DEV_MANIFEST AOA10_MANIFEST KIT_ROOT \
#     SIM_REAL_INIT STAGE_A_57K_CHECKPOINT OUT_ROOT [WORKERS]
#
# Scientific settings are frozen inside run_sota_merge_serial_unattended.py.
# This launcher only detaches the OS process so SSH/Codex session loss does not
# terminate the experiment.

if [[ $# -lt 7 || $# -gt 8 ]]; then
  echo "usage: $0 DATA_ROOT TRAIN_DEV_MANIFEST AOA10_MANIFEST KIT_ROOT SIM_REAL_INIT STAGE_A_57K_CHECKPOINT OUT_ROOT [WORKERS]" >&2
  exit 2
fi

DATA_ROOT="$1"
MANIFEST="$2"
AOA_MANIFEST="$3"
KIT_ROOT="$4"
INIT_CKPT="$5"
STAGE_A_CKPT="$6"
OUT_ROOT="$7"
WORKERS="${8:-4}"

mkdir -p "$OUT_ROOT"
LOCK_FILE="$OUT_ROOT/.serial_unattended.lock"
PID_FILE="$OUT_ROOT/supervisor.pid"
LOG_FILE="$OUT_ROOT/supervisor.stdout.log"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another serial supervisor already holds $LOCK_FILE" >&2
  exit 3
fi

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "supervisor already running: pid=$OLD_PID" >&2
    exit 4
  fi
fi

nohup python -u -B tools/run_sota_merge_serial_unattended.py \
  --data-root "$DATA_ROOT" \
  --manifest "$MANIFEST" \
  --aoa-manifest "$AOA_MANIFEST" \
  --kit-root "$KIT_ROOT" \
  --init-checkpoint "$INIT_CKPT" \
  --stage-a-checkpoint "$STAGE_A_CKPT" \
  --out-root "$OUT_ROOT" \
  --workers "$WORKERS" \
  --poll-seconds 120 \
  --max-attempts 5 \
  --retry-cooldown-seconds 30 \
  >>"$LOG_FILE" 2>&1 < /dev/null &

PID=$!
echo "$PID" > "$PID_FILE"
echo "started serial SOTA-merge supervisor pid=$PID"
echo "log: $LOG_FILE"
echo "heartbeat: $OUT_ROOT/heartbeat.json"

# The child inherits fd 9 and therefore keeps the flock after this shell exits.
disown "$PID" 2>/dev/null || true
