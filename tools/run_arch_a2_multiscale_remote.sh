#!/usr/bin/env bash
set -euo pipefail

: "${MANIFEST:?set MANIFEST to the frozen 50/16 manifest}"
: "${START_CHECKPOINT:?set START_CHECKPOINT to the exact Direct@1500 checkpoint}"
: "${REFERENCE_CHECKPOINT:?set REFERENCE_CHECKPOINT to the exact Direct@3000 checkpoint}"
: "${KIT_ROOT:?set KIT_ROOT to the official Track 1 kit root}"
: "${OUT_ROOT:?set OUT_ROOT to a fresh remote artifact directory}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
EXPERIMENT_ID="${EXPERIMENT_ID:-T1-ID-ARCH-A2-MULTISCALE-S20260906}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFLIGHT_OUT="$OUT_ROOT/preflight"
RUN_OUT="$OUT_ROOT/run"

mkdir -p "$OUT_ROOT"
git -C "$ROOT" rev-parse HEAD | tee "$OUT_ROOT/execution_commit.txt"

printf '%s\n' "[A2] preflight start"
"$PYTHON_BIN" "$ROOT/tools/realpde_arch_a2_multiscale.py" \
  --experiment-id "$EXPERIMENT_ID" \
  --manifest "$MANIFEST" \
  --checkpoint "$START_CHECKPOINT" \
  --reference-checkpoint "$REFERENCE_CHECKPOINT" \
  --kit-root "$KIT_ROOT" \
  --out-dir "$RUN_OUT" \
  --device "$DEVICE" \
  --preflight \
  --preflight-out "$PREFLIGHT_OUT"

printf '%s\n' "[A2] preflight passed; formal 50/16 training start"
"$PYTHON_BIN" "$ROOT/tools/realpde_arch_a2_multiscale.py" \
  --experiment-id "$EXPERIMENT_ID" \
  --manifest "$MANIFEST" \
  --checkpoint "$START_CHECKPOINT" \
  --reference-checkpoint "$REFERENCE_CHECKPOINT" \
  --kit-root "$KIT_ROOT" \
  --out-dir "$RUN_OUT" \
  --device "$DEVICE" \
  --preflight-evidence "$PREFLIGHT_OUT/preflight.json"

printf '%s\n' "[A2] complete; see $RUN_OUT/gate_result.json"
