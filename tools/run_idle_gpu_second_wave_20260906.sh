#!/usr/bin/env bash
set -euo pipefail

: "${WAIT_PID:?set WAIT_PID to the currently running first overnight queue PID}"
: "${MANIFEST:?set MANIFEST to the frozen 50/16 manifest}"
: "${DIRECT3000_CHECKPOINT:?set DIRECT3000_CHECKPOINT to exact matched Direct@3000}"
: "${MF3000_CHECKPOINT:?set MF3000_CHECKPOINT to Campaign02 C0 / MF@3000}"
: "${KIT_ROOT:?set KIT_ROOT to official Track 1 kit root}"
: "${SECOND_WAVE_OUT_ROOT:?set SECOND_WAVE_OUT_ROOT to a fresh artifact directory}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFLIGHT_OUT="$SECOND_WAVE_OUT_ROOT/preflight"
FORMAL_OUT="$SECOND_WAVE_OUT_ROOT/formal"

if ! [[ "$WAIT_PID" =~ ^[0-9]+$ ]]; then
  echo "WAIT_PID must be numeric: $WAIT_PID" >&2
  exit 2
fi

for path in "$MANIFEST" "$DIRECT3000_CHECKPOINT" "$MF3000_CHECKPOINT" "$KIT_ROOT/scoring.py"; do
  if [[ ! -f "$path" ]]; then
    echo "required asset missing: $path" >&2
    exit 2
  fi
done

if [[ -e "$SECOND_WAVE_OUT_ROOT" ]] && [[ -n "$(ls -A "$SECOND_WAVE_OUT_ROOT" 2>/dev/null || true)" ]]; then
  echo "SECOND_WAVE_OUT_ROOT must be fresh/empty: $SECOND_WAVE_OUT_ROOT" >&2
  exit 2
fi
mkdir -p "$SECOND_WAVE_OUT_ROOT"

git -C "$ROOT" rev-parse HEAD | tee "$SECOND_WAVE_OUT_ROOT/execution_commit.txt"
printf '%s\n' "[SECOND-WAVE] waiting for first overnight queue PID=$WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep "$WAIT_POLL_SECONDS"
done
sleep 20
printf '%s\n' "[SECOND-WAVE] first queue exited; verification start"

"$PYTHON_BIN" -m pytest -q "$ROOT/tests/test_mf_long_convergence.py"
"$PYTHON_BIN" -m py_compile "$ROOT/tools/realpde_mf_long_convergence.py"

printf '%s\n' "[SECOND-WAVE] preflight"
"$PYTHON_BIN" "$ROOT/tools/realpde_mf_long_convergence.py" \
  --preflight \
  --manifest "$MANIFEST" \
  --direct-checkpoint "$DIRECT3000_CHECKPOINT" \
  --mf-checkpoint "$MF3000_CHECKPOINT" \
  --kit-root "$KIT_ROOT" \
  --out-dir "$FORMAL_OUT" \
  --preflight-out "$PREFLIGHT_OUT"

printf '%s\n' "[SECOND-WAVE] formal Direct@3000->15000 then MF@3000->15000"
"$PYTHON_BIN" "$ROOT/tools/realpde_mf_long_convergence.py" \
  --manifest "$MANIFEST" \
  --direct-checkpoint "$DIRECT3000_CHECKPOINT" \
  --mf-checkpoint "$MF3000_CHECKPOINT" \
  --kit-root "$KIT_ROOT" \
  --out-dir "$FORMAL_OUT" \
  --preflight-evidence "$PREFLIGHT_OUT/preflight.json"

printf '%s\n' "[SECOND-WAVE] complete"
touch "$SECOND_WAVE_OUT_ROOT/COMPLETE"
