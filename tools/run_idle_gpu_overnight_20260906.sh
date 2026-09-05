#!/usr/bin/env bash
set -euo pipefail

: "${MANIFEST:?set MANIFEST to the frozen 50/16 manifest}"
: "${DIRECT1500_CHECKPOINT:?set DIRECT1500_CHECKPOINT to exact Direct@1500}"
: "${DIRECT3000_CHECKPOINT:?set DIRECT3000_CHECKPOINT to exact Direct@3000}"
: "${SIM_PRETRAIN_CHECKPOINT:?set SIM_PRETRAIN_CHECKPOINT to exact official sim_pretrain}"
: "${KIT_ROOT:?set KIT_ROOT to official Track 1 kit root}"
: "${OVERNIGHT_OUT_ROOT:?set OVERNIGHT_OUT_ROOT to a fresh artifact directory}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
A2_ROOT="$OVERNIGHT_OUT_ROOT/a2_multiscale"
HCUR_ROOT="$OVERNIGHT_OUT_ROOT/horizon_curriculum"

mkdir -p "$OVERNIGHT_OUT_ROOT"
git -C "$ROOT" rev-parse HEAD | tee "$OVERNIGHT_OUT_ROOT/execution_commit.txt"

printf '%s\n' "[OVERNIGHT] verification start"
"$PYTHON_BIN" -m pytest -q \
  "$ROOT/tests/test_arch_a2_multiscale.py" \
  "$ROOT/tests/test_horizon_curriculum.py"
"$PYTHON_BIN" -m py_compile \
  "$ROOT/tools/realpde_arch_a2_multiscale.py" \
  "$ROOT/tools/realpde_horizon_curriculum.py"

printf '%s\n' "[OVERNIGHT] task A: A2 multiscale"
MANIFEST="$MANIFEST" \
START_CHECKPOINT="$DIRECT1500_CHECKPOINT" \
REFERENCE_CHECKPOINT="$DIRECT3000_CHECKPOINT" \
KIT_ROOT="$KIT_ROOT" \
OUT_ROOT="$A2_ROOT" \
PYTHON_BIN="$PYTHON_BIN" \
bash "$ROOT/tools/run_arch_a2_multiscale_remote.sh"

printf '%s\n' "[OVERNIGHT] task B: horizon curriculum"
MANIFEST="$MANIFEST" \
SIM_PRETRAIN_CHECKPOINT="$SIM_PRETRAIN_CHECKPOINT" \
REFERENCE_CHECKPOINT="$DIRECT3000_CHECKPOINT" \
KIT_ROOT="$KIT_ROOT" \
OUT_ROOT="$HCUR_ROOT" \
PYTHON_BIN="$PYTHON_BIN" \
bash "$ROOT/tools/run_horizon_curriculum_remote.sh"

printf '%s\n' "[OVERNIGHT] all queued experiments complete"
