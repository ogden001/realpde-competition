#!/usr/bin/env bash
set -euo pipefail

: "${MANIFEST:?set MANIFEST to the frozen 50/16 manifest}"
: "${KIT_ROOT:?set KIT_ROOT to the official Track 1 kit root}"
: "${CHECKPOINT_26240:?set CHECKPOINT_26240}"
: "${CHECKPOINT_27880:?set CHECKPOINT_27880}"
: "${CHECKPOINT_30340:?set CHECKPOINT_30340}"
: "${DIRECT3000_CHECKPOINT:?set DIRECT3000_CHECKPOINT}"
: "${MF3000_CHECKPOINT:?set MF3000_CHECKPOINT}"
: "${LOWMEM_OUT_ROOT:?set LOWMEM_OUT_ROOT to a fresh output directory}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOUP_OUT="$LOWMEM_OUT_ROOT/checkpoint_soup"
MF_PREFLIGHT="$LOWMEM_OUT_ROOT/mf_long_preflight"
MF_OUT="$LOWMEM_OUT_ROOT/mf_long_convergence"

mkdir -p "$LOWMEM_OUT_ROOT"
git -C "$ROOT" rev-parse HEAD | tee "$LOWMEM_OUT_ROOT/execution_commit.txt"

printf '%s\n' '[LOWMEM] static verification'
"$PYTHON_BIN" -m pytest -q \
  "$ROOT/tests/test_checkpoint_soup.py" \
  "$ROOT/tests/test_mf_long_convergence.py"
"$PYTHON_BIN" -m py_compile \
  "$ROOT/tools/realpde_checkpoint_soup.py" \
  "$ROOT/tools/realpde_mf_long_convergence.py" \
  "$ROOT/tools/realpde_mf_long_convergence_lowmem.py"

printf '%s\n' '[LOWMEM] task 1: checkpoint soup / prediction ensemble'
"$PYTHON_BIN" "$ROOT/tools/realpde_checkpoint_soup.py" \
  --manifest "$MANIFEST" \
  --kit-root "$KIT_ROOT" \
  --checkpoint-26240 "$CHECKPOINT_26240" \
  --checkpoint-27880 "$CHECKPOINT_27880" \
  --checkpoint-30340 "$CHECKPOINT_30340" \
  --out-dir "$SOUP_OUT" \
  --batch-size 2 \
  --workers 2 \
  --max-gpu-memory-gib 9.0

printf '%s\n' '[LOWMEM] task 2 preflight: MF long convergence, effective batch 8 / GPU microbatch 2'
"$PYTHON_BIN" "$ROOT/tools/realpde_mf_long_convergence_lowmem.py" \
  --manifest "$MANIFEST" \
  --direct-checkpoint "$DIRECT3000_CHECKPOINT" \
  --mf-checkpoint "$MF3000_CHECKPOINT" \
  --kit-root "$KIT_ROOT" \
  --out-dir "$MF_OUT" \
  --preflight \
  --preflight-out "$MF_PREFLIGHT" \
  --batch-size 8 \
  --micro-batch-size 2 \
  --workers 2 \
  --max-gpu-memory-gib 9.0

printf '%s\n' '[LOWMEM] task 2 formal: Direct@3000 and MF@3000 -> absolute 15000'
"$PYTHON_BIN" "$ROOT/tools/realpde_mf_long_convergence_lowmem.py" \
  --manifest "$MANIFEST" \
  --direct-checkpoint "$DIRECT3000_CHECKPOINT" \
  --mf-checkpoint "$MF3000_CHECKPOINT" \
  --kit-root "$KIT_ROOT" \
  --out-dir "$MF_OUT" \
  --preflight-evidence "$MF_PREFLIGHT/preflight.json" \
  --batch-size 8 \
  --micro-batch-size 2 \
  --workers 2 \
  --max-gpu-memory-gib 9.0

printf '%s\n' '[LOWMEM] queue complete'
