#!/usr/bin/env bash
# Dev evaluation for the 80-point solution.
#
# Two layers:
#   (1) full metric evaluation (rel_l2 / tke / mvpe / time / sps) is embedded in
#       Stage 2 training: residual_multi.py prints EVAL_TOP every eval_interval
#       updates on the 640-window dev split.
#   (2) SPS-only dev evaluation of the uncertainty head uses scan_bounds.py
#       (300-point grid, exact official SPS formula).
#
# Reference: docs/colleague_80pt_handoff/SPS.md
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="${REPO_ROOT}/tools/colleague_80pt"

RUNS="${RUNS:-/runs}"
RESIDUAL_DIR="${RESIDUAL_DIR:-${RUNS}/residual_h96x8_all81_20260920}"
CACHE_DIR="${CACHE_DIR:-${RUNS}/frozen_h96x8}"
HEAD="${HEAD:-${RUNS}/head_h96x8_h64logmae/head_5000.pth}"

if [[ -f "${CODE_DIR}/eval_corrector.py" ]]; then
  echo "[eval] standalone corrector evaluation"
  python -u -B "${CODE_DIR}/eval_corrector.py" \
    --real-root "${REAL_ROOT:-/data/p0ab_real_h5_20260830}" \
    --checkpoint "${RESIDUAL_DIR}/model_best.pth"
else
  echo "[eval] eval_corrector.py not present in this checkout."
  echo "[eval] full dev metrics are in the Stage 2 log (EVAL_TOP lines)."
fi

echo "[eval] SPS grid scan on dev split"
CACHE_DIR="${CACHE_DIR}" HEAD="${HEAD}" TAG="${TAG:-eval}" \
  bash "${REPO_ROOT}/scripts/colleague_80pt_sps.sh"