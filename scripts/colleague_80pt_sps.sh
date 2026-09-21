#!/usr/bin/env bash
# SPS parameter scan on the dev split (640 windows, 16 trajectories, P00 phase).
# Reference: docs/colleague_80pt_handoff/SPS.md
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="${REPO_ROOT}/tools/colleague_80pt"

RUNS="${RUNS:-/runs}"
CACHE_DIR="${CACHE_DIR:-${RUNS}/frozen_h96x8}"
HEAD="${HEAD:-${RUNS}/head_h96x8_h64logmae/head_5000.pth}"
TAG="${TAG:-h96x8}"
SCAN_SCRIPT="${RUNS}/scan_bounds_${TAG}.py"

sed "s|/runs/frozen_cache|${CACHE_DIR}|" "${CODE_DIR}/scan_bounds.py" > "${SCAN_SCRIPT}"

python -u -B "${SCAN_SCRIPT}" \
  --head "${HEAD}" \
  --hidden 64 \
  --blocks 2 \
  --tag "${TAG}"

echo "[done] scan output: ${RUNS}/scan_bounds_${TAG}.json"