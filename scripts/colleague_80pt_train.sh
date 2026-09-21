#!/usr/bin/env bash
# RealPDE Track 1 - 80.078849 solution training pipeline.
# Reference: docs/colleague_80pt_handoff/TRAINING.md
#
# Usage:
#   bash scripts/colleague_80pt_train.sh cno
#   bash scripts/colleague_80pt_train.sh residual
#   bash scripts/colleague_80pt_train.sh head
#   bash scripts/colleague_80pt_train.sh all
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="${REPO_ROOT}/tools/colleague_80pt"

REAL_ROOT="${REAL_ROOT:-/data/p0ab_real_h5_20260830}"
BENCH_ROOT="${BENCH_ROOT:-/third_party}"
RUNS="${RUNS:-/runs}"
BASE_CNO="${BASE_CNO:-/data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth}"

CNO_OUT="${CNO_OUT:-${RUNS}/cno_final_all81.pt}"
RESIDUAL_OUT="${RESIDUAL_OUT:-${RUNS}/residual_h96x8_all81_20260920}"
CACHE_OUT="${CACHE_OUT:-${RUNS}/frozen_h96x8}"
HEAD_OUT="${HEAD_OUT:-${RUNS}/head_h96x8_h64logmae}"

STAGE="${1:-all}"

run_cno() {
  echo "[stage1] CNO all81 fine-tune"
  python -u -B "${CODE_DIR}/final_all81.py" \
    --stride 1 \
    --updates 8723 \
    --batch 8 \
    --lr 1e-4 \
    --loss decomp \
    --out "${CNO_OUT}"
}

run_residual() {
  echo "[stage2] residual corrector h96/b2/38400"
  python -u -B "${CODE_DIR}/residual_multi.py" \
    --real-root "${REAL_ROOT}" \
    --checkpoint "${CNO_OUT}" \
    --realpdebench-root "${BENCH_ROOT}" \
    --base-model cno \
    --out-dir "${RESIDUAL_OUT}" \
    --updates 38400 \
    --hidden 96 \
    --blocks 2 \
    --batch-size 8 \
    --stride 20 \
    --train-on-all \
    --train-alpha 1.0 \
    --bound-abs 0.0075 \
    --bound-rel 0.0075 \
    --max-delta 0.04
}

run_head() {
  echo "[stage3a] frozen feature cache (stride=5, all-data)"
  python -u -B "${CODE_DIR}/cache_frozen.py" \
    --stride 5 \
    --all-data \
    --champion "${RESIDUAL_OUT}/model_best.pth" \
    --out "${CACHE_OUT}"

  echo "[stage3b] uncertainty head h64/b2/logmae/6000"
  python -u -B "${CODE_DIR}/train_head_fast.py" \
    --cache "${CACHE_OUT}" \
    --hidden 64 \
    --blocks 2 \
    --loss logmae \
    --updates 6000 \
    --eval-every 500 \
    --out "${HEAD_OUT}"
}

case "${STAGE}" in
  cno)      run_cno ;;
  residual) run_residual ;;
  head)     run_head ;;
  all)      run_cno; run_residual; run_head ;;
  *) echo "unknown stage: ${STAGE} (expected cno|residual|head|all)" >&2; exit 2 ;;
esac

echo "[done] stage=${STAGE}"