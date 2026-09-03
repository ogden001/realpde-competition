#!/usr/bin/env bash
# Run bounded official-v9 screening/confirmation inside the approved GPU image.
# Usage: run_loss_v9_remote.sh TASK_ROOT KIND [VARIANTS...]
set -euo pipefail

TASK_ROOT=${1:?task root is required}
KIND=${2:?kind is required}
REMOTE_KIT_DIR=${REMOTE_KIT_DIR:?set REMOTE_KIT_DIR (unpacked Track 1 v9 kit)}
REMOTE_DATA_DIR=${REMOTE_DATA_DIR:?set REMOTE_DATA_DIR (RealPDE data root)}
shift 2

case "$KIND" in
  screen)
    UPDATES=300
    EVAL_INTERVAL=300
    MANIFEST=/task/manifests/id_seed20260901.json
    EVAL_SPLIT=dev
    OUT_PREFIX=screen
    ;;
  confirm)
    UPDATES=1200
    EVAL_INTERVAL=300
    MANIFEST=/task/manifests/id_seed20260901.json
    EVAL_SPLIT=dev
    OUT_PREFIX=confirm
    ;;
  ood)
    UPDATES=600
    EVAL_INTERVAL=300
    MANIFEST=/task/manifests/ood_aoa20_seed20260901.json
    EVAL_SPLIT=dev
    OUT_PREFIX=ood_train
    ;;
  long)
    # Four sequential 90-minute training continuations.  The Python loop owns
    # the clock; UPDATES is merely a safely unreachable ceiling.
    UPDATES=50000
    EVAL_INTERVAL=600
    MAX_TRAIN_SECONDS=5400
    MANIFEST=/task/manifests/id_seed20260901.json
    EVAL_SPLIT=dev
    OUT_PREFIX=long
    ;;
  precision)
    UPDATES=4100
    EVAL_INTERVAL=820
    MANIFEST=/task/manifests/id_seed20260901.json
    EVAL_SPLIT=dev
    OUT_PREFIX=precision
    ;;
  final)
    UPDATES=0
    EVAL_INTERVAL=1
    MANIFEST=/task/manifests/id_seed20260901.json
    EVAL_SPLIT=final
    OUT_PREFIX=final_screen
    ;;
  long-final)
    UPDATES=0
    EVAL_INTERVAL=1
    MANIFEST=/task/manifests/id_seed20260901.json
    EVAL_SPLIT=final
    OUT_PREFIX=long_final
    SKIP_GRADIENT_AUDIT=1
    ;;
  *)
    echo "unknown run kind: $KIND" >&2
    exit 64
    ;;
esac

for SPEC in "$@"; do
  VARIANT=${SPEC%%:*}
  SEED=${SPEC#*:}
  if [[ "$VARIANT" == "$SEED" ]]; then SEED=20260901; fi
  OUT_DIR="/task/${OUT_PREFIX}_${VARIANT}_s${SEED}"
  MODE=train
  CHECKPOINT=/data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth
  if [[ "$KIND" == "final" || "$KIND" == "long-final" ]]; then
    MODE=audit
    if [[ "$KIND" == "final" ]]; then
      CHECKPOINT="/task/screen_${VARIANT}_s20260901/model_best.pth"
    else
      CHECKPOINT="/task/long_${VARIANT}_s20260901/model_best.pth"
    fi
  fi
  EXTRA_ARGS=()
  if [[ -n "${MAX_TRAIN_SECONDS:-}" ]]; then
    EXTRA_ARGS=(--max-train-seconds "$MAX_TRAIN_SECONDS")
  fi
  if [[ -n "${SKIP_GRADIENT_AUDIT:-}" ]]; then
    EXTRA_ARGS+=(--skip-gradient-audit)
  fi
  if [[ "$KIND" == "precision" ]]; then
    EXTRA_ARGS+=(--weights-json "/task/configs/precision_weights_${VARIANT}.json")
  fi
  echo "START ${KIND} ${VARIANT} seed=${SEED} $(date -Is)"
  docker run --rm --gpus all --shm-size=8g \
    -v "${TASK_ROOT}:/task" \
    -v "${REMOTE_KIT_DIR}:/kit:ro" \
    -v "${REMOTE_DATA_DIR}:/data:ro" \
    realpde-pytorch-h5py:0831 \
    python /task/source/realpde_loss_official_v9.py \
      --mode "${MODE}" --real-root /data/p0ab_real_h5_20260830 \
      --manifest "${MANIFEST}" --eval-split "${EVAL_SPLIT}" \
      --kit-root /kit --checkpoint "${CHECKPOINT}" \
      --out-dir "${OUT_DIR}" --variant "${VARIANT}" --seed "${SEED}" \
      --updates "${UPDATES}" --eval-interval "${EVAL_INTERVAL}" --batch-size 8 --workers 2 --lr 1e-5 \
      "${EXTRA_ARGS[@]}"
  echo "DONE ${KIND} ${VARIANT} seed=${SEED} $(date -Is)"
done
