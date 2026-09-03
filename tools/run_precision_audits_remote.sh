#!/usr/bin/env bash
# Audit same-update last checkpoints for the precision-injection round.
set -euo pipefail

TASK_ROOT=${1:?task root is required}
REMOTE_KIT_DIR=${REMOTE_KIT_DIR:?set REMOTE_KIT_DIR (unpacked Track 1 v9 kit)}
REMOTE_DATA_DIR=${REMOTE_DATA_DIR:?set REMOTE_DATA_DIR (RealPDE data root)}
for SPLIT in locked ood; do
  if [[ "$SPLIT" == locked ]]; then
    MANIFEST=/task/manifests/id_seed20260901.json
    EVAL_SPLIT=final
  else
    MANIFEST=/task/manifests/ood_aoa20_seed20260901.json
    EVAL_SPLIT=ood
  fi
  for VARIANT in N0 N1 N2; do
    OUT_DIR="/task/precision_${SPLIT}_${VARIANT}_s20260901"
    echo "START ${SPLIT} ${VARIANT} $(date -Is)"
    docker run --rm --gpus all --shm-size=8g \
      -v "${TASK_ROOT}:/task" \
      -v "${REMOTE_KIT_DIR}:/kit:ro" \
      -v "${REMOTE_DATA_DIR}:/data:ro" \
      realpde-pytorch-h5py:0831 \
      python /task/source/realpde_loss_official_v9.py \
        --mode audit --real-root /data/p0ab_real_h5_20260830 \
        --manifest "$MANIFEST" --eval-split "$EVAL_SPLIT" \
        --kit-root /kit --checkpoint "/task/precision_${VARIANT}_s20260901/model_latest.pth" \
        --out-dir "$OUT_DIR" --variant "$VARIANT" --weights-json "/task/configs/precision_weights_${VARIANT}.json" \
        --seed 20260901 --batch-size 8 --workers 2 --skip-gradient-audit
    echo "DONE ${SPLIT} ${VARIANT} $(date -Is)"
  done
done
