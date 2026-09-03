#!/usr/bin/env bash
# Run Feature Engineering V2.1 in the verified remote Docker environment.
# All external locations remain explicit arguments or environment variables.
set -euo pipefail

: "${FE_SOURCE_DIR:?set FE_SOURCE_DIR (contains the runner and manifest)}"
: "${FE_KIT_DIR:?set FE_KIT_DIR (unpacked Track 1 v9 kit)}"
: "${FE_DATA_ROOT:?set FE_DATA_ROOT (directory containing real H5 files)}"
: "${FE_OUTPUT_DIR:?set FE_OUTPUT_DIR (empty output directory)}"
: "${FE_CHECKPOINT:?set FE_CHECKPOINT (sim_pretrain CNO checkpoint)}"

FE_IMAGE="${FE_IMAGE:-realpde-pytorch-h5py:0831}"
FE_MANIFEST="${FE_MANIFEST:-${FE_SOURCE_DIR}/id_seed20260901.json}"
FE_GIT_COMMIT="${FE_GIT_COMMIT:-unknown}"
FE_BATCH_SIZE="${FE_BATCH_SIZE:-12}"
FE_WORKERS="${FE_WORKERS:-2}"
FE_TRAIN_SECONDS="${FE_TRAIN_SECONDS:-1800}"
FE_EVAL_INTERVAL="${FE_EVAL_INTERVAL:-500}"

docker run --rm --gpus all \
  -e "FE_SOURCE_GIT_COMMIT=${FE_GIT_COMMIT}" \
  -v "${FE_SOURCE_DIR}:/source:ro" \
  -v "${FE_KIT_DIR}:/kit:ro" \
  -v "${FE_DATA_ROOT%/*}:/data:ro" \
  -v "${FE_OUTPUT_DIR}:/out" \
  "${FE_IMAGE}" bash -lc "
    cd /source
    python realpde_fe_v21.py --mode train \\
      --manifest '${FE_MANIFEST}' --kit-root /kit --checkpoint '${FE_CHECKPOINT}' \\
      --out-dir /out --batch-size '${FE_BATCH_SIZE}' --workers '${FE_WORKERS}' \\
      --max-train-seconds '${FE_TRAIN_SECONDS}' --eval-interval '${FE_EVAL_INTERVAL}'
  "
