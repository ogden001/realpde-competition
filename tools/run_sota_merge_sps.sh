#!/usr/bin/env bash
set -euo pipefail

# REALPDE matched SPS launcher for the reviewed Joint@6k point predictor.
# Scientific code is fully defined in tools/train_sota_merge_sps.py.
# This launcher only validates explicit paths and starts that frozen protocol.

if [[ $# -ne 6 ]]; then
  cat >&2 <<'EOF'
usage:
  tools/run_sota_merge_sps.sh \
    DATA_ROOT CLEAN_51_12_18_MANIFEST KIT_ROOT \
    JOINT6K_BACKBONE_CKPT JOINT6K_CORRECTOR_CKPT OUT_DIR

example:
  tools/run_sota_merge_sps.sh \
    /hy-tmp/realpde_data/train_real_clean_train51_seen_dev12_386920e \
    configs/clean_baseline_v1_split.json \
    /hy-tmp/realpde_t1_kit_v9/realpde_t1_starting_kit_v9 \
    /hy-tmp/realpde_runs/realpde_joint_training_v1_4090_20260926_run2/checkpoints/backbone_joint_006000.pth \
    /hy-tmp/realpde_runs/realpde_joint_training_v1_4090_20260926_run2/checkpoints/corrector_joint_006000.pth \
    /hy-tmp/realpde_runs/realpde_sota_merge_sps_joint6k
EOF
  exit 2
fi

DATA_ROOT=$1
MANIFEST=$2
KIT_ROOT=$3
BACKBONE=$4
CORRECTOR=$5
OUT_DIR=$6

for f in "$MANIFEST" "$BACKBONE" "$CORRECTOR"; do
  [[ -f "$f" ]] || { echo "missing file: $f" >&2; exit 3; }
done
[[ -d "$DATA_ROOT" ]] || { echo "missing data root: $DATA_ROOT" >&2; exit 3; }
[[ -d "$KIT_ROOT" ]] || { echo "missing kit root: $KIT_ROOT" >&2; exit 3; }
[[ ! -e "$OUT_DIR" ]] || { echo "OUT_DIR already exists: $OUT_DIR" >&2; exit 4; }

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

python -u -B tools/train_sota_merge_sps.py \
  --data-root "$DATA_ROOT" \
  --manifest "$MANIFEST" \
  --kit-root "$KIT_ROOT" \
  --backbone-checkpoint "$BACKBONE" \
  --corrector-checkpoint "$CORRECTOR" \
  --out-dir "$OUT_DIR" \
  --workers 4 \
  --calibration-workers 12 \
  --require-cuda
