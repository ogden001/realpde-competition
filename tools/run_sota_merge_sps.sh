#!/usr/bin/env bash
set -euo pipefail

# REALPDE final SPS optimization launcher.
# Scientific code lives in tools/train_sota_merge_sps.py. This file only
# resolves explicit inputs and launches it. No Codabench/locked-final access.

if [[ $# -ne 8 ]]; then
  cat >&2 <<'EOF'
usage:
  tools/run_sota_merge_sps.sh \
    DATA_ROOT CLEAN_51_12_MANIFEST AOA10_MANIFEST AOA10_SPLIT KIT_ROOT \
    JOINT_BACKBONE_CKPT JOINT_CORRECTOR_CKPT OUT_DIR

example:
  tools/run_sota_merge_sps.sh \
    /data/real \
    configs/clean_baseline_v1_split.json \
    <aoa10_manifest.json> dev \
    /third_party/realpdebench \
    /hy-tmp/.../checkpoints/backbone_joint_006000.pth \
    /hy-tmp/.../checkpoints/corrector_joint_006000.pth \
    /hy-tmp/realpde_runs/final_sps
EOF
  exit 2
fi

DATA_ROOT=$1
MANIFEST=$2
AOA10_MANIFEST=$3
AOA10_SPLIT=$4
KIT_ROOT=$5
BACKBONE=$6
CORRECTOR=$7
OUT_DIR=$8

for f in "$MANIFEST" "$AOA10_MANIFEST" "$BACKBONE" "$CORRECTOR"; do
  [[ -f "$f" ]] || { echo "missing file: $f" >&2; exit 3; }
done
[[ -d "$DATA_ROOT" ]] || { echo "missing data root: $DATA_ROOT" >&2; exit 3; }
[[ -d "$KIT_ROOT" ]] || { echo "missing kit root: $KIT_ROOT" >&2; exit 3; }
[[ ! -e "$OUT_DIR" ]] || { echo "OUT_DIR already exists: $OUT_DIR" >&2; exit 4; }

python -u -B tools/train_sota_merge_sps.py \
  --data-root "$DATA_ROOT" \
  --manifest "$MANIFEST" \
  --aoa10-manifest "$AOA10_MANIFEST" \
  --aoa10-split "$AOA10_SPLIT" \
  --kit-root "$KIT_ROOT" \
  --backbone-checkpoint "$BACKBONE" \
  --corrector-checkpoint "$CORRECTOR" \
  --out-dir "$OUT_DIR" \
  --workers 4 \
  --require-cuda
