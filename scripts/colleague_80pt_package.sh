#!/usr/bin/env bash
# Build the submission zip from the final checkpoints.
#
# Requires (not in git, generate with the train scripts):
#   ${RUNS}/residual_h96x8_all81_20260920/model_best.pth
#   ${RUNS}/head_h96x8_h64logmae/head_5000.pth
#   ${RUNS}/scan_bounds_h96x8.json
#   ${RUNS}/submission_f3_20260919.zip  (template zip, external artifact)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="${REPO_ROOT}/tools/colleague_80pt"
RUNS="${RUNS:-/runs}"

# The original packer hard-codes /runs paths; run it as-is to stay bit-faithful.
python -u -B "${CODE_DIR}/make_submission_h96x8.py"

ls -la "${RUNS}/submission_h_h96x8_20260920.zip"
sha256sum "${RUNS}/submission_h_h96x8_20260920.zip"