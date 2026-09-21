#!/usr/bin/env bash
# Smoke inference for the packaged submission.
#
# Unpacks the submission zip into a temp dir, copies the official bench.py,
# and runs it in the same PyTorch runtime image. Expected output contains:
#   finite True  lo<=hi True  not_fallback True
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZIP="${ZIP:-${REPO_ROOT}/submission_h_h96x8_20260920.zip}"
BENCH_PY="${BENCH_PY:-${REPO_ROOT}/tools/colleague_80pt/bench.py}"
IMAGE="${IMAGE:-pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime}"

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

unzip -q "${ZIP}" -d "${WORK}"
cp "${BENCH_PY}" "${WORK}/bench.py"

docker run --rm --gpus all -v "${WORK}:/app" "${IMAGE}" \
  python -u /app/bench.py