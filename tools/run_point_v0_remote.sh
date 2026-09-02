#!/usr/bin/env bash
# Detached-run entrypoint. All paths are explicit; no machine-specific path is embedded in Python.
set -euo pipefail
: "${POINT_SOURCE:?set POINT_SOURCE}" "${POINT_MANIFEST:?set POINT_MANIFEST}" \
  "${POINT_DATA_ROOT:?set POINT_DATA_ROOT}" "${POINT_KIT_ROOT:?set POINT_KIT_ROOT}" "${POINT_OUT_DIR:?set POINT_OUT_DIR}"
cd "$POINT_SOURCE"
exec python realpde_point_v0.py \
  --manifest "$POINT_MANIFEST" --data-root "$POINT_DATA_ROOT" --kit-root "$POINT_KIT_ROOT" \
  --out-dir "$POINT_OUT_DIR" --workers "${POINT_WORKERS:-2}"
