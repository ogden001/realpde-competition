#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?set DATA_ROOT to released real-PIV H5 root}"
: "${MANIFEST:?set MANIFEST to frozen 50/16 manifest}"
: "${KIT_ROOT:?set KIT_ROOT to official Track 1 v9 starting-kit root}"
: "${VALIDATION_CHECKPOINT:?set VALIDATION_CHECKPOINT to exact P0-A validation update-30900 checkpoint}"
: "${OUT_ROOT:?set OUT_ROOT to a new SPS screen output directory}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -e "$OUT_ROOT" ]]; then
  echo "refusing to reuse existing OUT_ROOT: $OUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUT_ROOT"

STATUS_JSON="$OUT_ROOT/status.json"
write_status() {
  python - "$STATUS_JSON" "$1" "$2" <<'PY'
import json, sys, time
from pathlib import Path
path = Path(sys.argv[1])
path.write_text(json.dumps({"state": sys.argv[2], "detail": sys.argv[3], "updated_at": time.time()}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

on_error() {
  rc=$?
  write_status FAILED "launcher exited with code ${rc}"
  exit "$rc"
}
trap on_error ERR

write_status SNAPSHOT "inventory runtime and exact validation checkpoint"
python tools/realpde_runtime_context.py snapshot \
  --data-root "$DATA_ROOT" \
  --kit-root "$KIT_ROOT" \
  --checkpoint "$VALIDATION_CHECKPOINT" \
  --output "$OUT_ROOT/runtime_snapshot.json"

write_status RUNNING "official-v9 SPS bounds screen on frozen 50/16 dev"
python tools/sota_sps_screen.py \
  --manifest "$MANIFEST" \
  --kit-root "$KIT_ROOT" \
  --checkpoint "$VALIDATION_CHECKPOINT" \
  --expected-iteration 30900 \
  --out "$OUT_ROOT/sps_screen.json"

write_status REVIEW_REQUIRED "see $OUT_ROOT/sps_screen.json"
trap - ERR
