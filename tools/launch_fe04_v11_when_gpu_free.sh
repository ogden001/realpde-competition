#!/usr/bin/env bash
# Cooperative FE-04 queue: preserve frozen batch=18 and yield to foreign jobs.
set -euo pipefail
: "${FE04_SOURCE_DIR:?}"; : "${FE04_KIT_DIR:?}"; : "${FE04_DATA_PARENT:?}"; : "${FE04_OUTPUT_DIR:?}"; : "${FE04_MANIFEST:?}"; : "${FE04_CHECKPOINT_IN_CONTAINER:?}"; : "${FE04_V21_OUT:?}"
FE04_IMAGE="${FE04_IMAGE:-realpde-pytorch-h5py:0831}"; FE04_IDLE_POLL_SECONDS="${FE04_IDLE_POLL_SECONDS:-20}"; FE04_IDLE_CONFIRMATIONS="${FE04_IDLE_CONFIRMATIONS:-3}"; FE04_SESSION_SECONDS="${FE04_SESSION_SECONDS:-0}"; FE04_MONITOR_SECONDS="${FE04_MONITOR_SECONDS:-5}"; FE04_STOP_GRACE_SECONDS="${FE04_STOP_GRACE_SECONDS:-120}"
gpu_pids(){ nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk '/^[0-9]+$/ {print $1}' | sort -u; }
wait_quiet(){ local n=0; while :; do if [ -z "$(gpu_pids)" ]; then n=$((n+1)); [ "$n" -ge "$FE04_IDLE_CONFIRMATIONS" ] && return; else n=0; fi; sleep "$FE04_IDLE_POLL_SECONDS"; done; }
run_python(){ docker run --rm --gpus all -v "$FE04_SOURCE_DIR:/source:ro" -v "$FE04_KIT_DIR:/kit:ro" -v "$FE04_DATA_PARENT:/data:ro" -v "$FE04_OUTPUT_DIR:/out" -v "$FE04_V21_OUT:/v21out:ro" "$FE04_IMAGE" bash -lc "cd /source && python realpde_fe04_raw_spatial8.py --manifest '$FE04_MANIFEST' --kit-root /kit --checkpoint '$FE04_CHECKPOINT_IN_CONTAINER' --v21-out /v21out --out-dir /out --batch-size 18 --workers '${FE04_WORKERS:-2}' --mode $1"; }
wait_quiet
if [ ! -f "$FE04_OUTPUT_DIR/preflight_b18.json" ]; then
  if ! run_python preflight >"$FE04_OUTPUT_DIR/preflight.log" 2>&1; then echo 'EXCLUSIVE_PREFLIGHT_FAILED; batch fallback forbidden; LIMITED_COMPARABILITY' >"$FE04_OUTPUT_DIR/queue_status.txt"; exit 1; fi
fi
while [ ! -f "$FE04_OUTPUT_DIR/FE-04-RawSpatial8/summary.json" ]; do
  wait_quiet; name="realpde-fe04-$RANDOM"; log="$FE04_OUTPUT_DIR/queue_fe04.log"
  docker run -d --name "$name" --gpus all -v "$FE04_SOURCE_DIR:/source:ro" -v "$FE04_KIT_DIR:/kit:ro" -v "$FE04_DATA_PARENT:/data:ro" -v "$FE04_OUTPUT_DIR:/out" -v "$FE04_V21_OUT:/v21out:ro" "$FE04_IMAGE" bash -lc "cd /source && exec python realpde_fe04_raw_spatial8.py --manifest '$FE04_MANIFEST' --kit-root /kit --checkpoint '$FE04_CHECKPOINT_IN_CONTAINER' --v21-out /v21out --out-dir /out --batch-size 18 --workers '${FE04_WORKERS:-2}' --checkpoint-interval 100 --max-session-seconds '$FE04_SESSION_SECONDS' --mode train" >/dev/null
  while [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null || true)" = true ]; do
    own="$(docker top "$name" -eo pid,cmd 2>/dev/null | awk '/realpde_fe04_raw_spatial8.py/ {print $1}' | sort -u)"; foreign="$(comm -23 <(gpu_pids) <(printf '%s\n' "$own" | awk 'NF' | sort -u))"
    if [ -n "$foreign" ]; then echo "foreign GPU PID(s): $foreign; checkpoint/yield" >>"$log"; docker stop --time "$FE04_STOP_GRACE_SECONDS" "$name" >/dev/null || true; break; fi; sleep "$FE04_MONITOR_SECONDS"
  done
  docker logs "$name" >>"$log" 2>&1 || true; docker rm "$name" >/dev/null 2>&1 || true
done
run_python analyze >>"$FE04_OUTPUT_DIR/queue_fe04.log" 2>&1
