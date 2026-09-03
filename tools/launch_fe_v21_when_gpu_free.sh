#!/usr/bin/env bash
# Opportunistic, preemptible remote queue for Feature Engineering V2.1.
#
# This is deliberately cooperative rather than pretending an idle GPU is an
# exclusive reservation. Each group runs in short leases, writes atomic resume
# checkpoints, and yields when a foreign GPU process appears.
set -euo pipefail

: "${FE_SOURCE_DIR:?}"
: "${FE_KIT_DIR:?}"
: "${FE_DATA_PARENT:?}"
: "${FE_OUTPUT_DIR:?}"
: "${FE_MANIFEST:?}"
: "${FE_CHECKPOINT_IN_CONTAINER:?}"
: "${FE_GIT_COMMIT:?}"

FE_IMAGE="${FE_IMAGE:-realpde-pytorch-h5py:0831}"
FE_IDLE_CONFIRMATIONS="${FE_IDLE_CONFIRMATIONS:-3}"
FE_IDLE_POLL_SECONDS="${FE_IDLE_POLL_SECONDS:-20}"
FE_SESSION_SECONDS="${FE_SESSION_SECONDS:-300}"
FE_CHECKPOINT_INTERVAL="${FE_CHECKPOINT_INTERVAL:-100}"
FE_MONITOR_SECONDS="${FE_MONITOR_SECONDS:-5}"
FE_STOP_GRACE_SECONDS="${FE_STOP_GRACE_SECONDS:-120}"

gpu_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk '/^[0-9]+$/ {print $1}' | sort -u
}

wait_for_quiet_gpu() {
  local confirmations=0
  while true; do
    if [ -z "$(gpu_pids)" ]; then
      confirmations=$((confirmations + 1))
      if [ "${confirmations}" -ge "${FE_IDLE_CONFIRMATIONS}" ]; then
        echo "GPU idle and stable for ${confirmations} checks" >&2
        return
      fi
    else
      confirmations=0
    fi
    sleep "${FE_IDLE_POLL_SECONDS}"
  done
}

label_for() {
  case "$1" in
    baseline) echo "FE-00-CNO-Baseline" ;;
    raw_control) echo "FE-00R-ResidualRaw-Control" ;;
    temporal) echo "FE-01-Temporal" ;;
    spatial) echo "FE-02-SpatialPhysics" ;;
    pixel) echo "FE-03-PixelPosition" ;;
    *) return 2 ;;
  esac
}

select_common_batch() {
  local candidate peak
  for candidate in 20 18 16 12; do
    if docker run --rm --gpus all \
      -e "FE_SOURCE_GIT_COMMIT=${FE_GIT_COMMIT}" \
      -v "${FE_SOURCE_DIR}:/source:ro" \
      -v "${FE_KIT_DIR}:/kit:ro" \
      -v "${FE_DATA_PARENT}:/data:ro" \
      -v "${FE_OUTPUT_DIR}:/out" \
      "${FE_IMAGE}" bash -lc "
        cd /source && exec python realpde_fe_v21.py --mode preflight \\
          --manifest '${FE_MANIFEST}' --kit-root /kit --checkpoint '${FE_CHECKPOINT_IN_CONTAINER}' \\
          --out-dir '/out/preflight_b${candidate}' --batch-size '${candidate}' --workers 0
      " >&2; then
      peak="$(python3 -c "import json; p=json.load(open('${FE_OUTPUT_DIR}/preflight_b${candidate}/preflight.json')); print(max(r['peak_gib'] for r in p['rows']))")"
      if awk "BEGIN {exit !(${peak} <= 22.0)}"; then
        echo "${candidate}"
        return
      fi
    fi
  done
  return 1
}

run_slice() {
  local group="$1" label="$2" name log_file resume_fragment own_pids foreign_pids exit_code
  if [ -f "${FE_OUTPUT_DIR}/${label}/progress.pth" ]; then
    resume_fragment="--resume /out/${label}/progress.pth"
  else
    resume_fragment="--restart-incomplete"
  fi
  name="realpde-fe-v21-${group}-$$"
  log_file="${FE_OUTPUT_DIR}/queue_${group}.log"
  echo "starting ${group} lease (${FE_SESSION_SECONDS}s; ${resume_fragment})" | tee -a "${log_file}" >&2
  docker run -d --name "${name}" --gpus all \
    -e "FE_SOURCE_GIT_COMMIT=${FE_GIT_COMMIT}" \
    -v "${FE_SOURCE_DIR}:/source:ro" \
    -v "${FE_KIT_DIR}:/kit:ro" \
    -v "${FE_DATA_PARENT}:/data:ro" \
    -v "${FE_OUTPUT_DIR}:/out" \
    "${FE_IMAGE}" bash -lc "
      cd /source && exec python realpde_fe_v21.py --mode train \\
        --manifest '${FE_MANIFEST}' --kit-root /kit --checkpoint '${FE_CHECKPOINT_IN_CONTAINER}' \\
        --out-dir /out --batch-size '${FE_SELECTED_BATCH}' --workers '${FE_WORKERS:-2}' \\
        --max-train-seconds '${FE_TRAIN_SECONDS:-1800}' --eval-interval '${FE_EVAL_INTERVAL:-500}' \\
        --checkpoint-interval '${FE_CHECKPOINT_INTERVAL}' --max-session-seconds '${FE_SESSION_SECONDS}' \\
        --experiments '${group}' ${resume_fragment}
    " >/dev/null

  while [ "$(docker inspect -f '{{.State.Running}}' "${name}" 2>/dev/null || true)" = "true" ]; do
    own_pids="$(docker top "${name}" -eo pid,cmd 2>/dev/null | awk '/realpde_fe_v21.py/ {print $1}' | sort -u)"
    foreign_pids="$(comm -23 <(gpu_pids) <(printf '%s\n' "${own_pids}" | awk 'NF' | sort -u))"
    if [ -n "${foreign_pids}" ]; then
      echo "foreign GPU PID(s) detected: ${foreign_pids}; requesting checkpoint and yield" | tee -a "${log_file}" >&2
      docker stop --time "${FE_STOP_GRACE_SECONDS}" "${name}" >/dev/null || true
      break
    fi
    sleep "${FE_MONITOR_SECONDS}"
  done
  exit_code="$(docker inspect -f '{{.State.ExitCode}}' "${name}" 2>/dev/null || echo unknown)"
  docker logs "${name}" >>"${log_file}" 2>&1 || true
  docker rm "${name}" >/dev/null 2>&1 || true
  echo "${group} lease exited code=${exit_code}" | tee -a "${log_file}" >&2
}

wait_for_quiet_gpu
FE_SELECTED_BATCH="$(select_common_batch)" || { echo "No common safe FE batch found" >&2; exit 1; }
echo "selected common batch=${FE_SELECTED_BATCH}" >&2

for FE_GROUP in baseline raw_control temporal spatial pixel; do
  FE_LABEL="$(label_for "${FE_GROUP}")"
  while [ ! -f "${FE_OUTPUT_DIR}/${FE_LABEL}/summary.json" ]; do
    wait_for_quiet_gpu
    run_slice "${FE_GROUP}" "${FE_LABEL}"
  done
  echo "completed ${FE_GROUP}" >&2
done
echo "all FE groups completed" >&2
