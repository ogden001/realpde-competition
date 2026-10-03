# Strata Qwen 3.8 Bootstrap — CUDA Build and Model Download — 2026-10-04

Status: `REVIEW_REQUIRED`
Bootstrap gate: `PARTIAL`
Required supplement commit: `de9963e25c465f32cddef10223f8cd4883fbbc25`
Execution commit: `de9963e25c465f32cddef10223f8cd4883fbbc25`

## Outcome

The user-installed CUDA Toolkit 13.0 is present: `cuda-toolkit-13-0` package version 13.0.3-1, `nvcc` version 13.0.88 at `/usr/local/cuda-13.0/bin/nvcc`. PATH and LD_LIBRARY_PATH were set for the setup shell only; no persistent or system-wide environment files were changed. `nvidia-smi` remained healthy at driver 595.84 (runtime reports CUDA 13.2), unchanged from prior evidence.

The pinned official Strata v0.1.38 source build completed all 129 build targets. The executable is `~/ai-stack/strata/engine/strata` (39 MiB). The existing Strata worktree remains clean at SHA `99f3dbd0b21d1401b3769e0c0d963913607f380b`.

The official installer then failed during the first IQ3_XXS shard HEAD request with `cannot reach huggingface.co (<urlopen error [Errno 101] Network is unreachable>)`; its setup log says to check the internet connection and run it again. The model directory contains no files, and the service was not started. Per NEXT_ACTION, execution stopped at this failure; no mirror, alternate downloader, model, or engine was substituted.

## Configuration and command

- Model: Qwen3.8-Flash-Next / IQ3_XXS / context 131072.
- Vision: off. KV: 8-bit / `int8`. Experimental speed projection: off. Normal Strata MTP setup path.
- Endpoint: planned local-only `127.0.0.1:8081`; port 8081 was free and no service is listening there. Existing wildcard listener on 8080 remains untouched.
- Data root: `~/ai-stack/models`; expected model subdirectory: `~/ai-stack/models/models/IQ3_XXS`, no model artifacts present.
- Setup log: `~/ai-stack/logs/setup-resume.log`; setup PID 432102 exited after the download failure.

Exact resumed setup command (PATH additions scoped to this process):

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
cd "$HOME/ai-stack/strata"
./setup.sh --yes --family qwen --model IQ3_XXS --context 131072 \
  --vision no --kv int8 --port 8081 --host 127.0.0.1 \
  --data-dir "$HOME/ai-stack/models" --no-start
```

## Runtime snapshot at stop

- GPU: RTX 3090, 83 / 24576 MiB used, 0% utilization; Xorg and gnome-shell only.
- RAM: 62 GiB total, 3.0 GiB used, 59 GiB available; swap 8 GiB total / 2.7 GiB used.
- Root filesystem: 1.9 TiB total / 1001 GiB available.
- Strata server PID: none. Port 8081 listener: none. Port 8080: existing wildcard IPv4 and IPv6 listeners, untouched.
- Model file size: no model files downloaded. Large raw setup log remains on the host and is not committed.

## API checks

Health, `/v1/models`, and one Chinese `/v1/chat/completions` request were not run because no model was downloaded and no service was started.

## Scope

- RealPDE scientific assets NOT accessed
- Agent framework NOT installed
- RAG NOT deployed
- public ingress NOT enabled by this task
- calibration NOT run
- persistent service NOT configured
- NVIDIA driver NOT changed
- no model binaries, caches, credentials, or large raw logs committed

## Blocker

The GPU host's official Strata model download could not reach `huggingface.co`. Once outbound network access to the model host is available, the same pinned Strata setup command is resumable; no reinstallation or variant change is needed.
