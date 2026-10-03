# Strata Qwen 3.8 Bootstrap Preflight — 2026-10-03

Status: `REVIEW_REQUIRED`
Bootstrap gate: `FAILED` (mandatory GPU/process conflict gate)
Execution commit: `a2c4f30bd928fcd9af417b13926c70a90fd1390a`

## Outcome

The host capacity checks confirmed an RTX 3090 with 24 GiB VRAM, about 62 GiB visible RAM, and about 1012 GiB free on the target filesystem. However, an existing long-running `uvicorn app.main:app` Python process is using 12082 MiB of GPU memory, and port 8080 is already listening on wildcard addresses. The process is unrelated to this deployment and was not stopped or modified. HARD CONSTRAINT H7 therefore requires stopping before Strata installation/startup.

No Strata repository was cloned, no model was downloaded, no service was started, and no curl smoke was attempted. The exact frozen configuration remains Qwen3.8-Flash-Next IQ3_XXS / 131072 context / vision OFF. No fallback or alternate model was used.

## Host and capacity checks

- OS: Ubuntu 24.04.4 LTS; kernel 7.0.0-31-generic; x86_64.
- CPU: Intel Core i7-13700KF; 24 logical CPUs.
- GPU: NVIDIA GeForce RTX 3090, 24576 MiB; driver 595.84; CUDA runtime reported 13.2. `nvidia-smi` succeeded.
- System RAM: 62 GiB total; 3.8 GiB used; 58 GiB available. Swap: 8.0 GiB total, 3.9 GiB used. A single snapshot cannot establish swap thrashing.
- Filesystem `/`: 1.9 TiB total, 1012 GiB available.
- The preferred `/opt/ai-stack` root was absent at inspection. No installation directories were created.
- Port 8080: existing TCP listeners on `0.0.0.0:8080` and `[::]:8080`; not touched.
- GPU conflict: PID 3619115, command `python3.12 ... uvicorn app.main:app --host 0.0.0.0 --port 5010`, using 12082 MiB VRAM at 0% observed GPU utilization. Not stopped or inspected beyond process metadata.
- Existing GUI GPU processes: Xorg and gnome-shell, together using 51 MiB.

## Git provenance

- Coordination repo HEAD before execution: `a2c4f30bd928fcd9af417b13926c70a90fd1390a`.
- `origin/main` matched HEAD; required commit was an ancestor.
- `git push --dry-run origin HEAD:main` succeeded before host preflight.
- The required deployment README was absent before this task; root `MEMORY.md` was also absent.
- The SSH client emitted a warning that the connection did not use a post-quantum key exchange.

## Commands and adaptations

Executed read-only host inspection via `ssh gpu`: `uname -a`, `cat /etc/os-release`, `id`, `lscpu`, `free -h`, `df -h`, `nvidia-smi`, `nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version,pstate --format=csv`, process metadata for PID 3619115, `ss -lntp` filtered for 8080, path existence checks, and Git/Python/curl version checks. No install/setup/start command was executed.

## Scope

- RealPDE scientific assets NOT accessed
- Agent framework NOT installed
- RAG NOT deployed
- public ingress NOT enabled by this task (existing unrelated wildcard listeners were observed and left untouched)
- calibration NOT run
- persistent service NOT configured

## Next action

Stop here for review. Resolve the pre-existing GPU and port conflicts outside this task, then issue a new authorized execution request. This task does not kill, pause, inspect application internals, or reconfigure the existing process/service.
