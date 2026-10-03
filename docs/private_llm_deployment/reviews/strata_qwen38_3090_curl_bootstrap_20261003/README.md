# Strata Qwen 3.8 Bootstrap — CUDA Resume Check

Status: `REVIEW_REQUIRED`
Bootstrap gate: `PARTIAL` (approved Strata setup preserved; CUDA prerequisite absent)
Required task commit: `f623049fef3474d95c59a5172c0e6739b0064fb9`
Execution commit: `f623049fef3474d95c59a5172c0e6739b0064fb9`

## Outcome

The user reported that CUDA Toolkit 13.0 had been installed. On the configured GPU SSH target, checks did not find it: `which nvcc` returned no path, `nvcc --version` returned command-not-found, `/usr/local/cuda*` did not exist, and dpkg reported no `cuda-toolkit-13-0`, `cuda-compiler-13-0`, or `cuda-nvcc-13-0` package. `sudo -n true` returned `sudo: a password is required`. Per the continuation NEXT_ACTION, Strata setup was not resumed; no attempt was made to type or request a password. `/tmp/cuda-keyring.deb` from the earlier official setup still exists.

The NVIDIA driver is healthy and unchanged at 595.84 (`nvidia-smi`, CUDA runtime 13.2). GPU memory use is 83 MiB / 24576 MiB (Xorg and gnome-shell only). RAM is 59 GiB available; disk has 1012 GiB free. The previous GPU blocker remains cleared. Port 8080 remains occupied by the unrelated wildcard listeners; 8081 is free and remains the planned local-only Strata endpoint.

The existing official Strata repository is intact at `~/ai-stack/strata`, SHA `99f3dbd0b21d1401b3769e0c0d963913607f380b`, clean worktree, upstream `https://github.com/Niko1221/Strata.git`. Its prior setup stopped when v0.1.38's ready-made engine lookup returned HTTP 404 and source compilation required CUDA Toolkit 13.0. Existing setup log is `~/ai-stack/logs/setup.log`. No model assets were downloaded; `~/ai-stack/models` is 12 KiB. No Strata engine/server started and curl smoke was not run.

## Exact continuation command

Once the toolkit is present and `nvcc --version` confirms CUDA 13.0, resume the existing install with:

```bash
cd "$HOME/ai-stack/strata"
./setup.sh --yes --family qwen --model IQ3_XXS --context 131072 \
  --vision no --kv int8 --port 8081 --host 127.0.0.1 \
  --data-dir "$HOME/ai-stack/models" --no-start
```

If the toolkit is still absent and sudo remains interactive, the task file gives the exact user-run commands:

```bash
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-0
```

Only `cuda-toolkit-13-0` is authorized; do not install CUDA or NVIDIA driver metapackages or change the existing driver.

## Scope

- RealPDE scientific assets NOT accessed
- Agent framework NOT installed
- RAG NOT deployed
- public ingress NOT enabled; Strata is not running and existing unrelated wildcard listener was left untouched
- calibration NOT run
- persistent service NOT configured
- no model binaries or large logs committed
