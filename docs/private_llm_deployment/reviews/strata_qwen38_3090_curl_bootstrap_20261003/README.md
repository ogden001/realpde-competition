# Strata Qwen 3.8 Bootstrap — 2026-10-03

Status: `REVIEW_REQUIRED`
Bootstrap gate: `PARTIAL` (host gate passed; setup stopped at missing build dependency)
Required commit: `a2c4f30bd928fcd9af417b13926c70a90fd1390a`
Execution commit: `9cedb64f9713c3861d9172b862aa04c70cb8c8af`

## Outcome

After the user cleared the earlier GPU-memory process, a fresh preflight showed 83 MiB / 24576 MiB VRAM used (Xorg and gnome-shell only), 59 GiB available RAM, 5.3 GiB free swap, no sustained swap-in/out in a five-second `vmstat` sample, and 1012 GiB free disk. The prior GPU process PID 3619115 is absent. The existing unrelated port 8080 wildcard listener remains; Strata is configured to use the free port 8081 and bind explicitly to `127.0.0.1`. Nothing on 8080 was touched.

The official Strata repository was cloned at `~/ai-stack/strata`, SHA `99f3dbd0b21d1401b3769e0c0d963913607f380b`, from `https://github.com/Niko1221/Strata.git`. Its setup confirmed Qwen3.8-Flash-Next, IQ3_XXS, context 131072, KV cache 8-bit, images off, and experimental speed projection off. Setup installed dependencies in Strata's virtual environment and fetched the pinned llama.cpp source revision `3cf0325`.

The current Strata release is v0.1.38. Its ready-made engine URL returned HTTP 404, so the official installer selected source compilation. It requires NVIDIA CUDA Toolkit 13.0 and reported an 8–10 GB install. The installer downloaded its CUDA repository key to `/tmp/cuda-keyring.deb`, then stopped at `sudo dpkg -i` because sudo had no terminal/password available. The CUDA toolkit, compiler, and engine were not installed/built. The user must run the documented privileged system installation, then the same setup command can resume. No NVIDIA driver changes were attempted.

No model assets were downloaded: `~/ai-stack/models` is 12 KiB. No Strata server started, and no curl smoke was attempted. No alternate model or quantization was selected.

## Current host snapshot

- OS: Ubuntu 24.04.4 LTS; kernel 7.0.0-31-generic; x86_64.
- CPU: Intel Core i7-13700KF; 24 logical CPUs.
- GPU: NVIDIA GeForce RTX 3090; 24576 MiB; driver 595.84; NVIDIA reports CUDA 13.2.
- RAM: 62 GiB total; 3.0 GiB used; 59 GiB available.
- Swap: 8.0 GiB total; 2.7 GiB used. `vmstat 1 5` showed initial 1/3 KiB/s si/so then zero for the remaining samples.
- Filesystem `/`: 1.9 TiB total; 1012 GiB available.
- Existing port 8080 listeners remain on wildcard IPv4 and IPv6. Port 8081 was free at preflight and is the selected local Strata port.
- Strata setup PID 423159 exited after reporting the sudo error. Setup log: `~/ai-stack/logs/setup.log`.

## Commands and adaptations

Official upstream instructions: [AI_SETUP.md](https://github.com/Niko1221/Strata/blob/main/docs/AI_SETUP.md). Exact setup command:

```bash
./setup.sh --yes --family qwen --model IQ3_XXS --context 131072 \
  --vision no --kv int8 --port 8081 --host 127.0.0.1 \
  --data-dir "$HOME/ai-stack/models" --no-start
```

The `--port 8081` adaptation follows upstream guidance for an occupied 8080. `--host 127.0.0.1` is explicit. No setup flags change the frozen model semantics.

## Scope

- RealPDE scientific assets NOT accessed
- Agent framework NOT installed
- RAG NOT deployed
- public ingress NOT enabled by this task; Strata is not running, and the pre-existing unrelated wildcard listener was left untouched
- calibration NOT run
- persistent service NOT configured
- no model binaries or large logs committed

## Next action

The Strata setup guide says that when a sudo step is needed, the user should run it. Please install the official CUDA Toolkit 13.0 using the setup's pending command/official Ubuntu package flow, then tell Codex to resume the same setup command. Do not change or replace the NVIDIA driver.
