# NEXT_ACTION — Resume Strata Qwen Bootstrap after CUDA Toolkit

Status: `IMPLEMENT_AND_EXECUTE_AUTHORIZED / REVIEW_REQUIRED`

`REQUIRED_BASE_COMMIT = 9fe5244e70732e9621eb2b6f6dbba6aa025bfbcb`

This is a continuation of the first local-LLM deployment only. Preserve the existing installation/evidence and resume from the CUDA-toolkit blocker. Do not reinstall from scratch unless the existing Strata tree is corrupt.

## Frozen target

```text
curl
  ↓ OpenAI-compatible HTTP API
Strata v0.1.38 source build on Linux
  ↓
Qwen3.8-Flash-Next / IQ3_XXS / 128K / vision OFF / INT8 KV
  ↓
RTX 3090 24 GB + 64 GB RAM
```

Known approved environment adaptation:

- Strata local port: `8081` because unrelated port 8080 remains occupied.
- Binding must remain exactly local-only: `127.0.0.1:8081`.

Known verified upstream state:

- Strata repo: `~/ai-stack/strata`
- Strata SHA: `99f3dbd0b21d1401b3769e0c0d963913607f380b`
- Strata release: `v0.1.38`
- model/data dir: `~/ai-stack/models`
- setup log: `~/ai-stack/logs/setup.log`
- previous setup command:

```bash
./setup.sh --yes --family qwen --model IQ3_XXS --context 131072 \
  --vision no --kv int8 --port 8081 --host 127.0.0.1 \
  --data-dir "$HOME/ai-stack/models" --no-start
```

Previous evidence established that Linux v0.1.38 fell back to the official source-build path and requires CUDA Toolkit 13.0. The existing NVIDIA driver is 595.84 and must NOT be replaced or downgraded.

---

# Privileged prerequisite

The only authorized system-level package installation for this continuation is:

```text
cuda-toolkit-13-0
```

Do NOT install these broader meta-packages:

```text
cuda
cuda-13-0
nvidia-driver-*
```

Do not change the current NVIDIA driver.

If CUDA Toolkit 13.0 is not yet installed and Codex cannot use sudo non-interactively, STOP and return the exact manual commands below for the user to run:

```bash
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-0
```

If `/tmp/cuda-keyring.deb` no longer exists, use the current official NVIDIA/Strata setup flow to recreate/download the repository key package; do not download an arbitrary third-party package.

After the user installs it, continue automatically from the checks below. Do not ask for further confirmation unless a HARD CONSTRAINT would change.

---

# HARD CONSTRAINTS

1. **Model is frozen:** `Qwen3.8-Flash-Next / IQ3_XXS / 131072 / vision OFF / INT8 KV`.
2. **Inference engine is frozen:** official Strata; do not patch Strata core/kernel/model code.
3. **Driver is frozen:** do not install/upgrade/downgrade NVIDIA driver.
4. **Port is frozen for this host:** `127.0.0.1:8081` only.
5. Do not kill or reconfigure the unrelated process/service on port 8080.
6. Do not install any Agent framework.
7. Do not install RAG/vector DB/embedding/reranker/Web UI/model router/cloud fallback.
8. Do not add Docker/Kubernetes/Nginx/HTTPS/systemd/persistence in this task.
9. Do not run Strata calibration or tuning sweeps.
10. Do not compare other model/quantization/context variants.
11. Do not access/modify/copy/delete RealPDE scientific assets, datasets, checkpoints, training jobs, or scientific code.
12. Do not use destructive cleanup (`git reset --hard`, broad `rm -rf`, `git clean -fdx`, etc.).
13. Do not commit credentials, model binaries, caches, or large raw logs.
14. Final state is always `REVIEW_REQUIRED`; do not continue into Agent/RAG/product work.

If any required fix would violate these constraints, STOP and preserve evidence.

---

# Preflight before resume

From the RealPDE coordination repo:

```bash
git status --short
git fetch origin
git pull --rebase origin main
git rev-parse HEAD
git rev-parse origin/main
git merge-base --is-ancestor 9fe5244e70732e9621eb2b6f6dbba6aa025bfbcb HEAD
git push --dry-run origin HEAD:main
```

Require:

- clean/known worktree;
- `HEAD == origin/main`;
- base commit is an ancestor;
- dry-run push succeeds.

Then re-check host state:

```bash
nvidia-smi
free -h
df -h
ss -lntp | grep -E ':(8080|8081)\b' || true
```

Require:

- RTX 3090 available with no unknown material GPU workload;
- roughly 60 GiB or more visible RAM and adequate free memory;
- at least 180 GiB free disk;
- port 8081 free before Strata start;
- existing 8080 service untouched.

If a new unknown material GPU workload appears, STOP rather than killing it.

---

# Phase 1 — Verify CUDA Toolkit only

After the privileged prerequisite has been completed, capture:

```bash
which nvcc || true
nvcc --version || true
ls -ld /usr/local/cuda* 2>/dev/null || true
nvidia-smi
```

Acceptance:

- a CUDA 13.0 toolkit/compiler is available to the Strata build;
- current NVIDIA driver remains unchanged from the pre-install state unless the host itself changed outside this task;
- `nvidia-smi` remains healthy.

If toolkit is installed but Strata cannot discover `nvcc`, bounded PATH/environment adaptation to the official `/usr/local/cuda-13.0/bin` location is allowed. Do not install another CUDA version.

---

# Phase 2 — Resume official Strata setup

Use the existing official Strata tree at:

```text
~/ai-stack/strata
```

First verify it is still the approved upstream tree and clean:

```bash
cd "$HOME/ai-stack/strata"
git remote -v
git status --short
git rev-parse HEAD
```

Expected SHA:

`99f3dbd0b21d1401b3769e0c0d963913607f380b`

Do not `git pull` Strata to a newer revision in this continuation unless the pinned revision is unusable for a reason that is proven and reviewed. We want to finish the same deployment attempt.

Resume the exact frozen setup:

```bash
./setup.sh --yes --family qwen --model IQ3_XXS --context 131072 \
  --vision no --kv int8 --port 8081 --host 127.0.0.1 \
  --data-dir "$HOME/ai-stack/models" --no-start
```

Allow the official setup to:

- compile the Strata engine;
- download the frozen model assets;
- prepare the model/pack/MTP assets required by that configuration.

Do not download any alternative model variant.

If setup fails, capture the relevant tail of the official log and STOP. Do not patch Strata or invent an alternate engine.

---

# Phase 3 — Start the frozen model

After setup succeeds, start Strata using the launcher/config generated by that pinned Strata version for this model.

A temporary background mechanism (`nohup`, `tmux`, shell background) is allowed only to keep it alive for curl smoke.

Verify before curl:

- process alive;
- no CUDA OOM;
- no immediate crash loop;
- listener is `127.0.0.1:8081`, not wildcard.

Capture the actual PID, launcher command/config path, and log path.

---

# Phase 4 — curl smoke only

Use port `8081`.

## A. Health

```bash
curl -sS -w '\nHTTP_STATUS=%{http_code}\n' \
  http://127.0.0.1:8081/health
```

## B. Models

```bash
curl -sS -w '\nHTTP_STATUS=%{http_code}\n' \
  http://127.0.0.1:8081/v1/models
```

Read the actual model identifier from the response if one is provided.

## C. One OpenAI-compatible Chinese chat completion

Use the actual model identifier accepted by the service. Semantically equivalent request:

```bash
curl -sS -w '\nHTTP_STATUS=%{http_code}\n' \
  http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<actual-model-id>",
    "messages": [
      {"role": "user", "content": "请用三句话说明大型化工工程投标技术文件通常需要关注哪些核心内容。"}
    ],
    "max_tokens": 256,
    "temperature": 0
  }'
```

Do not repeat requests for benchmarking. One successful chat completion is enough.

---

# Phase 5 — Runtime evidence

Immediately after successful chat smoke capture:

```bash
nvidia-smi
free -h
ps -ef | grep -i '[s]trata' || true
ss -lntp | grep ':8081' || true
```

Record:

- GPU VRAM used and utilization snapshot;
- system RAM used/available;
- Strata PID;
- exact local listener;
- model/quant/context configuration;
- model/data directory size;
- log/config/launcher paths;
- health/models/chat HTTP status;
- returned model identifier;
- token usage/elapsed time only if trivially exposed by the single smoke request.

This is evidence, not a performance benchmark.

Then STOP the investigation. Do not install the next layer.

---

# Gate

Set:

`bootstrap_gate: LLM_BOOTSTRAP_PASS`

only if all are true:

1. CUDA Toolkit 13.0 is available and driver remains healthy;
2. pinned Strata source build succeeds;
3. Qwen3.8-Flash-Next IQ3_XXS assets/config are installed;
4. context = 128K and vision = off;
5. Strata listens only on `127.0.0.1:8081`;
6. health curl returns success;
7. `/v1/models` curl returns success;
8. one `/v1/chat/completions` returns a valid Chinese completion;
9. runtime RAM/VRAM/PID/listener evidence is captured;
10. no RealPDE scientific assets were accessed;
11. no Agent/RAG/persistence stack was installed;
12. no secrets or large artifacts enter Git.

Otherwise use `PARTIAL` or `FAILED` with the exact blocker.

---

# Evidence / Git delivery

Update the existing review folder rather than creating a parallel history unless the date has materially changed:

```text
docs/private_llm_deployment/reviews/strata_qwen38_3090_curl_bootstrap_20261003/
```

Update at least:

- `README.md`
- `preflight.md`
- `runtime.json`
- `curl_smoke.md`

Also update `docs/private_llm_deployment/README.md` with stable deployment facts only after they are actually established.

Evidence must explicitly state:

- CUDA toolkit version;
- NVIDIA driver version after installation;
- Strata SHA/version;
- exact setup/start commands;
- local endpoint `127.0.0.1:8081`;
- model/quant/context;
- curl results;
- runtime snapshot;
- `Agent framework NOT installed`;
- `RAG NOT deployed`;
- `RealPDE scientific assets NOT accessed`;
- `public ingress NOT enabled`;
- `calibration NOT run`;
- `persistent service NOT configured`.

Before final commit:

```bash
git status --short
git fetch origin
git pull --rebase origin main
```

Stage only deployment-direction evidence. Do not use blind `git add .`.

Suggested result commit:

```text
ops: complete Strata Qwen curl bootstrap
```

Then:

```bash
git push origin main
git rev-parse HEAD
git rev-parse origin/main
git status --short
```

Result evidence is not formally delivered until it is on remote `main`.

---

# Final response

Return only a compact handoff:

```text
PRIVATE_LLM_BOOTSTRAP
status: REVIEW_REQUIRED
bootstrap_gate: LLM_BOOTSTRAP_PASS | PARTIAL | FAILED | INVALID
execution_commit: <sha>
evidence_commit: <sha>
cuda_toolkit: <version>
nvidia_driver: <version>
strata_commit/version: 99f3dbd0b21d1401b3769e0c0d963913607f380b / v0.1.38
model: Qwen3.8-Flash-Next IQ3_XXS
context: 128K
endpoint: 127.0.0.1:8081
health_curl: PASS | FAIL
models_curl: PASS | FAIL
chat_curl: PASS | FAIL
local_only_binding: PASS | FAIL
gpu_vram_snapshot: <value>
system_ram_snapshot: <value>
agent_framework_installed: NO
rag_deployed: NO
realpde_assets_accessed: NO
public_ingress_enabled: NO
evidence: docs/private_llm_deployment/reviews/strata_qwen38_3090_curl_bootstrap_20261003/README.md
blocking_issue: <none or concise blocker>
NEXT_ACTION: REVIEW_REQUIRED
```

Do not proceed to Agent, RAG, calibration, tuning, or persistence after this handoff.
