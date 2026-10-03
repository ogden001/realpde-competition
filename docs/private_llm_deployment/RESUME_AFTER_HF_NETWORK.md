# RESUME_AFTER_HF_NETWORK — Strata model download continuation

Status: `IMPLEMENT_AND_EXECUTE_AUTHORIZED / REVIEW_REQUIRED`

This is a narrow continuation of the existing Strata bootstrap. CUDA Toolkit 13.0 and the pinned Strata v0.1.38 engine build are already verified successful. Do not rebuild or redesign the stack unless the existing engine is proven unusable.

## Frozen state

- Strata repo: `~/ai-stack/strata`
- Strata SHA: `99f3dbd0b21d1401b3769e0c0d963913607f380b`
- Strata version: `v0.1.38`
- engine build: PASS, 129/129 targets
- engine path: `~/ai-stack/strata/engine/strata`
- CUDA Toolkit: 13.0.88
- NVIDIA driver: 595.84, healthy and unchanged
- GPU: RTX 3090 24 GB
- model: `Qwen3.8-Flash-Next / IQ3_XXS`
- context: `131072`
- vision: OFF
- KV: INT8
- endpoint: `127.0.0.1:8081`
- model/data root: `~/ai-stack/models`

Previous failure:

`cannot reach huggingface.co (<urlopen error [Errno 101] Network is unreachable>)`

No model shard was downloaded.

## Important upstream fact

Pinned Strata v0.1.38 natively supports the `HF_ENDPOINT` environment variable. Its own documentation and setup code state that a Hugging Face mirror may be used this way, while retaining the same pinned revisions and SHA-256 checks. The MTP downloader honors the same environment variable.

Therefore, using `HF_ENDPOINT=https://hf-mirror.com` is an authorized bounded network adaptation if direct `huggingface.co` access is unavailable and the mirror is reachable.

---

# HARD CONSTRAINTS

1. Do not change the frozen model, quantization, context, vision, KV, Strata SHA, or local endpoint.
2. Do not install an alternate inference engine.
3. Do not patch Strata downloader/model/kernel/core code.
4. Do not bypass Strata's pinned revision or SHA-256 validation.
5. Do not manually download arbitrary GGUF files from unverified third-party URLs.
6. Do not use `--gguf-dir` with files whose provenance/checks are not established by the official Strata flow.
7. Do not disable TLS verification or certificate checking.
8. Do not change DNS, firewall, routing, `/etc/hosts`, system proxy configuration, or system network policy.
9. Do not install a VPN/tunnel/proxy service.
10. Do not expose Strata publicly; endpoint remains `127.0.0.1:8081` only.
11. Do not install Agent/RAG/Web UI/systemd/Docker/Nginx or perform calibration/tuning.
12. Do not access or modify RealPDE scientific assets.
13. Do not commit tokens, proxy credentials, cookies, model binaries, caches, or large logs.
14. Final state remains `REVIEW_REQUIRED`.

---

# Phase 1 — Minimal network diagnosis

Do not immediately rerun the 75+ GB download. First establish which endpoint is reachable.

Run lightweight checks only:

```bash
getent ahosts huggingface.co | head -20 || true
curl -4 -I -L --connect-timeout 10 --max-time 20 https://huggingface.co/ 2>&1 | head -40 || true
curl -6 -I -L --connect-timeout 10 --max-time 20 https://huggingface.co/ 2>&1 | head -40 || true
curl -4 -I -L --connect-timeout 10 --max-time 20 https://hf-mirror.com/ 2>&1 | head -40 || true
```

Also inspect whether the current shell already has proxy variables, but redact values in evidence:

```bash
env | grep -Ei '^(http|https|all|no)_proxy=' | sed 's/=.*/=<redacted>/' || true
```

Do not print proxy credentials or full proxy URLs into Git evidence.

Classification:

### A. Direct Hugging Face works over IPv4

If `curl -4` to `huggingface.co` succeeds with a normal HTTP response, first retry the unchanged official setup once without a mirror. Do not make system network changes.

### B. Direct Hugging Face unavailable, hf-mirror reachable

Authorized path:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Then rerun the exact existing Strata setup. This is the preferred continuation when direct HF is unavailable.

### C. Neither endpoint reachable

STOP and return `REVIEW_REQUIRED` with network evidence. Do not invent another mirror, install a proxy/VPN, or change host networking.

---

# Phase 2 — Resume official Strata model setup

Keep CUDA environment available for this shell if needed:

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
```

If Case B above applies:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Then:

```bash
cd "$HOME/ai-stack/strata"
./setup.sh --yes --family qwen --model IQ3_XXS --context 131072 \
  --vision no --kv int8 --port 8081 --host 127.0.0.1 \
  --data-dir "$HOME/ai-stack/models" --no-start
```

Requirements:

- preserve pinned Strata SHA/version;
- use the official Strata downloader only;
- allow resumable official downloads;
- allow Strata's own revision/hash verification to complete;
- do not download another quant/model as a fallback.

Record whether `HF_ENDPOINT` was used and which endpoint actually served the model assets.

If setup fails after a partial download, preserve official partial/resumable files and log evidence. Do not delete and restart blindly.

---

# Phase 3 — Start and curl smoke

Only after setup/model preparation fully succeeds, start using the generated official launcher/config.

Verify listener is exactly local-only on port 8081.

Then run the existing curl smoke from `NEXT_ACTION.md`:

1. `/health`
2. `/v1/models`
3. one Chinese `/v1/chat/completions`

Use the actual model identifier returned by the service.

Capture runtime GPU VRAM, RAM, PID, listener and relevant model/config paths.

Do not run a benchmark sweep.

---

# Evidence / Git delivery

Update the existing evidence folder:

`docs/private_llm_deployment/reviews/strata_qwen38_3090_curl_bootstrap_20261003/`

Update at least:

- `README.md`
- `preflight.md`
- `runtime.json`
- `curl_smoke.md`

Evidence must state:

- direct `huggingface.co` reachability result;
- `hf-mirror.com` reachability result;
- whether `HF_ENDPOINT` was used;
- that pinned revisions/SHA validation remained enabled;
- model download/preparation outcome;
- Strata start/curl outcome;
- no alternate model/engine/downloader was used;
- Agent/RAG/public ingress/persistence/calibration remain NOT enabled.

Commit and push evidence to `origin/main` before returning.

---

# Final handoff

Return:

```text
PRIVATE_LLM_BOOTSTRAP
status: REVIEW_REQUIRED
bootstrap_gate: LLM_BOOTSTRAP_PASS | PARTIAL | FAILED | INVALID
execution_commit: <sha>
evidence_commit: <sha>
cuda_toolkit: 13.0.88
nvidia_driver: 595.84
strata_commit/version: 99f3dbd0b21d1401b3769e0c0d963913607f380b / v0.1.38
model: Qwen3.8-Flash-Next IQ3_XXS
context: 128K
hf_direct: PASS | FAIL
hf_mirror: PASS | FAIL | NOT_USED
hf_endpoint_used: <direct | https://hf-mirror.com | none>
model_download: PASS | FAIL
model_prepare: PASS | FAIL
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

Do not proceed beyond the local LLM curl bootstrap.
