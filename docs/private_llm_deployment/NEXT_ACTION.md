# NEXT_ACTION — Strata + Qwen3.8-Flash-Next Bootstrap on RTX 3090

Status: `IMPLEMENT_AND_EXECUTE_AUTHORIZED / REVIEW_REQUIRED`

This is the **first deployment step only**.

The RealPDE repository is used only as the ChatGPT/Sol ↔ Codex/Luna coordination and evidence channel. Do not modify or access RealPDE scientific assets.

The user will provide the exact commit containing this task as `REQUIRED_COMMIT`. Before execution, verify that commit is an ancestor of `HEAD`.

## Goal

Deploy and start exactly this local LLM stack on the existing RTX 3090 server:

```text
curl
  ↓ OpenAI-compatible HTTP API
Strata
  ↓
Qwen3.8-Flash-Next / IQ3_XXS / 128K
  ↓
RTX 3090 24 GB + 64 GB RAM
```

Expected host:

- GPU: NVIDIA RTX 3090 24 GB
- RAM: 64 GB
- CPU: Intel i7-13700KF
- Disk: 1 TB
- OS: Linux

Success for this task means:

1. Strata is installed from the official upstream repository;
2. Qwen3.8-Flash-Next IQ3_XXS is installed/configured for 128K context;
3. the service starts successfully on localhost only;
4. `curl` can query health/model endpoints and receive one successful OpenAI-compatible chat completion;
5. lightweight evidence is committed and pushed to `origin/main`;
6. final state is `REVIEW_REQUIRED`.

**Do not install any Agent framework in this task.**

---

# Roles

## ChatGPT / Sol owns

- deployment architecture;
- model / quantization / context choice;
- hard/soft constraints;
- review of execution evidence;
- decision on later Agent, RAG, calibration, service hardening, or production work.

## Codex / Luna owns

- host/environment inspection;
- bounded installation and environment adaptation;
- following current official Strata documentation;
- starting the frozen model server;
- curl smoke tests;
- lightweight evidence collection;
- commit + push to `origin/main`.

If a required adaptation would change a HARD CONSTRAINT, STOP and report. Do not choose a new architecture or model yourself.

---

# HARD CONSTRAINTS

These are absolute.

## H1. Do not touch RealPDE scientific assets

Do not:

- modify RealPDE model/training/evaluation code;
- read/copy/delete/move any RealPDE H5 dataset or checkpoint;
- start/stop any RealPDE training/evaluation job;
- reuse RealPDE run directories for this deployment;
- add model binaries, large logs, credentials, or generated model state into Git.

Only files under `docs/private_llm_deployment/` may be added/updated for this task, plus a tiny secret-free helper script only if strictly needed.

## H2. Frozen model configuration

Deploy exactly:

- model family: `Qwen3.8-Flash-Next`
- Strata quantization: `IQ3_XXS`
- context target: `131072` tokens (128K)
- vision: OFF
- MTP/speculative decoding: use the normal supported Strata path
- KV: keep Strata's normal/default 8-bit path; do not opt into experimental q4/k8v4 variants

Do not silently switch to:

- IQ2_XS;
- IQ3_S;
- Coder;
- Swift;
- another Qwen model;
- another inference engine.

If this exact configuration cannot start, preserve evidence and STOP at `REVIEW_REQUIRED`.

## H3. Strata is the only inference server

Use the official repository:

`https://github.com/Niko1221/Strata`

Requirements:

- clone/use official upstream;
- record exact Strata Git SHA actually used;
- follow the current official setup flow;
- do not patch Strata core/kernel/model code;
- do not create a private fork to work around an upstream problem.

If official Strata cannot run this frozen configuration on the host, capture the failure and STOP.

## H4. No Agent installation

Do **not** install or configure:

- DeepSeek Harness;
- AgentScope;
- Qwen-Agent;
- OpenHands;
- LangGraph;
- Claude Code integrations;
- Codex integrations;
- any other Agent framework/runtime.

This task ends after direct HTTP/curl model API validation.

## H5. No RAG/application stack

Do not install or configure:

- RAG;
- vector database;
- embedding model/service;
- reranker;
- document parsing pipeline;
- Web UI;
- application backend;
- model router;
- cloud model fallback.

## H6. Local-only network binding

For this stage:

- Strata must bind to `127.0.0.1` only;
- do not bind to `0.0.0.0`;
- do not open firewall ports;
- do not configure public DNS, TLS, Nginx, Cloudflare, tunnels, or ingress.

All smoke tests are performed locally with `curl`.

## H7. Do not kill unrelated processes

Before installation/startup inspect `nvidia-smi` and relevant process/port state.

If an unknown/unrelated process is using material GPU memory or compute:

- do not kill it;
- do not reset the GPU;
- do not steal its port;
- STOP and report before starting Strata.

## H8. No destructive cleanup

Do not use destructive broad commands such as:

- `rm -rf` on unknown/existing project trees;
- `git reset --hard`;
- `git clean -fdx`;
- deleting existing model caches to make room;
- overwriting unrelated services/configuration.

Unknown existing installations/configs must be preserved and reported.

## H9. Capacity gate before large download

Before downloading model assets verify:

- RTX 3090 is visible and reports about 24 GB VRAM;
- visible system RAM is about 60 GiB or more;
- target filesystem has at least `180 GiB` free;
- NVIDIA driver/runtime are healthy;
- no material swap thrashing or disk-full condition exists.

If any gate fails, STOP before the large download.

## H10. No secrets in Git

Never commit:

- passwords;
- SSH keys;
- API secrets;
- tokens;
- sensitive hostnames/IPs;
- unredacted environment dumps containing credentials.

## H11. No optimization sweep

Do not do:

- model comparison;
- quantization comparison;
- context sweep;
- performance benchmark matrix;
- kernel tuning;
- expert-cache tuning;
- speculative decoding tuning;
- production load test.

A single minimal smoke measurement/resource snapshot is allowed only as evidence that deployment works.

## H12. No persistence/hardening work in this task

Do not spend time on:

- systemd service creation;
- Docker/Kubernetes;
- watchdogs;
- production logging stack;
- monitoring stack;
- auto-restart policy;
- boot persistence.

The objective is first successful local deployment and curl validation. Persistence comes later after review.

## H13. No calibration in this task

Do not run Strata calibration yet. First establish an untouched upstream baseline that works.

## H14. Final state is always REVIEW_REQUIRED

Do not proceed to Agent, RAG, optimization, or hardening after curl smoke.

End at:

`REVIEW_REQUIRED`

and wait for user + ChatGPT/Sol review.

---

# SOFT CONSTRAINTS

These may be adapted to the real host without changing the frozen deployment semantics.

## S1. Installation root

Preferred:

`/opt/ai-stack`

Suggested minimal layout:

```text
/opt/ai-stack/
├── strata/
├── models/
└── logs/
```

If `/opt` is unsuitable, use an equivalent location such as `$HOME/ai-stack` and record the actual path.

Do not weaken permissions merely to force `/opt`.

## S2. Native Linux preferred

Prefer native Linux. Do not introduce Docker simply for convenience.

If current official Strata unexpectedly requires a materially different deployment mechanism, STOP for review rather than changing architecture automatically.

## S3. Follow current upstream CLI

The Strata repository can change. Codex may adapt exact CLI syntax to current official docs, but final semantics must remain:

- Qwen3.8-Flash-Next;
- IQ3_XXS;
- 128K;
- vision off;
- local-only API.

Record exact commands actually executed.

## S4. Minimal dependencies

Install only dependencies required by official Strata documentation.

Do not perform broad OS or NVIDIA driver upgrades. If the current driver is incompatible and replacement is required, STOP and report rather than modifying the GPU software stack automatically.

## S5. Background process for smoke is allowed

After successful installation, Codex may launch Strata with the project's supported launcher under `nohup`, a shell background process, `tmux`, or equivalent solely to keep the model alive long enough for curl smoke and evidence collection.

Record PID/command/log path when applicable.

Do not convert this into a permanent service in this task.

---

# Preflight

Run from the RealPDE coordination repo before heavy work:

```bash
git status --short
git fetch origin
git pull --rebase origin main
git rev-parse HEAD
git rev-parse origin/main
git merge-base --is-ancestor "$REQUIRED_COMMIT" HEAD
git push --dry-run origin HEAD:main
```

Requirements:

- no unknown local modifications;
- `HEAD == origin/main`;
- `REQUIRED_COMMIT` is an ancestor of `HEAD`;
- dry-run push succeeds.

If unknown repo changes exist, do not stash/reset/restore/clean. STOP and report.

Then capture host facts:

```bash
uname -a
cat /etc/os-release
id
lscpu
free -h
df -h
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version,pstate --format=csv
```

Also inspect:

- GPU process list;
- listener state for port `8080`;
- existing Strata / AI-stack directories;
- Git/Python/tool versions required by current Strata docs.

Do not expose secrets in evidence.

Create lightweight preflight evidence before the large model download.

---

# Execution

## Phase 1 — Prepare isolated Strata installation

1. Choose actual AI stack root using S1.
2. Keep the installation completely outside RealPDE data/run directories.
3. Create only directories needed for Strata/model/logs.
4. Record actual path and disk free space.

## Phase 2 — Clone and pin official Strata

Clone/use:

`https://github.com/Niko1221/Strata`

Record:

- upstream URL;
- current branch/tag if relevant;
- exact Git SHA;
- install/setup command;
- Strata/engine version information if exposed.

Do not modify Strata core source.

## Phase 3 — Install frozen model

Follow the current upstream setup flow to install exactly:

```text
Qwen3.8-Flash-Next
IQ3_XXS
context = 131072
vision = OFF
```

Use default/supported MTP and normal/default KV path.

Do not download alternate Qwen model variants for comparison.

If the setup is interactive, answer consistently with the frozen configuration and record the effective choices.

## Phase 4 — Start Strata locally

Start using the official generated/recommended launcher.

Required:

- localhost only;
- expected model/quantization;
- 128K context;
- vision disabled;
- no fallback model.

Verify:

- process remains alive;
- no CUDA OOM;
- no immediate crash-loop;
- expected listening port is local-only.

## Phase 5 — curl smoke

Use `curl` directly against Strata. Adapt endpoint names only if the current official API differs.

At minimum test:

### A. Health

Equivalent to:

```bash
curl -sS http://127.0.0.1:8080/health
```

### B. Models

Equivalent to:

```bash
curl -sS http://127.0.0.1:8080/v1/models
```

### C. OpenAI-compatible chat completion

Equivalent to:

```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "strata",
    "messages": [
      {"role": "user", "content": "请用三句话说明大型化工工程投标技术文件通常需要关注哪些核心内容。"}
    ],
    "max_tokens": 256,
    "temperature": 0
  }'
```

If the current API reports a different actual model name, use the name returned by `/v1/models` instead of guessing.

Capture:

- HTTP success/failure;
- response body in redacted/lightweight form;
- returned model identifier;
- token usage if exposed;
- elapsed request time if easily available from curl;
- no need to run repeated performance trials.

## Phase 6 — Runtime snapshot

Immediately after successful chat smoke, capture:

```bash
nvidia-smi
free -h
ps -ef | grep -i strata
ss -lntp | grep 8080 || true
```

Record at minimum:

- GPU VRAM used;
- GPU utilization snapshot;
- system RAM used/available;
- Strata PID;
- local listening address;
- model/context configuration;
- log path.

This is deployment evidence, not a benchmark.

Then STOP. Do not install anything else.

---

# Verification Gate

Return `LLM_BOOTSTRAP_PASS` only if all are true:

1. host/capacity preflight passed;
2. exact Strata SHA/version recorded;
3. exact Qwen3.8-Flash-Next IQ3_XXS installed;
4. configured context is 128K;
5. vision is off;
6. Strata listens only on localhost;
7. health curl passes;
8. models curl passes;
9. OpenAI-compatible chat curl returns a valid completion;
10. runtime RAM/VRAM evidence captured;
11. no RealPDE scientific asset was accessed or modified;
12. no Agent/RAG/application stack was installed;
13. no secret was committed.

If the service starts but one API smoke fails, classify `PARTIAL` and preserve exact evidence.

Any HARD CONSTRAINT violation means `INVALID` until Sol review.

---

# Evidence and Deliverables

Maintain only:

```text
docs/private_llm_deployment/
├── README.md
├── NEXT_ACTION.md
└── reviews/
    └── strata_qwen38_3090_curl_bootstrap_20261003/
        ├── README.md
        ├── preflight.md
        ├── runtime.json
        └── curl_smoke.md
```

A slightly different date suffix is acceptable if execution crosses midnight.

## `docs/private_llm_deployment/README.md`

Record stable facts only:

- purpose of this local LLM node;
- actual host hardware/OS;
- actual Strata install root;
- exact Strata SHA/version;
- model / quantization / context;
- local endpoint;
- exact start command/launcher;
- known limitations;
- review status.

## Review evidence must include

- execution commit in the coordination repo;
- hardware/capacity preflight;
- GPU process conflict check;
- exact Strata SHA/version;
- exact install/setup/start commands, secret-free;
- actual install/model paths and sizes;
- health/models/chat curl results;
- runtime RAM/VRAM snapshot;
- bounded environment adaptations;
- warnings/errors;
- explicit scope statement:
  - `RealPDE scientific assets NOT accessed`
  - `Agent framework NOT installed`
  - `RAG NOT deployed`
  - `public ingress NOT enabled`
  - `calibration NOT run`
  - `persistent service NOT configured`

Do not commit model files or large raw logs. Reference paths only.

---

# Git Delivery

Before final commit:

```bash
git status --short
git fetch origin
git pull --rebase origin main
```

If rebase creates a substantive conflict, STOP and report.

Stage only files in this deployment direction. Do not use blind `git add .`.

Suggested commit message:

```text
ops: record Strata Qwen curl bootstrap
```

Then:

```bash
git push origin main
git rev-parse HEAD
git rev-parse origin/main
git status --short
```

Requirements:

- evidence commit is present on remote `main`;
- `HEAD == origin/main` after delivery;
- no unknown local changes introduced by this task.

---

# Final Response Format

After GitHub delivery, return only a compact handoff:

```text
PRIVATE_LLM_BOOTSTRAP
status: REVIEW_REQUIRED
bootstrap_gate: LLM_BOOTSTRAP_PASS | PARTIAL | FAILED | INVALID
execution_commit: <sha>
strata_commit/version: <value>
model: Qwen3.8-Flash-Next IQ3_XXS
context: 128K
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
evidence: docs/private_llm_deployment/reviews/<run>/README.md
blocking_issue: <none or concise issue>
NEXT_ACTION: REVIEW_REQUIRED
```

Do not install an Agent, RAG stack, Web UI, model router, or persistent service after this handoff.
