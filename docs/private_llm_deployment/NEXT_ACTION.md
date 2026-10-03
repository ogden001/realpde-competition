# NEXT_ACTION — Private LLM Node Bootstrap on RTX 3090

Status: `IMPLEMENT_AND_EXECUTE_AUTHORIZED / REVIEW_REQUIRED`

This task is an **independent infrastructure/deployment direction**. The RealPDE repository is used only as the shared ChatGPT/Sol ↔ Codex/Luna coordination and evidence channel. Do **not** modify or access RealPDE scientific assets unless explicitly required to read the collaboration protocol.

The user will provide the exact Git commit containing this task as `REQUIRED_COMMIT`. Before execution, verify that commit is an ancestor of `HEAD`.

## Goal

Turn the existing single-GPU server into a stable local AI node with the following frozen first-stage stack:

```text
DeepSeek Harness (Agent)
        ↓ OpenAI-compatible API
Strata serving
        ↓
Qwen3.8-Flash-Next / IQ3_XXS / 128K
        ↓
RTX 3090 24 GB + 64 GB system RAM
```

Target machine expected hardware:

- GPU: NVIDIA RTX 3090 24 GB
- RAM: 64 GB
- CPU: Intel i7-13700KF
- Disk: 1 TB
- OS: Linux, preferably Ubuntu

This task is **deployment/bootstrap only**. Do not turn it into RAG, benchmark research, model comparison, multi-user serving, or product engineering.

## Roles

### ChatGPT / Sol owns

- frozen deployment architecture;
- model family / quantization / context choice;
- hard/soft constraints;
- review of execution evidence;
- decision on any follow-up optimization, RAG, alternative agent, or production hardening.

### Codex / Luna owns

- host/environment inspection;
- bounded installation and environment adaptation;
- following upstream installation docs;
- service wiring and local API configuration;
- smoke tests;
- reproducible command/config capture;
- lightweight evidence commit + push to `origin/main`.

If a required adaptation would change the frozen architecture or a HARD CONSTRAINT, STOP and report. Do not choose a new architecture yourself.

---

# HARD CONSTRAINTS

These are absolute.

## H1. Do not touch RealPDE scientific assets

The RealPDE repository is only the coordination/evidence repo for this task.

Do not:

- modify RealPDE model/training/evaluation code;
- read/copy/delete/move any RealPDE H5 dataset or checkpoint;
- start/stop any RealPDE training/evaluation job;
- reuse RealPDE run directories for this deployment;
- add model binaries, large logs, credentials, or generated deployment state into Git.

Only the protocol/task/evidence files under this new deployment direction may be added to this repo, plus small generic deployment scripts/config templates if needed.

## H2. Frozen first-stage model

Deploy exactly:

- family: `Qwen3.8-Flash-Next`
- Strata quantization: `IQ3_XXS`
- context target: `131072` tokens (128K)
- vision: OFF
- MTP/speculative decoding: use Strata supported/default path
- KV: keep Strata's normal/default 8-bit KV path; do not opt into experimental q4/k8v4 variants

Do not silently switch to:

- IQ2_XS;
- IQ3_S;
- Coder;
- Swift;
- another Qwen model;
- another inference engine.

If the frozen configuration cannot start, preserve evidence and STOP. A fallback requires Sol/user review.

## H3. Strata is the model server

Use the official upstream repository:

`https://github.com/Niko1221/Strata`

Prefer the current supported upstream installation flow rather than recreating engine flags by hand.

Pin and record the exact Strata Git commit actually used.

Do not patch Strata core/kernel/model code in this task. If upstream fails due to a genuine compatibility bug, capture the failure and STOP rather than inventing a local fork.

## H4. Agent is DeepSeek Harness first

Use the official DeepSeek Harness project/package and its documented installation/configuration flow.

Primary goal:

`DeepSeek Harness -> local Strata OpenAI-compatible endpoint`

Do not silently replace Harness with AgentScope, Qwen-Agent, OpenHands, LangGraph, or another agent framework.

If Harness cannot interoperate with the local Strata endpoint, leave the verified Strata service intact, capture the compatibility evidence, and STOP at `REVIEW_REQUIRED`.

## H5. No public exposure

For this stage:

- Strata must listen only on `127.0.0.1`;
- DeepSeek Harness must listen only on `127.0.0.1` where supported;
- do not bind either service to `0.0.0.0`;
- do not open firewall ports;
- do not configure public DNS, TLS, Nginx, Cloudflare, tunnels, or external ingress.

Remote user access, if needed, is via SSH port forwarding outside this task.

## H6. Agent process must not run as root

DeepSeek Harness / agent tool execution must not run with UID 0.

Preferred:

- use an existing non-root deployment user; or
- if Codex is currently root and host policy permits, create a dedicated unprivileged `aiagent` user.

The agent workspace must be a dedicated directory under the AI stack root and must not be the RealPDE repo, `/root`, `/`, or another project directory.

Do not grant the agent passwordless sudo.

If the environment cannot satisfy a non-root agent process safely, STOP.

## H7. Do not kill unrelated processes

Before installation/startup inspect GPU processes with `nvidia-smi`.

If an unknown or unrelated process is using material GPU memory/compute:

- do not kill it;
- do not reset the GPU;
- do not steal the port/process;
- report the conflict and STOP before starting Strata.

## H8. No destructive cleanup

Do not use destructive broad commands such as:

- `rm -rf` on unknown/existing project trees;
- `git reset --hard`;
- `git clean -fdx`;
- deleting existing model caches to make room;
- rewriting unrelated service configs.

Unknown existing installations/configs must be preserved and reported.

## H9. Capacity gate before large model download

Before downloading model assets verify:

- RTX 3090 is visible and reports about 24 GB VRAM;
- visible system RAM is about 60 GiB or more;
- target filesystem has at least `180 GiB` free;
- NVIDIA driver/runtime are healthy;
- there is no material swap thrashing or disk-full condition.

If any capacity gate fails, STOP before the large download.

## H10. No secrets in Git

Never commit:

- passwords;
- SSH keys;
- API secrets;
- tokens;
- private hostnames/IPs if they are sensitive;
- full environment dumps containing credentials.

Use redacted templates and record only non-secret paths/version facts.

## H11. Scope exclusions

Do NOT add in this task:

- RAG;
- vector DB;
- embedding service;
- reranker;
- Nginx;
- HTTPS;
- Kubernetes;
- Dockerization merely for packaging convenience;
- multi-model router;
- cloud API fallback;
- multi-user auth;
- performance tuning sweep;
- production load test.

## H12. Final state is always REVIEW_REQUIRED

Codex does not declare the architecture production-ready and does not start a second-stage optimization on its own.

End at:

`REVIEW_REQUIRED`

Then wait for user + ChatGPT/Sol review.

---

# SOFT CONSTRAINTS

These may be adapted to the real host as long as the architecture and HARD CONSTRAINTS remain unchanged.

## S1. Installation root

Preferred root:

`/opt/ai-stack`

Preferred layout:

```text
/opt/ai-stack/
├── strata/
├── agents/
│   └── deepseek-harness/
├── models/
├── workspace/
├── config/
├── logs/
└── scripts/
```

If `/opt` is not writable or host policy prefers user-local software, use an equivalent root such as:

`$HOME/ai-stack`

Record the actual root. Do not force `/opt` by weakening permissions.

## S2. Native Linux preferred

Prefer native Linux installation. Do not introduce Docker unless upstream absolutely requires it, and if so STOP for review instead of changing architecture automatically.

## S3. Service manager

Preferred persistence:

- system `systemd` service when appropriate; or
- `systemd --user` for user-owned services.

If systemd is unavailable, complete a foreground/local smoke test, provide explicit start/stop scripts, record that persistence is not established, and return `REVIEW_REQUIRED`. Do not install a new process manager just for this task.

## S4. Follow upstream commands, do not guess stale CLI flags

The repositories may have changed since this task was written.

Codex may adapt exact CLI syntax to the current official upstream docs, but the resulting semantics must remain:

- Qwen3.8-Flash-Next;
- IQ3_XXS;
- 128K;
- vision off;
- local-only API.

Record the exact commands actually executed.

## S5. Calibration

After the frozen model can start and pass API smoke, run Strata's official NVIDIA calibration flow if supported by the pinned version and it does not change the frozen model/quantization/context semantics.

Calibration failure is non-fatal for the base deployment. Preserve the failure and keep the known-working configuration rather than experimenting broadly.

## S6. Minimal package installation

Install only dependencies required by official Strata/Harness docs. Prefer host package manager and project-managed virtual/node environments.

Do not perform broad OS upgrades or driver upgrades unless the existing driver is incompatible. Driver replacement is out of scope and requires review.

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

If unknown repo changes exist, do not stash/reset/restore/clean them. STOP and report.

Then capture host facts, at minimum:

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
- current listeners around proposed ports `8080` and `3080`;
- existing Strata/Harness/AI-stack directories;
- whether systemd/systemd-user is available;
- available Node/Python/Git versions as required by current upstream docs.

Do not expose secrets in evidence.

Create a lightweight preflight evidence record before the large model download.

---

# Execution

## Phase 1 — Prepare isolated AI stack

1. Choose the actual AI stack root using S1.
2. Create separate locations for Strata, Harness, workspace, logs and config.
3. Ensure the Harness runtime identity is non-root.
4. Keep the RealPDE repository outside the agent workspace.
5. Record ownership/permissions relevant to the services.

Do not weaken broad filesystem permissions with `chmod -R 777`.

## Phase 2 — Install and pin Strata

1. Clone official Strata into the chosen stack root.
2. Record:
   - upstream URL;
   - branch/tag if any;
   - exact Git SHA;
   - install command;
   - engine/version information emitted by the project.
3. Follow the official setup flow for the frozen model configuration.
4. Keep model assets outside the RealPDE Git repository.
5. Do not download additional model variants.

If upstream setup asks interactive questions, answer them consistently with the frozen semantics and record the final choices.

## Phase 3 — Start Strata locally

Start Strata with the exact launcher generated/recommended by the pinned upstream version.

Required properties:

- bind only to `127.0.0.1`;
- model = Qwen3.8-Flash-Next IQ3_XXS;
- context = 128K;
- vision disabled;
- no experimental quantization/context fallback.

Verify the process is stable for at least several minutes and is not crash-looping/OOMing.

## Phase 4 — Strata API smoke

At minimum verify the supported equivalents of:

- health endpoint;
- model listing endpoint;
- one OpenAI-compatible chat request;
- one Chinese response.

Capture:

- HTTP status;
- model identity returned by the service;
- prompt/completion usage if exposed;
- process exit/state;
- GPU VRAM snapshot;
- system RAM snapshot.

This is a smoke test, not a benchmark. Do not start a performance sweep.

## Phase 5 — Optional official calibration

If supported by the pinned Strata version, run the project's official calibration routine once.

Record:

- command;
- selected settings;
- before/after service start status;
- any warning/error.

After calibration, rerun the basic API smoke. If calibration breaks the working service, revert only the calibration-generated change using the project's supported mechanism or preserve the pre-calibration working config and report. Do not begin manual kernel/tuning experiments.

## Phase 6 — Make Strata persistent

Prefer a systemd service or systemd-user service based on the actual install/runtime user.

Use the exact known-working Strata launcher rather than reconstructing low-level engine flags.

Required service behavior:

- local-only binding;
- predictable working directory;
- logs retrievable with journal or a documented file path;
- restart on genuine failure where appropriate;
- no secrets embedded in Git-tracked unit templates.

Verify:

```text
start -> healthy -> stop -> start -> healthy
```

Do not reboot the server solely for this task.

## Phase 7 — Install and pin DeepSeek Harness

Use the official DeepSeek Harness source/package and current official docs.

Record:

- upstream project/package;
- exact source SHA and/or package version;
- Node/runtime version;
- installation command.

Configure its model provider to target the local Strata OpenAI-compatible API, semantically equivalent to:

```text
provider: strata-local
base_url: http://127.0.0.1:8080/v1
model: local Strata model
protocol: OpenAI Chat Completions (or the exact currently supported compatible mode)
```

Use a non-secret local placeholder key only if the client requires a syntactic API key.

Do not expose Harness publicly.

## Phase 8 — Agent isolation and smoke

Run Harness as a non-root identity with a dedicated workspace, preferred:

`<AI_STACK_ROOT>/workspace`

Perform only minimal smoke checks:

1. Harness starts without fatal error.
2. Harness can reach the local Strata provider.
3. A basic chat request returns through the full chain.
4. If the current Harness supports tool/file operations in this configuration, ask it to create a harmless text file inside the dedicated workspace and read it back.

The tool smoke file must stay inside the dedicated workspace.

If plain chat works but tool calling fails due to model/provider compatibility:

- do not patch Harness or Strata core;
- capture the request/error/log evidence;
- classify `HARNESS_CHAT_SMOKE=PASS`, `HARNESS_TOOL_SMOKE=FAIL`;
- return `REVIEW_REQUIRED`.

## Phase 9 — Persistence for Harness

If Harness has a stable documented daemon/server mode, make it persistent using systemd/systemd-user while preserving:

- non-root execution;
- local-only binding;
- dedicated workspace;
- local Strata provider.

If current Harness is clearly developer-preview/interactive and persistent service wrapping is not reliable, do not invent a fragile daemon. Provide a reproducible start script and record `HARNESS_PERSISTENCE=NOT_ESTABLISHED`.

---

# Verification gates

The deployment may be summarized as `BOOTSTRAP_PASS` only when all of the following are true:

1. hardware/capacity preflight passed;
2. Strata pinned version recorded;
3. Qwen3.8-Flash-Next IQ3_XXS starts at 128K;
4. Strata stays local-only;
5. Strata API smoke passes;
6. Strata survives stop/start;
7. Harness exact version recorded;
8. Harness runs non-root;
9. Harness uses the dedicated workspace;
10. Harness reaches the local Strata endpoint;
11. full-chain basic chat smoke passes;
12. no RealPDE scientific asset was accessed or modified;
13. no secret was committed.

`HARNESS_TOOL_SMOKE` may be `PASS` or `FAIL`; a tool-call compatibility failure does not invalidate a working Strata deployment, but it must be explicitly surfaced for Sol review.

Any hard-constraint violation makes the execution `INVALID` until reviewed.

---

# Evidence and deliverables

Create/update this direction only:

```text
docs/private_llm_deployment/
├── README.md
├── NEXT_ACTION.md
└── reviews/
    └── strata_qwen38_3090_bootstrap_20261003/
        ├── README.md
        ├── preflight.md
        ├── runtime.json
        ├── strata_smoke.md
        ├── harness_smoke.md
        └── service_inventory.md
```

Names may vary slightly if a current-date suffix is needed, but keep everything under this direction.

### `docs/private_llm_deployment/README.md`

Long-term direction memory. Record only stable facts:

- purpose of the node;
- actual hardware/OS facts;
- actual AI stack root;
- Strata upstream SHA/version;
- model/quant/context;
- Harness SHA/version;
- local ports;
- service names;
- start/stop/status/log commands;
- known limitations;
- review status.

### Review evidence

At minimum record:

- execution Git commit from the RealPDE coordination repo;
- host hardware facts;
- disk/RAM/VRAM preflight;
- existing GPU process check;
- installed versions/SHAs;
- exact non-secret commands used;
- actual install paths;
- service unit names and status;
- local listen addresses/ports;
- Strata API smoke results;
- Harness chat/tool smoke results;
- runtime RAM/VRAM snapshots;
- any bounded environment adaptation;
- any failure/warning;
- explicit scope statement:
  - `RealPDE scientific assets NOT accessed`
  - `RAG NOT deployed`
  - `vector DB NOT deployed`
  - `public ingress NOT enabled`
  - `cloud model fallback NOT configured`

Do not commit full downloaded models or large raw logs. Reference their local paths and sizes only.

Generic reproducibility helpers may be added under a small deployment-specific path if they are genuinely needed, are secret-free, and do not modify RealPDE scientific code.

---

# Git delivery

Before final commit:

```bash
git status --short
git fetch origin
git pull --rebase origin main
```

If rebase creates a substantive conflict, STOP and report rather than guessing.

Stage only this deployment direction and any explicitly related small helper/template files. Do not use blind `git add .`.

Commit with a scoped message, for example:

```text
ops: record local Qwen Strata bootstrap
```

Then:

```bash
git push origin main
git rev-parse HEAD
git rev-parse origin/main
git status --short
```

Requirements:

- result/evidence commit is present on remote `main`;
- `HEAD == origin/main` after delivery;
- no unknown local changes introduced by this task.

---

# Final response format

Return a compact handoff only after the GitHub delivery is complete:

```text
PRIVATE_LLM_BOOTSTRAP
status: REVIEW_REQUIRED
bootstrap_gate: BOOTSTRAP_PASS | PARTIAL | FAILED | INVALID
execution_commit: <sha>
strata_commit/version: <value>
model: Qwen3.8-Flash-Next IQ3_XXS
context: 128K
strata_api_smoke: PASS | FAIL
strata_restart_smoke: PASS | FAIL
harness_version: <value>
harness_chat_smoke: PASS | FAIL
harness_tool_smoke: PASS | FAIL | NOT_SUPPORTED
agent_non_root: PASS | FAIL
local_only_binding: PASS | FAIL
realpde_assets_accessed: NO
public_ingress_enabled: NO
evidence: docs/private_llm_deployment/reviews/<run>/README.md
blocking_issue: <none or concise issue>
NEXT_ACTION: REVIEW_REQUIRED
```

Do not start RAG, switch models, expose the service, tune the model, or begin another deployment after this handoff.
