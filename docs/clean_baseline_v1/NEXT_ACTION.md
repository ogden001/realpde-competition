# NEXT_ACTION — Execute REALPDE_CLEAN_BASELINE_V1

Status: REVIEW_REQUIRED

## Goal

Execute the already-implemented clean baseline unattended for an overnight 8–10 hour run.

The priority order is:

1. preserve scientific validity;
2. prevent avoidable infrastructure/environment failures;
3. preserve evidence if anything fails;
4. never improvise a new scientific experiment.

Do not redesign the model, loss, split, sampler, budgets, or checkpoint-selection rule.

---

# A. HARD CONSTRAINTS — NEVER CHANGE

These define the experiment. Codex is NOT authorized to change them to make the run easier or faster.

## Repository / workflow

- Branch: `main` only.
- Pull latest `main` before execution.
- Working tree must be clean before training.
- Do not edit tracked source/config files during execution.
- Do not create a new experimental branch.
- Status remains `REVIEW_REQUIRED`.

## Data / privacy

- Released real PIV only.
- Canonical split must remain exactly:
  - Train51
  - Seen-Dev12
  - AoA10-Holdout18
- All available AoA=10 trajectories must remain holdout-only.
- Train / Dev / Holdout must remain trajectory-disjoint.
- `7575_0.h5` remains excluded.
- Do not access locked-final/private data.
- Do not access Codabench.
- Do not inspect Holdout18 before Stage1 + Stage2 training is complete.

## Scientific training protocol

Do NOT change any of the following:

- Stage1 architecture.
- Stage1 loss.
- Stage2 residual architecture.
- Stage2 loss weights.
- Stage1 or Stage2 baseline batch size = 8. Disposable runtime profiling may benchmark batch=16, but must never apply it to this baseline.
- train stride = 1.
- eval stride = 20.
- Past20 / Future20.
- sub_sample = 2 / P00.
- seed = 41.
- Stage1 max updates = 8723.
- Stage2 max updates = 38400.
- eval interval = 1000.
- Stage1 lr = 1e-4.
- Stage2 lr = 2e-4.
- Stage2 weight_decay = 1e-5.
- Stage2 hidden = 96.
- Stage2 blocks = 2.
- Stage2 max_delta = 0.04.
- Stage2 alpha = 1.0.
- point-only checkpoint selection rule.
- no SPS / runtime contribution to model selection.
- no AMP/bf16/fp16 conversion.
- no augmentation.
- no end-to-end joint fine-tuning.
- no all81/all82/full-data refit.
- no submission packaging or submission.

If one of these must change for the job to run, STOP and report `REVIEW_REQUIRED`.

---

# B. SOFT CONSTRAINTS — SELF-HEALING IS ALLOWED

These do NOT define the science. Codex SHOULD solve them autonomously when safe.

Allowed examples:

- activate an already-existing Python/conda/venv environment;
- resolve an approved existing data/checkpoint/model path;
- choose an idle approved GPU and set `CUDA_VISIBLE_DEVICES`;
- set `PYTHONUNBUFFERED=1`;
- set `PYTHONHASHSEED=41`;
- use `tmux`, or `nohup + setsid` if tmux is unavailable;
- run the built-in disposable b8/b16 runtime profiler and record its recommendation; the recommendation must not change this baseline;
- choose another local output directory with enough free disk;
- create runtime-only logs/PID/health files outside tracked Git files;
- reduce DataLoader `--workers` from 4 -> 2 -> 0 if and only if worker/process/HDF5 I/O issues occur;
- retry a PRE-TRAINING environment/path check after fixing a non-scientific issue;
- wait briefly for a temporarily busy GPU rather than changing the scientific configuration.

Not allowed under "self-healing":

- changing batch size;
- changing updates;
- changing stride;
- changing model/loss/lr;
- changing split;
- using fewer training samples;
- skipping evaluation;
- changing precision;
- using another checkpoint;
- reading holdout early;
- restarting training from a scientifically different state.

When uncertain whether a recovery changes scientific semantics, treat it as HARD and stop.

---

# C. PRE-FLIGHT GATE — DO THIS BEFORE ANY TRAINING

The purpose of pre-flight is to prevent wasting the overnight window on a trivial environment failure.

## C1. Repository

Run:

```bash
git checkout main
git pull --ff-only
git status --short
git rev-parse HEAD
```

Requirements:

- on `main`;
- working tree clean;
- HEAD contains the clean-baseline implementation.

Do not start training if the tree is dirty.

## C2. Python environment

Before training, verify:

```bash
python -V
which python
python - <<'PY'
import torch, numpy, h5py
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
print("gpu_count", torch.cuda.device_count())
PY
```

If the current shell has the wrong environment, Codex may activate an already-existing project/conda/venv environment.

Do NOT upgrade/downgrade Torch, CUDA, NumPy, h5py, drivers, or system packages for this task.

If no compatible existing environment can be found, stop.

## C3. GPU health

Run:

```bash
nvidia-smi
```

Before launch verify:

- target GPU is visible;
- no GPU hardware/driver error is reported;
- enough free VRAM exists for the frozen batch=8 protocol;
- there is no unknown foreign job consuming most of the GPU.

If multiple approved GPUs exist, use an idle one and explicitly set:

```bash
export CUDA_VISIBLE_DEVICES=<GPU_ID>
```

Record the selected physical GPU ID and model.

Do NOT kill an unknown/foreign GPU process.

If the GPU is temporarily busy, Codex may wait/recheck rather than change batch size.

## C4. Disk and RAM

Check:

```bash
df -h
free -h
```

Require at least 20 GB free space on the selected output filesystem before launch.

If the preferred output filesystem lacks space, choose another existing local high-capacity filesystem. Do not delete unrelated user files.

## C5. Required assets

Locate only approved released assets:

- released real PIV root;
- official `sim_real_cno.pth`;
- RealPDEBench/model root.

Typical locations may include:

- `/data/p0ab_real_h5_20260830`
- corresponding migrated `/hy-tmp/...`
- `/data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth`
- `/third_party`

Do not search locked-final/private paths.

Verify official initialization SHA256 exactly:

`82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61`

## C6. Tests and imports

Run:

```bash
python -m pytest -q \
  tests/test_clean_baseline_v1.py \
  tests/test_clean_runtime_profiles.py \
  tests/test_h5_ram_preload.py \
  tests/test_colleague_incremental_screen.py
```

Then:

```bash
python -m py_compile \
  tools/colleague_80pt/train_clean_baseline_cno.py \
  tools/colleague_80pt/run_clean_baseline_v1.py \
  tools/colleague_80pt/residual_multi.py
```

Also verify the runner CLI can import:

```bash
python -u -B tools/colleague_80pt/run_clean_baseline_v1.py --help >/dev/null
```

If tests/imports fail, stop.

Do not patch scientific code during this execution task.

---

# D. DATA SPLIT GATE

The runner must validate the canonical split against the actual released data.

Expected:

```text
usable trajectories = 81
Train               = 51
Seen Dev            = 12
AoA10 Holdout       = 18
```

Requirements:

```text
Train ∩ Dev = empty
Train ∩ Holdout = empty
Dev ∩ Holdout = empty

Train/Dev AoA in {0,5,15,20}

Seen Dev:
0°  = 3
5°  = 3
15° = 3
20° = 3

Holdout:
all and only AoA=10°
18 trajectories
```

If the files on the machine do not satisfy this exact audit, stop. Do not alter the manifest.

---

# E. UNATTENDED PROCESS SURVIVAL

The 8–10 hour training MUST NOT depend on the current SSH/Codex shell staying connected.

Preferred launch mechanism:

1. create a unique output root;
2. launch the runner inside a detached `tmux` session;
3. if tmux is unavailable, use `nohup + setsid`;
4. record session name/PID and exact command.

Suggested session name:

`realpde_clean_baseline_v1`

Suggested output root:

`/hy-tmp/runs/realpde_clean_baseline_v1_20260924`

Example with tmux:

```bash
tmux new-session -d -s realpde_clean_baseline_v1 \
  "cd <REPO_ROOT> && \
   export PYTHONUNBUFFERED=1 && \
   export PYTHONHASHSEED=41 && \
   export CUDA_VISIBLE_DEVICES=<GPU_ID> && \
   python -u -B tools/colleague_80pt/run_clean_baseline_v1.py \
     --runtime-profile \
     --real-root <REAL_PIV_ROOT> \
     --sim-pretrain-checkpoint <SIM_REAL_CNO_PT> \
     --model-root <REALPDEBENCH_MODEL_ROOT> \
     --out-root <NEW_OUTPUT_ROOT> \
     --workers 4 \
     > <NEW_OUTPUT_ROOT>.supervisor.log 2>&1"
```

If `--workers 4` fails before meaningful optimizer progress because of DataLoader/HDF5 worker issues, it is permitted to restart once with workers=2, then workers=0 if necessary. No other argument may change.

Do not run two copies concurrently.

Before launching, explicitly check that no existing `realpde_clean_baseline_v1` runner is already active.

---

# F. HEALTH MONITORING

After launch, Codex should periodically verify the process rather than assuming it is healthy.

Approximately every 10–15 minutes, check:

- tmux/session or PID still alive;
- latest training log tail;
- output filesystem free space;
- GPU visible;
- GPU memory/utilization non-pathological;
- latest checkpoint/eval files advancing.

Write lightweight runtime evidence under the output area, for example:

`health.log`

A health check may include:

```bash
date
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,temperature.gpu \
  --format=csv,noheader
df -h <OUTPUT_FILESYSTEM>
tail -n 20 <SUPERVISOR_OR_STAGE_LOG>
```

Do not modify the experiment based on normal metric fluctuations.

Do not early-stop because metrics look bad or good.

---

# G. FAILURE POLICY

## G1. Before first optimizer update

Codex may autonomously fix non-scientific issues and retry, including:

- wrong existing Python environment;
- wrong approved path;
- missing shell environment variable;
- output directory collision;
- insufficient output disk, by moving to another approved local filesystem;
- DataLoader worker failure, by workers 4 -> 2 -> 0;
- SSH/session loss, by relaunching under tmux/nohup if training had not actually begun.

## G2. After meaningful optimizer updates have begun

Protect scientific comparability.

If the training process dies after optimizer updates have begun:

- preserve the entire output directory;
- preserve logs/checkpoints;
- record last completed stage/update;
- diagnose the failure;
- DO NOT silently restart with changed scientific settings;
- DO NOT delete partial evidence;
- DO NOT start another experimental arm.

If an exact continuation from the saved optimizer/scheduler/model state is not explicitly supported by the frozen code, stop and report `REVIEW_REQUIRED` rather than pretending a restart is equivalent.

## G3. OOM

Do NOT solve OOM by changing batch size or precision.

Record:

- GPU model;
- free/used VRAM;
- failing stage/update;
- traceback.

Then stop and report.

## G4. Holdout discipline

If Stage1 or Stage2 fails, do NOT run Holdout18 evaluation.

Holdout is exposed only after successful completion of both stages.

---

# H. FROZEN TRAINING SPECIFICATION

## Stage 1 — CNO

```text
architecture:
colleague-80 CNO

init:
official sim_real_cno.pth

Train:
51 trajectories

training windows:
Past20 -> Future20
stride=1
all legal starts
global deterministic shuffle
seed=41
without replacement per epoch

batch:
8

optimizer:
AdamW
lr=1e-4

scheduler:
CosineAnnealingLR

updates:
8723

loss:
exact colleague-80 Stage1 loss

Seen Dev:
12 trajectories
stride=20
start=0

eval:
step0
every 1000 updates
final
```

Best checkpoint:

```text
point_score =
mean(
  official-v9 Rel-L2 score,
  official-v9 TKE score,
  official-v9 MVPE score
)
```

SPS/time do not participate.

## Stage 2 — Residual

Start from Stage1 `model_best.pth`.

CNO frozen.

```text
ResidualCorrector3D
hidden=96
blocks=2
dropout=0
max_delta=0.04
alpha=1.0
```

Train:

```text
Train51
stride=1
all legal starts
global deterministic shuffle
seed=41
batch=8

AdamW
lr=2e-4
weight_decay=1e-5
gradient clip=1.0

updates=38400
```

Loss:

```text
point          1.0
MSE            0.05
TKE            0.06
temporal       0.03
gradient       0.015
pressure-zero  0.01
residual-MSE   0.25
delta penalty  0.02
```

Dev:

```text
Seen Dev12
stride=20
start=0
eval every 1000
alpha fixed 1.0
point-only selection
```

---

# I. HOLDOUT

AoA10 Holdout18 is never used for:

- gradient;
- early stopping;
- checkpoint selection;
- alpha selection;
- hyperparameter tuning.

Only after Stage1 and Stage2 complete successfully, evaluate:

- Stage2 step0;
- Stage2 best;
- Stage2 final.

Do not use the holdout result to automatically rerun or retune anything.

---

# J. FINAL REPORT

Return:

```text
REALPDE CLEAN BASELINE V1

Status:
REVIEW_REQUIRED / FAILED

Execution commit:
...

Detached execution:
tmux / nohup
session or PID:
...

GPU:
...
CUDA_VISIBLE_DEVICES:
...

Python:
...
Torch:
...
CUDA runtime:
...

Preflight:
PASS / FAIL

Tests:
PASS / FAIL

Disk free at launch:
...

Canonical split:
Train: 51
Seen Dev: 12
AoA10 Holdout: 18

Split audit:
PASS / FAIL

Train/Dev/Holdout overlap:
NONE / ...

Stage1:
best iteration:
best Rel-L2:
best TKE:
best MVPE:
best point_score:

final Rel-L2:
final TKE:
final MVPE:
final point_score:

Stage2 step0:
Rel-L2:
TKE:
MVPE:
point_score:

Stage2 best:
iteration:
Rel-L2:
TKE:
MVPE:
point_score:

Stage2 final:
Rel-L2:
TKE:
MVPE:
point_score:

AoA10 Holdout:

Stage2 step0:
Rel-L2:
TKE:
MVPE:

Stage2 best:
Rel-L2:
TKE:
MVPE:

Stage2 final:
Rel-L2:
TKE:
MVPE:

Stage1 best SHA256:
Stage1 final SHA256:
Stage2 init SHA256:
Stage2 best SHA256:
Stage2 final SHA256:

Last health check:
...

Runtime:
...

OOM:
YES / NO

Automatic recoveries performed:
- ...

Last completed stage/update if failed:
...

Files modified in Git:
NO

Locked-final accessed:
NO

Codabench accessed:
NO

Full-data refit started:
NO

Submission/package started:
NO
```

Do not interpret the experiment beyond a short factual summary.

ChatGPT/Sol will review learning curves and decide the next experiment.


# K. 24G GPU RUNTIME PROFILE

This overnight task must use `--runtime-profile`.

It performs disposable Stage1 and Stage2 batch=8 versus batch=16 benchmarks using RAM-preloaded Train51 / Seen-Dev12 data.

Rules:

- no Holdout18 access;
- no validation metric may influence profile choice;
- only samples/sec and VRAM headroom are used;
- benchmark outputs are not scientific experiment arms;
- batch16 failure is allowed and should fall back to a b8 recommendation;
- the actual REALPDE_CLEAN_BASELINE_V1 training remains batch=8 regardless of the recommendation.

Report for both Stage1 and Stage2:

```text
b8 samples/sec:
b8 peak allocated:
b8 peak reserved:

b16 success:
b16 samples/sec:
b16 peak allocated:
b16 peak reserved:

throughput gain:
recommended future profile:
```

These recommendations are for later 3090 / 3090 Ti 24G experiment families only.
