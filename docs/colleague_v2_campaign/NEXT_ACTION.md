# NEXT_ACTION — Run now, do not globally block

Status: `READY_FOR_EXECUTION / RUN_NOW / REVIEW_REQUIRED`

ChatGPT/Sol is relaxing the previous over-strict orchestration. The priority is
to get useful GPU work running immediately. Codex may make bounded engineering
and environment changes needed to execute the frozen experiments.

## Core rule

**Missing Arm B strong-backbone checkpoint must NOT block Arm A or Arm C.**

Run tests first. Then benchmark/train A and C immediately. Resolve B
independently.

## Allowed Codex autonomy

Codex may, without asking for another prompt:

- fix paths, environment variables, imports, Python paths, DataLoader workers,
  launcher wiring and checkpoint discovery;
- add a small `--arms` / `--skip-arm` style option if useful;
- split the unified launcher into per-arm execution if that is simpler;
- use shell scripts / tmux / nohup / detached runners;
- search local disks, OSS and existing assets;
- recover or reconstruct the documented strong-backbone asset;
- make bounded integration fixes required to run on this 3090 Ti host.

Do not change the scientific definition of Arm A, B or C without recording it
explicitly as a deviation.

## Step 1 — tests

Run the focused test suite immediately. The missing Arm B checkpoint is not a
reason to skip tests.

## Step 2 — A/C batch benchmark

Benchmark Arm A and Arm C at b8 and b16 using the existing benchmark logic:

- 200 steps per profile;
- b16 uses lr 2.8e-4;
- b8 uses lr 2e-4;
- select b16 only if throughput gain >= 20% and VRAM guard passes;
- selection uses no validation quality metric.

If the current benchmark script requires Arm B preflight, patch it so A/C can
run independently. This is an allowed engineering fix.

## Step 3 — immediately start formal Arm A and Arm C

After each A/C profile is selected, start its frozen formal training without
waiting for Arm B:

### Arm A
- mature colleague 80pt residual start;
- Pareto-TKE projection semantics unchanged;
- lambda_TKE = 0.12;
- b8 profile: batch8, lr2e-4, 12k updates, eval2k;
- b16 profile: batch16, lr2.8e-4, 6k updates, eval1k;
- equal sample exposure.

### Arm C
- mature colleague 80pt residual start;
- adjacent-AoA mean-field semantics unchanged;
- b8 profile: batch8, lr2e-4, 20k updates, eval2.5k;
- b16 profile: batch16, lr2.8e-4, 10k updates, eval1.25k;
- equal sample exposure;
- historical b8 step5000 maps to b16 step2500.

A and C may run sequentially on the single GPU. Prefer the order that minimizes
idle time; do not wait for additional approval.

## Step 4 — resolve Arm B independently

The historical strong-backbone evidence is:

- run: `docs/sota迭代/reviews/sota_v2_full_20260916/`
- update: `53582`
- exact historical checkpoint:
  `/home/chyfuture/realpde_runs/sota_v2_full_20260916/run/checkpoints/model_update_53582.pth`
- correct SHA256:
  `f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce8`

If the binary cannot be recovered, Codex is allowed to reconstruct the **same
documented SOTA-V2 full-data refit** from repository code/evidence, provided the
recipe is preserved and all deviations are logged. Do not substitute an
arbitrary "similar" checkpoint silently.

Arm B must not hold A/C hostage.

Once B asset is available:
- benchmark b8/b16;
- select profile by the same throughput/VRAM rule;
- train fresh h96/b2 residual with the frozen original colleague scalar
  residual objective;
- no joint backbone fine-tuning.

## Hard scientific constraints

Still forbidden:

- locked-final/private access;
- Codabench submission;
- automatic Combo;
- changing A/B/C scientific mechanisms;
- parameter sweeps disguised as environment fixes.

## Evidence

Keep lightweight Git evidence and remote checkpoints/logs. Record:

- exact execution commit;
- any engineering fixes;
- selected batch profile per arm;
- throughput and peak VRAM;
- training progress and standard diagnostics;
- checkpoint paths and SHA256;
- whether B was recovered or reconstructed.

## Final handoff

Return:

`REALPDE COLLEAGUE80 V2 RUN`

with:

- Status: REVIEW_REQUIRED / PARTIAL / BLOCKED
- Tests PASS/FAIL
- Arm A benchmark profile + training completion + metrics
- Arm C benchmark profile + training completion + metrics
- Arm B asset status + benchmark/training status
- engineering fixes made
- Codabench accessed: NO
- locked-final accessed: NO
- automatic Combo started: NO

Do not stop merely because Arm B is missing if A or C can still run.
