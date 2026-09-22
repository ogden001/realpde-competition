# NEXT_ACTION — 3090 batch-size benchmark only

Status: `READY_FOR_EXECUTION / BENCHMARK_ONLY / REVIEW_REQUIRED`

`REQUIRED_COMMIT = 9211a4b951195a45f0dad89f0f2d26900041aa5b`

ChatGPT/Sol has implemented the b8/b16 profiling path. This task is **only**
to measure the actual RTX 3090 24G throughput/VRAM for the three frozen training
paths and return the selected profiles.

**Do not start the long A/B/C campaign in this task.**

## Goal

For each arm, benchmark exactly 200 synchronized training steps at:

- b8, lr 2e-4
- b16, lr 2.8e-4

The benchmark runner automatically selects b16 only if:

- b16 succeeds without OOM;
- b16 samples/sec >= 1.20 × b8;
- b16 peak allocated VRAM <= 92%;
- b16 peak reserved VRAM <= 96%.

Otherwise that arm selects b8.

Selection must use **no validation metric**.

## Frozen full-training mappings

These are encoded in `batch_profiles.py`; do not edit them.

| Arm | b8 | b16 |
| --- | --- | --- |
| A Pareto-TKE | b8 / lr2e-4 / 12k / eval2k | b16 / lr2.8e-4 / 6k / eval1k |
| B Strong Backbone | b8 / lr2e-4 / 38.4k / eval4.8k | b16 / lr2.8e-4 / 19.2k / eval2.4k |
| C AoA Mean-Field | b8 / lr2e-4 / 20k / eval2.5k | b16 / lr2.8e-4 / 10k / eval1.25k |

Sample exposure is exactly matched between b8 and b16.

## Preflight

```bash
git fetch origin
git pull --rebase origin main
git status --short
git rev-parse HEAD
git rev-parse origin/main
git merge-base --is-ancestor 9211a4b951195a45f0dad89f0f2d26900041aa5b HEAD
git push --dry-run origin HEAD:main
nvidia-smi
```

Requirements:

- worktree clean;
- `HEAD == origin/main`;
- required commit is an ancestor of HEAD;
- CUDA available;
- target device is RTX 3090 24G;
- released-data audit passes;
- colleague base/residual checkpoint SHA checks pass;
- strong SOTA-V2 full@53582 checkpoint SHA256 is exactly:
  `f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce`.

Locate the strong checkpoint on the machine by SHA. Do not substitute another
checkpoint. If unavailable, stop as `BLOCKED`.

## Tests before benchmark

```bash
pytest -q \
  tests/test_colleague_v2_campaign.py \
  tests/test_colleague_incremental_screen.py \
  tests/test_colleague_next_two_experiments.py \
  tests/test_post_train_diagnostics.py
```

Only after PASS may benchmark start.

## Execute benchmark

Use the actual existing paths on the GPU host. Typical command:

```bash
python -u -B tools/colleague_80pt/benchmark_v2_batch_profiles.py \
  --real-root /hy-tmp/realpde_data/train_real \
  --split-manifest /hy-tmp/realpde_runs/colleague_incremental_screen_20260921/colleague_dev16_manifest.json \
  --data-manifest /hy-tmp/realpde_data/data_manifest.tsv \
  --colleague-base-checkpoint /hy-tmp/realpde_assets/colleague-80pt/20260921/extracted/handoff_colleague_80pt_archive_20260921/checkpoints/cno_final_all81.pt \
  --colleague-residual-checkpoint /hy-tmp/realpde_assets/colleague-80pt/20260921/extracted/handoff_colleague_80pt_archive_20260921/checkpoints/residual_model_best.pth \
  --colleague-model-root tools/colleague_80pt/submission \
  --strong-backbone-checkpoint <PATH_WITH_SHA_f808fbd3...> \
  --strong-kit-root tools/colleague_80pt/submission \
  --out-root /hy-tmp/realpde_runs/colleague80_v2_batch_benchmark_20260922_v1 \
  --workers 4
```

If output root exists, use v2/v3. Never delete a prior run.

The script runs:

- Arm A b8 200 steps
- Arm A b16 200 steps
- Arm B b8 200 steps
- Arm B b16 200 steps
- Arm C b8 200 steps
- Arm C b16 200 steps

These runs are disposable profiling runs only. Their checkpoints must never be
used as scientific starting points.

## Required evidence

Return:

- `benchmark_results.json`
- `selected_profiles.json`
- for each arm/profile:
  - success / failure
  - samples/sec
  - training-step seconds
  - peak allocated VRAM GiB and fraction
  - peak reserved VRAM GiB and fraction
  - b16/b8 throughput gain
  - automatic selection
- GPU model and total memory
- execution commit

Copy only the two JSON summaries into:

`docs/colleague_v2_campaign/results/20260922_batch_benchmark_<run-id>/`

Do not commit benchmark checkpoints or raw full logs.

Then:

```bash
git diff --check
git status --short
git add docs/colleague_v2_campaign/results/20260922_batch_benchmark_<run-id>
git commit -m "Archive colleague80 batch profile benchmark"
git pull --rebase origin main
git push origin main
```

Do not use `git add .`.

## Hard constraints

- Long A/B/C training: **NO**
- Automatic campaign after benchmark: **NO**
- Locked-final/private: **NO**
- Codabench: **NO**
- Scientific metric selection: **NO**
- Manual editing of `selected_profiles.json`: **NO**
- Testing batch24/batch32: **NO**
- Changing LR scaling/profile thresholds: **NO**

Environment-only fixes are allowed. If a fix changes scientific or profile
semantics, stop.

## Final handoff

Return:

```text
REALPDE 3090 BATCH BENCHMARK

Status:
REVIEW_REQUIRED / BLOCKED

Execution commit:
...

Results commit:
...

Tests:
PASS / FAIL

GPU:
...

Arm A Pareto-TKE:
b8 samples/s:
b8 peak VRAM:
b16 success:
b16 samples/s:
b16 peak VRAM:
throughput gain:
selected profile:

Arm B Strong Backbone:
b8 samples/s:
b8 peak VRAM:
b16 success:
b16 samples/s:
b16 peak VRAM:
throughput gain:
selected profile:

Arm C AoA Mean-Field:
b8 samples/s:
b8 peak VRAM:
b16 success:
b16 samples/s:
b16 peak VRAM:
throughput gain:
selected profile:

Long campaign started:
NO

Codabench accessed:
NO

Locked-final accessed:
NO

Missing items:
NONE / ...
```

Then stop and wait for Sol review.
