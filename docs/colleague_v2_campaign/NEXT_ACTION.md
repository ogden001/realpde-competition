# NEXT_ACTION — Arm B only: benchmark then formal residual training

Status: `READY_FOR_EXECUTION / ARM_B_ONLY / REVIEW_REQUIRED`

The SOTA-V2 full-data strong backbone has been successfully reconstructed on
the current GPU host. Do **not** rerun A or C.

## Approved strong backbone

Historical checkpoint SHA256:

`f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce8`

Audited 2026-09-23 reconstruction SHA256:

`cc732555859cb00b3ea1af31415632c087af1fc60abd4f4056263e6f02ffa307`

The reconstruction is approved for Arm B. It is not bitwise identical to the
historical checkpoint and must be recorded as `audited_rebuild_20260923`.

At update 53582 the reconstruction versus historical run differs only by about:

- MSE: +0.033%
- TKE: +0.129%
- Rel: +0.069%
- MVPE: +0.092%
- vorticity: -0.189%

Do not retrain the backbone again.

## Step 1 — sync and focused tests

```bash
git pull --rebase origin main
git status --short
pytest -q \
  tests/test_colleague_v2_campaign.py \
  tests/test_colleague_incremental_screen.py \
  tests/test_post_train_diagnostics.py
```

Engineering/environment fixes are allowed. Do not wait for another prompt for
path/import/launcher issues.

## Step 2 — locate the reconstructed checkpoint

Locate the existing checkpoint with SHA256 exactly:

`cc732555859cb00b3ea1af31415632c087af1fc60abd4f4056263e6f02ffa307`

Expected artifact name:

`model_update_53582.pth`

Use the existing file. Do not copy it into Git.

## Step 3 — Arm B batch benchmark only

Run the existing benchmark with:

`--arms B_strong_backbone`

Benchmark exactly 200 updates at b8 and b16.

Selection rule remains:

- b16 throughput >= 1.20 × b8;
- peak allocated <= 92%;
- peak reserved <= 96%;
- no OOM;
- do not use validation quality to choose batch.

Frozen profiles:

- b8: batch=8, lr=2e-4, updates=38400, eval=4800
- b16: batch=16, lr=2.8e-4, updates=19200, eval=2400

If b16 does not clearly win, use b8. Do not try batch24/32.

## Step 4 — immediately run formal Arm B

Do not wait for Sol review after the 200-step benchmark.

Arm B scientific definition:

- base = frozen reconstructed SOTA-V2 P0-A/MF-CNO full-data checkpoint;
- base weights frozen;
- fresh zero-initialized colleague `ResidualCorrector3D`;
- hidden=96;
- blocks=2;
- max_delta=0.04;
- dropout=0;
- alpha=1;
- NO continuation from colleague residual checkpoint;
- scalar colleague residual objective;
- optimizer AdamW;
- weight_decay=1e-5;
- seed=41.

Loss weights:

- point = 1.0
- mse = 0.05
- tke = 0.06
- temporal = 0.03
- grad = 0.015
- p_zero = 0.01
- residual_mse = 0.25
- delta_penalty = 0.02

Use the selected b8/b16 profile exactly.

The trainer must record both:

1. frozen strong-backbone metrics before residual correction;
2. final/best corrected metrics after residual training.

This is required because the key scientific question is whether the colleague
residual adds value on top of the stronger backbone or destroys its signal.

## Step 5 — evidence and diagnostics

Run standard post-train diagnostics and build the compact training review log.

Archive lightweight evidence only under:

`docs/colleague_v2_campaign/results/20260923_formal_b_training/`

Include:

- benchmark_results.json / selected profile
- run_config.json
- final_primary_metrics.json
- summary.json
- eval milestones
- training review log + meta
- standard trajectory/horizon diagnostics
- checkpoint SHA manifest
- explicit backbone provenance = `audited_rebuild_20260923`

Do not commit checkpoint binaries.

## Mechanical review gate

Relative to current colleague80 baseline:

- TKE improvement >= 3%
- Rel degradation <= 1%
- MVPE degradation <= 1%

This gate is for review only. Do not automatically start Combo or another
experiment.

## Hard constraints

- A rerun: NO
- C rerun: NO
- backbone retrain: NO
- locked-final/private: NO
- Codabench: NO
- automatic Combo: NO
- automatic SOTA merge: NO
- parameter sweep: NO

## Final handoff

Return:

```text
REALPDE ARM B STRONG BACKBONE

Status:
REVIEW_REQUIRED / BLOCKED

Execution commit:
...

Tests:
PASS / FAIL

Strong backbone:
provenance: audited_rebuild_20260923
sha256: cc732555...

Batch benchmark:
b8 samples/s:
b16 samples/s:
throughput gain:
selected profile:

Base before residual:
Rel-L2:
TKE:
MVPE:

Best corrected:
step:
Rel-L2:
TKE:
MVPE:

Final corrected:
Rel-L2:
TKE:
MVPE:

Change vs strong base:
Rel-L2:
TKE:
MVPE:

Mechanical gate:
PASS / FAIL

Standard diagnostics:
PASS / FAIL

Results commit:
...

Automatic Combo started:
NO

Codabench accessed:
NO

Locked-final accessed:
NO
```

Then stop for Sol review.
