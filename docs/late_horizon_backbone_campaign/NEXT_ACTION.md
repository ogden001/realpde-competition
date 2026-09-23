# NEXT_ACTION — Late-horizon backbone fingerprint + matched ramp continuation

Status: `READY_FOR_EXECUTION / REVIEW_REQUIRED_AFTER_RUN`

## Scientific question

We repeatedly observed a sharp F19/F20 error rise in our 77-point SOTA family.
The colleague 80-point pipeline does not show the same tail shape. Arm B reused
our audited SOTA-V2 backbone and the F19/F20 rise returned even after the
colleague residual corrector.

This task answers two questions in one bounded campaign:

1. Is the late-horizon cliff already present in our backbone before residual correction?
2. If yes, can stronger late-horizon supervision reduce it without sacrificing earlier horizons?

No locked-final/private data and no Codabench access are allowed.

## Code

Main runner:

```bash
tools/late_horizon_backbone_campaign.py
```

Tests:

```bash
tests/test_late_horizon_backbone_campaign.py
```

The runner performs both experiments serially and stops after the 5k matched screen.

## Experiment 0 — same-window horizon fingerprint, no training

Replay the exact same colleague Dev16 / stride20 / 640 windows for:

- colleague CNO backbone, SHA256
  `ff28aaf0114d57e320e5ea9cf9945d874a7482f22e3f0aeaed5ebbe286573f8a`
- current80 residual model, SHA256
  `909fdc7f8a6a42335e4ea4ce7471fc9a507135e9a29a4927bc7b7318be9c85b2`
- audited SOTA-V2 strong backbone @53582, SHA256
  `cc732555859cb00b3ea1af31415632c087af1fc60abd4f4056263e6f02ffa307`
- Arm-B final residual model @38400, SHA256
  `fde7f3c20e1e42f93f616741a3c6e1a68d4e38c4dc002f3f4875f9f2c5b18762`

Required evidence:

- aggregate Rel-L2 / TKE / MVPE
- F1..F20 frame Rel-L2 and velocity RMSE
- F1..F20 TKE contribution metrics
- F17/F18/F19/F20 Rel
- `F20 / mean(F1:F17)`
- `F20 / F18`
- mean/fluctuation/TKE-energy diagnostics

The purpose is attribution only. Do not use this replay to select a submission.

## Experiment 1 — matched 5k backbone continuation

Both arms start from the same audited strong-backbone checkpoint @53582 and
restore the same optimizer state from that checkpoint.

Common contract:

- train: all 82 released PIV trajectories
- Dense-All sampling
- micro-batch 4
- grad accumulation 2
- effective batch 8
- seed `20260901`
- AdamW optimizer state restored from checkpoint
- lr forced to historical Stage-B `3e-6`
- 5000 updates
- eval at 0/1000/2000/3000/4000/5000
- same Dev16 / 640 windows
- gradient clip 1.0
- fp32
- same P0-A / MF-CNO architecture
- same N2 / vorticity / Stage-B extra-Rel objective
- no residual training
- no uncertainty-head training

### Arm A: Control

Historical Stage-B objective exactly:

```text
MSE
+ 0.05*TKE
+ 0.027514*Rel
+ 0.009757*MVPE
+ 15.5385751724*vorticity
+ 0.027514*extra Rel
```

### Arm B: Ramp

The only scientific change is the existing N2 MSE term.

Control uses uniform Future20 MSE.

Ramp uses:

```text
raw horizon weight: F1=1.0 -> F20=2.0, linear
normalized by mean weight
mean normalized weight = 1.0
```

Only the velocity MSE term is reweighted. TKE, Rel, MVPE, vorticity, extra-Rel,
data, optimizer, LR, initialization and sample order remain unchanged.

Implementation equivalently computes:

```text
L_ramp = L_control + (ramped_velocity_MSE - uniform_velocity_MSE)
```

Because the historical N2 MSE coefficient is exactly 1.0.

This preserves the average MSE loss scale and isolates the horizon-weighting variable.

## Primary review

Primary comparison:

```text
Ramp@5000 vs Control@5000
```

Do not choose different early-stop steps to manufacture a win.

Review:

1. Does Ramp reduce F19/F20 Rel and `F20/F18`?
2. Is the improvement selective to the tail rather than shifting error earlier?
3. Are aggregate Rel-L2 and MVPE protected?
4. Does TKE remain stable?
5. Is the effect visible by 1k-3k and sustained at 5k?

No automatic GO/NO-GO and no automatic long continuation.

## Preflight

Pull latest main and record execution HEAD.

Run:

```bash
python -m pytest -q \
  tests/test_late_horizon_backbone_campaign.py \
  tests/test_post_train_diagnostics.py \
  tests/test_colleague_v2_campaign.py
```

Also:

```bash
python -m py_compile tools/late_horizon_backbone_campaign.py
```

If a pure environment/path issue occurs, fix minimally and continue.
If a scientific contract must change, return `BLOCKED`.

## Execute

Locate the exact checkpoint files by SHA256. Expected paths on the current GPU
host are approximately:

```text
COLLEAGUE_BASE=/hy-tmp/realpde_assets/colleague-80pt/20260921/extracted/handoff_colleague_80pt_archive_20260921/checkpoints/cno_final_all81.pt
CURRENT80=/hy-tmp/realpde_assets/colleague-80pt/20260921/extracted/handoff_colleague_80pt_archive_20260921/checkpoints/residual_model_best.pth
STRONG=/hy-tmp/realpde_runs/sota_v2_full_20260922_rebuild/run/checkpoints/model_update_53582.pth
ARM_B=/hy-tmp/realpde_runs/colleague80_v2_arm_b_formal_20260923/B_strong_backbone/model_final.pth
```

Do not trust the path without the SHA check.

Run:

```bash
python -u -B tools/late_horizon_backbone_campaign.py \
  --real-root /hy-tmp/realpde_data/train_real \
  --split-manifest /hy-tmp/realpde_runs/colleague_incremental_screen_20260921/colleague_dev16_manifest.json \
  --kit-root /hy-tmp/realpde_t1_kit_v9/realpde_t1_starting_kit_v9 \
  --colleague-base "$COLLEAGUE_BASE" \
  --current80-residual "$CURRENT80" \
  --strong-backbone "$STRONG" \
  --arm-b-final "$ARM_B" \
  --out-root /hy-tmp/realpde_runs/late_horizon_backbone_campaign_20260923_v1 \
  --workers 4 \
  --require-cuda
```

If the output root exists, use v2/v3. Do not overwrite or delete prior runs.

## Archive

After `DONE`, archive lightweight evidence only:

```bash
python -u -B tools/colleague_80pt/archive_v2_campaign.py \
  --run-root /hy-tmp/realpde_runs/late_horizon_backbone_campaign_20260923_v1 \
  --dest docs/colleague_screening/results/20260923_late_horizon_backbone_campaign
```

If destination exists, use v2/v3.

Then:

```bash
git diff --check
git add docs/colleague_screening/results/20260923_late_horizon_backbone_campaign*
git commit -m "Archive late-horizon backbone campaign"
git pull --rebase origin main
git push origin main
```

Do not commit checkpoint binaries or raw training data.

## Stop

After archive + push, stop.

Do not automatically:

- continue beyond 5k
- train residuals
- train uncertainty heads
- package submission
- access Codabench
- access locked-final/private
- sweep ramp endpoints or LR

Return:

```text
REALPDE LATE-HORIZON BACKBONE CAMPAIGN

Status:
REVIEW_REQUIRED / BLOCKED

Execution commit:
...

Tests:
PASS / FAIL

Experiment 0 fingerprint:
COMPLETE / BLOCKED

Colleague backbone F18/F19/F20:
...

Current80 F18/F19/F20:
...

Strong backbone F18/F19/F20:
...

Arm B F18/F19/F20:
...

Strong vs colleague F20/F18 amplification:
...

Arm B vs current80 F20/F18 amplification:
...

Experiment 1:
Control COMPLETE / BLOCKED
Ramp COMPLETE / BLOCKED

Control@5000:
Rel:
TKE:
MVPE:
F18:
F19:
F20:
F20/F18:

Ramp@5000:
Rel:
TKE:
MVPE:
F18:
F19:
F20:
F20/F18:

Ramp vs Control:
Rel %:
TKE %:
MVPE %:
F20 %:
F20/F18 %:

Long continuation started:
NO

Residual training started:
NO

Uncertainty training started:
NO

Codabench accessed:
NO

Locked-final accessed:
NO

Results commit:
...
```

Codex does not make the final scientific judgment. Sol reviews the archived evidence.
