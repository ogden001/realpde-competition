# RealPDE Tail-Cliff Loss-Geometry Campaign

Status: REVIEW_REQUIRED

This task is intentionally bounded. It has exactly one scientific hypothesis,
one no-training diagnosis, and at most one 5,000-update repair candidate.

## Scientific hypothesis

The strong SOTA-V2 backbone's F19/F20 cliff is primarily an optimization-loss
geometry effect: the historical Stage-B objective remains dominated by a
unit-weight absolute velocity MSE term, which can keep improving easier
early/middle horizons while leaving the hardest tail under-corrected.

Historical evidence motivates exactly one repair candidate.

Historical Stage-B:
1.0*MSE + 0.05*TKE + 0.027514*Rel + 0.009757*MVPE
+ 15.5385751724*Vorticity + 0.027514*Rel(extra)

Candidate:
1.0*Rel + 0.05*TKE + 0.027514*Rel + 0.009757*MVPE
+ 15.5385751724*Vorticity + 0.027514*Rel(extra)

Only 1.0*MSE -> 1.0*Rel changes.

Do not add horizon ramps, tail-only training losses, new lambdas, new heads,
model changes, learning-rate sweeps, seeds, samplers, residual training,
uncertainty training, or any other candidate.

## Frozen provenance

Strong backbone expected iteration: 53582

Expected SHA256:
cc732555859cb00b3ea1af31415632c087af1fc60abd4f4056263e6f02ffa307

Use the same colleague Dev16 manifest used by the completed late-horizon
campaign. If the historical remote manifest path is absent, use a byte-identical
copy only after verifying its SHA/provenance. Do not substitute a different
Dev16.

Use the same all-released PIV root and official v9 kit used by the completed
campaign.

## Preflight

Pull latest main. The checkout must contain at least commit:

a52e9875d996e032e36961d8b82491401b054b2b

Run:

~~~bash
PYTHONPATH=tools:tools/colleague_80pt \
python -m pytest -q \
  tests/test_late_horizon_loss_geometry_campaign.py \
  tests/test_late_horizon_backbone_campaign.py \
  tests/test_post_train_diagnostics.py

python -m py_compile tools/late_horizon_loss_geometry_campaign.py
~~~

Any failure caused by code is BLOCKED: report it and do not change the
scientific contract. Pure path/import/environment compatibility fixes are
allowed only if they do not change data, model, loss, optimizer, sampling,
budget, or evaluation semantics.

## Phase A: mechanism diagnosis only

Create a fresh remote output directory, for example:

~~~bash
OUT=/hy-tmp/realpde_runs/late_horizon_loss_geometry_20260923_v1
~~~

If it already exists, use v2, v3, etc. Never overwrite an earlier run.

Run only:

~~~bash
PYTHONPATH=tools:tools/colleague_80pt \
python tools/late_horizon_loss_geometry_campaign.py diagnose \
  --real-root /hy-tmp/realpde_data/train_real \
  --split-manifest <EXACT_COLLEAGUE_DEV16_MANIFEST> \
  --kit-root /hy-tmp/realpde_t1_kit_v9/realpde_t1_starting_kit_v9 \
  --strong-backbone /hy-tmp/realpde_runs/sota_v2_full_20260922_rebuild/run/checkpoints/model_update_53582.pth \
  --out-root "$OUT" \
  --require-cuda
~~~

Phase A performs zero optimizer steps. It selects one deterministic window from
each of 8 distinct Dev trajectories and measures parameter-gradient alignment
with a tail-only F19/F20 Rel-L2 diagnostic.

Required outputs:

- diagnosis/probe_windows.csv
- diagnosis/gradient_alignment.csv
- diagnosis/diagnosis_result.json
- diagnosis/DIAGNOSIS_DONE

Archive Phase-A evidence before review. Use a fresh destination and do not
overwrite prior evidence:

~~~bash
DEST=docs/colleague_screening/results/20260923_tail_loss_geometry_diagnosis
# If DEST exists, use a v2/v3 suffix.
python tools/colleague_80pt/archive_v2_campaign.py   --run-root "$OUT"   --dest "$DEST"

git add "$DEST"
git diff --check
git commit -m "Archive tail loss-geometry diagnosis"
git pull --rebase
git push
~~~

The archive must exclude checkpoints and raw data. Record the pushed results
commit in the report.

After Phase A, STOP. Do not invoke train. This is the mandatory REVIEW_REQUIRED
boundary.

Report exactly:

~~~text
REALPDE TAIL LOSS-GEOMETRY DIAGNOSIS

Status:
REVIEW_REQUIRED

Execution commit:
<sha>

Tests:
<PASS/FAIL>

Strong checkpoint SHA:
<sha>

Probe trajectories:
<count>

Control tail cosine mean:
<value>

Rel-dominant candidate tail cosine mean:
<value>

Candidate - control tail cosine:
<value>

MSE tail cosine mean:
<value>

Rel tail cosine mean:
<value>

Rel - MSE tail cosine:
<value>

Candidate alignment wins:
<x>/<batches>

Rel alignment wins:
<x>/<batches>

Optimizer steps:
0

Train phase started:
NO

Codabench accessed:
NO

Locked-final accessed:
NO

Results commit:
<sha>
~~~

Do not give an automatic GO/NO-GO. Sol reviews the mechanism evidence.

## Phase B: prewritten repair screen, NOT authorized until review

The code is already implemented so no code edit should be needed after review.
Only after explicit Sol/user approval, run train mode on the same OUT directory.

The frozen historical Control is reused, not retrained:

docs/colleague_screening/results/20260923_late_horizon_backbone_campaign/experiment1_backbone_continuation/control/result.json

docs/colleague_screening/results/20260923_late_horizon_backbone_campaign/experiment1_backbone_continuation/control/run_config.json

The train runner hard-checks:

- exact strong-backbone SHA;
- restored optimizer state;
- LR 3e-6;
- all 82 released PIV trajectories;
- Dense-All;
- micro-batch 4, accumulation 2, effective batch 8;
- seed 20260901;
- exactly 5000 updates;
- eval at 0/1k/2k/3k/4k/5k;
- candidate sampler-prefix SHA equals the frozen historical Control;
- step-0 metric parity with the frozen historical Control.

Authorized command, only after review:

~~~bash
PYTHONPATH=tools:tools/colleague_80pt \
python tools/late_horizon_loss_geometry_campaign.py train \
  --real-root /hy-tmp/realpde_data/train_real \
  --split-manifest <EXACT_COLLEAGUE_DEV16_MANIFEST> \
  --kit-root /hy-tmp/realpde_t1_kit_v9/realpde_t1_starting_kit_v9 \
  --strong-backbone /hy-tmp/realpde_runs/sota_v2_full_20260922_rebuild/run/checkpoints/model_update_53582.pth \
  --frozen-control-result docs/colleague_screening/results/20260923_late_horizon_backbone_campaign/experiment1_backbone_continuation/control/result.json \
  --frozen-control-config docs/colleague_screening/results/20260923_late_horizon_backbone_campaign/experiment1_backbone_continuation/control/run_config.json \
  --out-root "$OUT" \
  --require-cuda
~~~

Primary comparison is fixed in advance:

Rel-dominant@5000 vs frozen historical Control@5000

Review F18/F19/F20, F20/F18, aggregate Rel-L2, TKE, MVPE, and trajectory-level
diagnostics. A tail improvement that merely damages early horizons or trades
away TKE/MVPE is not a clean repair.

## Forbidden

- no training before Phase-A review;
- no continuation beyond 5k;
- no second Rel coefficient;
- no MSE/Rel interpolation sweep;
- no horizon-ramp combination;
- no MF/non-MF architecture ablation in this task;
- no residual or uncertainty training;
- no full refit;
- no submission/package;
- no Codabench;
- no locked-final/private access;
- no automatic follow-up experiment.

All results remain REVIEW_REQUIRED.
