# NEXT ACTION: current80 ↔ A18 corrector weight interpolation

Status: `READY_FOR_EXECUTION / REVIEW_REQUIRED`

This is a zero-training evaluation. Do not start any training job.

## Scientific question

Can a weight-space interpolation between the mature colleague current80 residual
and Arm A@18k retain most of A18's TKE gain while bringing Rel-L2 back inside
the conservative +0.5% gate?

Frozen betas:

- 0.50
- 0.65
- 0.80
- 1.00

Runtime residual alpha is fixed to 1.0. Do not scan alpha.

## Exact checkpoints

current80 residual SHA256:

`909fdc7f8a6a42335e4ea4ce7471fc9a507135e9a29a4927bc7b7318be9c85b2`

A18 best residual SHA256:

`852b41cda7d15ce9b205aecb32e206712e7a036b18981a62a9afc4e9a0b9a238`

Expected A18 path from the completed continuation:

`/hy-tmp/realpde_runs/colleague80_v2_arm_a_pareto_tke_continuation_20260923/A_pareto_tke/model_best.pth`

Locate the current80 checkpoint by exact SHA on the current host. Do not
substitute another residual checkpoint.

## Implementation

Use:

`tools/colleague_80pt/eval_a18_weight_interpolation.py`

The script:

- verifies both checkpoint SHA256 values;
- requires bitwise-identical non-corrector state/backbone;
- interpolates only `corrector.*` tensors;
- evaluates exactly the 640-window Dev protocol;
- fixes runtime alpha=1;
- uses the historical fixed runtime for comparable local final_est;
- checks beta=1 reproduces A18 metrics;
- writes one JSON per beta, `per_beta.csv`, `summary.json`, and `DONE`;
- starts no training and writes no checkpoint binaries.

## Execute

First sync main and run focused tests:

```bash
git pull --rebase origin main
pytest -q tests/test_a18_weight_interpolation.py tests/test_colleague_v2_campaign.py
```

Then run the evaluator with the existing approved real-data root, colleague
Dev16 split/data manifests, current80 checkpoint, A18 checkpoint, and model
root.

Example shape only, resolve existing host paths yourself:

```bash
python -u -B tools/colleague_80pt/eval_a18_weight_interpolation.py \
  --real-root <train_real> \
  --split-manifest <colleague_dev16_manifest.json> \
  --data-manifest <data_manifest.tsv> \
  --current80-checkpoint <sha909fdc...pth> \
  --a18-checkpoint /hy-tmp/realpde_runs/colleague80_v2_arm_a_pareto_tke_continuation_20260923/A_pareto_tke/model_best.pth \
  --model-root <official_v9_model_root> \
  --out-root /hy-tmp/realpde_runs/a18_weight_interpolation_20260923
```

Engineering path/import/environment fixes are allowed. Do not change the
scientific variables.

## Review criteria

For each beta report:

- Rel-L2
- TKE
- MVPE
- delta of each metric versus current80
- local final_est
- mechanical gate pass/fail

Mechanical gate:

- TKE improvement >= 3%
- Rel-L2 degradation <= 0.5%
- MVPE degradation <= 0.3%

The script selects the highest final_est among gate-passing betas. This is only
a review candidate, not permission to submit or merge.

## Evidence

Archive lightweight outputs only under:

`docs/colleague_v2_campaign/results/20260923_a18_weight_interpolation/`

Include:

- `run_manifest.json`
- `per_beta.csv`
- `beta_*.json`
- `summary.json`

Do not commit checkpoint binaries or training data.

## Hard constraints

- no training;
- no extra beta sweep;
- no runtime alpha sweep;
- no A/B/C retraining;
- no Combo;
- no automatic SOTA merge;
- no Codabench;
- no locked-final/private access.

## Final handoff

Return:

```text
REALPDE A18 WEIGHT INTERPOLATION

Status:
REVIEW_REQUIRED / BLOCKED

Execution commit:
...

Tests:
PASS / FAIL

current80 SHA:
...

A18 SHA:
...

beta=0.50:
Rel-L2:
TKE:
MVPE:
vs current80:
final_est:
gate:

beta=0.65:
...

beta=0.80:
...

beta=1.00:
...

A18 parity:
PASS / FAIL

Gate-passing betas:
...

Selected beta:
...

Selected final_est:
...

Results commit:
...

Training started:
NO

Codabench accessed:
NO

Locked-final accessed:
NO
```

Then stop for Sol review.
