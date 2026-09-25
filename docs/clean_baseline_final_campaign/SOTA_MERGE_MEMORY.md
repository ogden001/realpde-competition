# SOTA MERGE MEMORY — Validated SPS Module

Status: `REVIEW_REQUIRED / SCIENTIFIC GO`

Purpose: this file is a compact merge-time memory for the validated residual-aware SPS module. Read this before building the final RealPDE Track 1 SOTA candidate.

## 1. What is validated

The validated method is **Residual-aware SPS** on the clean baseline:

`Past20 + pre-residual base forecast -> uncertainty head -> uncertainty of post-residual final forecast`

The point predictor is frozen and unchanged by the SPS head.

The uncertainty head learns heteroscedastic error structure for the final predictor. It does not change Rel-L2, TKE or MVPE point predictions.

## 2. Frozen training recipe

- Point predictor: Clean Stage1 CNO + Clean Stage2 residual model, frozen.
- Head: h64, 2 blocks, dropout0.
- Head input: Past20 plus **pre-residual base CNO forecast**.
- Target: absolute error of the **post-residual final forecast**.
- Loss: masked log-MAE.
- Train split: Train51.
- SPS training stride: 5.
- Seen-Dev: Seen-Dev12, stride20/start0.
- Batch: 16.
- Optimizer: AdamW, lr=1e-3, warmup/cosine.
- Training budget: 5000 updates.
- Evaluation cadence: every 500 updates.
- Calibration family: original fixed 300-combination adaptive grid only.
- No point-model training during SPS training.

## 3. Correct calibration-selection rule

Do **not** select the unconstrained maximum SPS and reject afterward.

Correct rule:

1. compute same-Dev best static interval;
2. for every adaptive calibration, require:
   `mean_width <= 1.20 * static_mean_width`;
3. among feasible candidates, select highest Seen-Dev SPS;
4. select checkpoint by this constrained SPS.

Frozen gate:

- Seen-Dev adaptive SPS gain >= `+1.0`;
- width ratio <= `1.20`;
- point-prediction parity max abs <= `1e-7`.

The unconstrained SPS maximum is diagnostic only.

## 4. Validated checkpoint and calibration

Selected checkpoint:

- step: `4500`
- head SHA256: `1cde13a23fe15547dc2ee16cede9edd89fe8c12bc88f8e9ad059641f0a6c90b8`

Selected calibration:

- `floor = 0.0025`
- `mult_u = 0.75`
- `mult_v = 1.5`
- `rel = 0.0075`

## 5. Seen-Dev evidence

Static:

- SPS: `48.4163598`
- coverage: `0.7883023`
- mean u/v width: `0.01243318`

Adaptive:

- SPS: `51.6544349`
- gain: `+3.2380751`
- coverage: `0.8496342`
- mean u/v width: `0.01482781`
- width ratio: `1.1926004`
- point parity max abs: `0`

Gate: `GO`.

## 6. Unseen AoA10 frozen audit

AoA10 Holdout18 was accessed once only after the Seen-Dev gate passed.

No AoA10 recalibration, checkpoint selection or hyperparameter selection was performed.

Static:

- SPS: `45.8426233`
- coverage: `0.7722953`
- mean u/v width: `0.01256728`

Adaptive:

- SPS: `49.0174144`
- gain: `+3.1747912`
- coverage: `0.8370198`
- coverage gain: `+0.0647244` = about `+6.47` percentage points
- mean u/v width: `0.01513140`

The AoA10 SPS gain is almost the same as the Seen-Dev gain.

## 7. Scientific conclusion

This direction is **validated for SOTA merge at the method level**.

The key evidence is not only the `+3.24` Seen-Dev gain. The same frozen head/calibration gives `+3.17` SPS on fully unseen AoA10 without recalibration.

This supports a real cross-AoA uncertainty-calibration benefit rather than a Seen-Dev-only interval-width trick.

Do not spend additional research budget on:
- wider SPS grids;
- different width caps;
- more head depth/width;
- longer head training;
- AoA10 tuning;
- another SPS architecture sweep.

## 8. SOTA merge rule

### Case A — final point predictor is exactly Clean Stage1 + Clean Stage2

Reuse directly:

- step-4500 head;
- head SHA above;
- calibration `0.0025 / 0.75 / 1.5 / 0.0075`.

No SPS retraining is required.

### Case B — final point predictor changes

Examples: a different backbone, different residual checkpoint/recipe, Pareto residual merge, or any change that materially changes final prediction errors.

Then:

- the **method is still GO**;
- do **not** assume the old head weights/calibration are still valid;
- freeze the final point predictor;
- train a matched uncertainty head using the exact validated recipe above;
- keep the same 300-combination calibration family and constrained-selection rule;
- select only on Seen-Dev;
- do not reopen SPS research;
- do not use AoA10 for retuning.

This is production adaptation of a validated method, not a new SPS research campaign.

## 9. Evidence

Primary evidence:

`docs/clean_baseline_final_campaign/results/20260925_exp3_constrained_recovery/`

Evidence commit:

`235e8255aa3bdb08b131a2c88bf9b2c54dee4056`

Global experiment registry:

`docs/track1_experiment_registry.md`

Campaign protocol:

`docs/clean_baseline_final_campaign/README.md`

Safety record:

- optimizer steps in recovery: 0
- locked-final/private: not accessed
- Codabench: not accessed
- full-data refit: not run
- submission/package: not run
