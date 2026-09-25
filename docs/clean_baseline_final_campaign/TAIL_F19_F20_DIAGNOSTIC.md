# Exp1 F19/F20 Zero-Training Diagnostic

Status: `REVIEW_REQUIRED`

Purpose: determine whether the Exp1 F19/F20 cliff is primarily caused by:

1. Strong Backbone tail-error growth;
2. Residual correction effectiveness collapse;
3. residual direction/magnitude mismatch;
4. or a mixture of the above.

This is an evaluation-only diagnostic. It is not a new training experiment.

## Hard constraints

- Clean Seen-Dev12 only.
- Train51 / Seen-Dev12 manifest only; the script rejects an 18-trajectory Holdout dev set.
- Exact Exp1 Strong Backbone checkpoint SHA256:
  `d340effd68031aba1cd2bc610c676878fb9105dd7f81b9ce15767280e665a5c0`
- Exact Exp1 Residual checkpoint SHA256:
  `d2b4ddd0df6064d053ec918c2671a93b7d9c2a40a15afe854bb3a5badf1438fc`
- No optimizer step, no training, no checkpoint selection.
- No AoA10 field access, locked-final/private, Codabench, full-data refit or submission packaging.

## Run

Use the existing Exp1 Train51/Seen-Dev12 manifest, for example:

```bash
python -u -B tools/diagnose_strong_backbone_tail.py \
  --real-root "$REAL_ROOT" \
  --split-manifest "$EXP1_RUN/train_dev_manifest.json" \
  --backbone-checkpoint "$EXP1_RUN/exp1_strong_backbone_clean/strong_backbone/checkpoints/model_best.pth" \
  --residual-checkpoint "$EXP1_RUN/exp1_strong_backbone_clean/residual_22k/model_best.pth" \
  --model-root "$KIT_ROOT" \
  --out-dir "$OUT" \
  --workers 2 \
  --require-cuda
```

Before execution:

```bash
python -m pytest -q tests/test_strong_backbone_tail_diagnostic.py
python -m py_compile tools/diagnose_strong_backbone_tail.py
git diff --check
```

## Outputs

- `backbone_by_horizon.csv`: Strong Backbone F1..F20.
- `final_by_horizon.csv`: Strong Backbone + Residual F1..F20.
- `tail_diagnostic_by_horizon.csv`: direct backbone/final comparison plus:
  - Rel/RMSE correction gain;
  - residual `delta` RMS;
  - true residual-target RMS;
  - `delta / target` magnitude ratio;
  - `delta_target_cosine`;
  - least-squares `alpha_star`;
  - per-window `help_fraction`;
  - actual SSE gain;
  - oracle SSE gain along the already-learned residual direction.
- `tail_by_trajectory.csv`: F18/F19/F20 correction gain for every Seen-Dev trajectory.
- `tail_summary.json`: F18/F19/F20 summary and diagnostic flags.
- `run_manifest.json`: provenance and safety fields.

## How to read the result

### Backbone problem

If Backbone itself shows a sharp F19/F20 growth while correction gain remains roughly stable, prioritize temporal-output-boundary / MF-representation analysis.

### Residual problem

If Backbone growth is smooth but correction gain or `delta_target_cosine` collapses at F19/F20, the main issue is Residual losing correction effectiveness at long horizon.

### Magnitude problem

If cosine stays high but `alpha_star` moves far above/below 1, the residual direction is still useful but its magnitude is wrong at the tail.

### Direction problem

If `delta_target_cosine` collapses, especially toward zero or negative values, the residual is no longer pointing toward the true correction at the tail.

Do not start a new training run from this diagnostic automatically. Return evidence to Sol for review.
