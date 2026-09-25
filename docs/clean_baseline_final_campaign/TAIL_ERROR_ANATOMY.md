# Exp1 F18/F19/F20 Error Anatomy

Status: `REVIEW_REQUIRED`

This diagnostic answers one question only:

> Why does the Exp1 Strong Backbone show a sharp F19/F20 error cliff?

It evaluates four zero-training mechanisms together on the exact Clean Seen-Dev12 windows:

1. **Spatial phase / position drift**
   - global non-wrapping integer shifts `dx,dy in [-2,2]`;
   - per-window shift oracle to distinguish coherent drift from heterogeneous drift.

2. **Temporal phase lag**
   - compare Pred F18/F19/F20 against GT at h-2, h-1, h;
   - test one-step extrapolation `p_h + gamma*(p_h-p_{h-1})` for
     `gamma in {0,0.25,0.5,0.75,1.0}`.

3. **Fluctuation amplitude mismatch**
   - compute least-squares scale of the horizon fluctuation around the model's
     original Future20 temporal mean.

4. **Residual magnitude under-correction**
   - reuse the existing residual direction;
   - apply per-horizon least-squares `alpha*` to F18/F19/F20 only;
   - recompute whole-Future20 Rel-L2/TKE/MVPE to expose benefit and side effects.

All oracle values are Seen-Dev diagnostics only. They are not submission parameters.

## Hard constraints

- Clean Train51 / Seen-Dev12 manifest only.
- Exactly 491 Seen-Dev windows.
- Exact Exp1 checkpoints:
  - Backbone SHA256:
    `d340effd68031aba1cd2bc610c676878fb9105dd7f81b9ce15767280e665a5c0`
  - Residual SHA256:
    `d2b4ddd0df6064d053ec918c2671a93b7d9c2a40a15afe854bb3a5badf1438fc`
- Zero optimizer steps.
- No AoA10 field access.
- No locked-final/private.
- No Codabench.
- No full-data refit or submission packaging.
- Do not start a training experiment automatically from these results.

## Outputs

- `backbone_error_anatomy.json`
- `final_error_anatomy.json`
- `mechanism_summary.json`
- `backbone_spatial_shift.csv`
- `backbone_temporal_extrapolation.csv`
- `backbone_amplitude_scaling.csv`
- `residual_geometry_f18_f20.csv`
- `run_manifest.json`

Each of the three zero-training transforms is also applied to the whole predictor
for F18/F19/F20 only, and whole-Future20 Rel-L2/TKE/MVPE are recomputed.

## Interpretation

### Coherent spatial phase drift

Strong evidence if:
- global shift gives a material F20 SSE reduction;
- the per-window oracle is not dramatically larger than the global-shift gain;
- the same shift direction is common across many windows.

This supports a small bounded tail warp rather than a general residual network.

### Heterogeneous spatial phase error

If the per-window shift oracle is much stronger than the single global shift,
the problem is phase-like but not correctable by one fixed translation.
A learned tiny warp head may be justified, but a fixed shift is not.

### Temporal lag / under-advanced dynamics

Evidence if:
- Pred F20 is closer to GT F19 than GT F20;
- and/or positive temporal extrapolation gamma materially lowers F20 SSE.

This supports a bounded temporal extrapolation / phase head.

### Amplitude mismatch

Evidence if:
- amplitude scaling alone materially reduces F20 SSE;
- with scale* meaningfully different from 1.

This supports a constrained tail amplitude correction.

### Residual magnitude under-correction

The diagnostic applies the existing residual direction with per-horizon alpha*.
If whole-Future20 Rel improves while TKE/MVPE remain protected, a bounded
tail residual-gain candidate may be worth a single validation run.

### None of the above

If spatial shift, temporal extrapolation, amplitude scaling and residual alpha
all have weak upside, the remaining hypothesis is a real temporal-boundary /
representation limitation of the Strong Backbone. Only then should a short
Stage-C tail-focused fine-tune be considered.
