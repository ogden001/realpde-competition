# H1 Frozen CLEAN CNO + LOCAL3 Point Residual Handoff

Reference: `T1-ID-HYBRID-CNO-POINT-H1-S20260902`

## Result

The bounded Phase-A run completed `1500/1500` updates and stopped at the
preregistered gate: `STOP_HYBRID_POINT_H1_EARLY`.

| model | Rel-L2 | TKE error | MVPE |
|---|---:|---:|---:|
| FROZEN_CNO | 0.19082105 | 0.64406884 | 0.14425756 |
| FROZEN_CNO + LOCAL3_POINT_HEAD | 0.14110152 | 0.70690584 | 0.10270503 |

Candidate relative to the same frozen CNO reference:

- Rel-L2 improvement: **26.056%**
- MVPE improvement: **28.804%**
- TKE error improvement: **-9.756%**, i.e. TKE error worsened by **9.756%**

Rel-L2 and MVPE passed their positive-improvement conditions. The TKE
protection condition failed because the error worsening exceeded the allowed
5%, so no continuation to 7500 updates and no full diagnostics were run.

## Frozen protocol

- FE-00 CLEAN CNO-only backbone frozen; no `sim_real_ft`, CNO retraining, or
  joint training.
- Zero-initialized Point head input 402 and output 40; architecture
  `402→256→256→256→128→40` with GELU. The CNO pressure channel is copied
  unchanged; only uv receives the learned correction.
- Absolute corrected-field uv MSE only; AdamW `lr=1e-4`, `weight_decay=0`,
  batch 8, seed `20260901`, B3_PACKED, fixed manifest, 50 train / 16 dev.
- The comparison is incremental value relative to this specific frozen FE-00
  reference, not a general CNO architecture comparison.

## Evidence and provenance

- Remote run: `/home/chyfuture/realpde_runs/hybrid_cno_point_h1_s20260902`
- Remote artifacts: `/home/chyfuture/realpde_runs/hybrid_cno_point_h1_s20260902/artifacts/`
- Local review artifacts (checkpoint excluded from Git):
  `artifacts/hybrid_cno_point_h1_s20260902_review/`
- Frozen manifest SHA-256:
  `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`
- FE-00 checkpoint SHA-256:
  `499ec748cc5db1b7f3ad24029a464e921ad31c3d383b69626ec540eba392903e`
- `last@1500` checkpoint SHA-256:
  `fb6735bff296cc53f028b894b74691697f7475f5bbfda9ae8ee0dcd70d1e3bd2`
- Executed runner SHA-256:
  `9b3d3382496208496ddfa9ac382134095400f957fb4390bfc50904ebe7735b15`
- Source implementation commit: `bb41c3f`

Exact runner command inside the bounded Docker job:

```bash
python -u /source/realpde_hybrid_cno_point_h1_runner.py \
  --data-root /data/p0ab_real_h5_20260830 \
  --manifest /runs/job/manifest.json --kit-root /kit \
  --checkpoint /backbone/last.pth --out-dir /runs/job/artifacts --device cuda
```

## Boundary and next decision

- `locked-final accessed: NO`
- `Codabench: NO`
- `H2: not authorized`
- `LOCAL5: not authorized`

The evidence supports only this bounded conclusion: on the frozen FE-00 CNO,
the LOCAL3 Point correction substantially improves Rel-L2 and MVPE at 1500
updates, but violates the protected TKE criterion. ChatGPT/Sol review is
required before any new design or loss change.
