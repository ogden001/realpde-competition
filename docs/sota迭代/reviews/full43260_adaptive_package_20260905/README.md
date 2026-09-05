# full@43260 + v5 base adaptive package

Status: `SUBMITTED / NEW_ONLINE_SOTA`

This candidate uses the frozen full@43260 backbone together with the already-trained v5 `base_head_state_dict@1400`. No backbone retraining, recalibration, corrector, locked-final/private-test access, or runtime benchmark was performed before packaging.

Execution source commit: `dcae3bc` (`Add bounded full43260 adaptive package builder`).

## Package evidence

- Backbone checkpoint SHA256: `50b692e236d5df9285a5cee976a51e3457a7eeed0f87d55b6568745077645d71`
- Adaptive package ZIP SHA256: `3285ad3a424988ab35061337ca836c23b5f7db04773246167da3a9f8eaa2178a`
- Prediction parity vs direct full@43260 backbone: `max_abs_diff=0.0`
- Bounds: `half_width_uv = 0.0025 + sigma`; pressure half-width `0`
- Clean-room smoke: PASS
- No `ResidualCorrector3D` path or weights in the package

Evidence files:

- `build_report.json` and `build_summary.md`
- `smoke_summary.md`
- `SHA256_PROVENANCE.md`
- `build.review.log`, `smoke.review.log`, `smoke_retry.review.log`

## Codabench result

Submitted on `2026-09-05`.

| Metric | Previous static-bounds SOTA | Adaptive uncertainty | Delta |
|---|---:|---:|---:|
| Final | `76.149726` | **`76.694784`** | **`+0.545058`** |
| Rel-L2 | `93.434384` | `93.434384` | `0.000000` |
| TKE | `77.588799` | `77.588799` | `0.000000` |
| MVPE | `92.519561` | `92.519563` | `+0.000002` |
| Time | `86.998134` | `87.066646` | `+0.068512` |
| SPS | `27.545059` | **`29.519724`** | **`+1.974665`** |

Conclusion: `KEEP / NEW_ONLINE_SOTA`.

Because package prediction parity was exact and the online Rel-L2/TKE/MVPE scores stayed effectively unchanged, this submission provides clean online evidence that the learned Adaptive Uncertainty Head improves SPS independently of the backbone prediction. The adaptive uncertainty component is promoted to the current SOTA recipe.

The generated ZIP remains outside Git at `artifacts/full43260_adaptive_package_20260905/submission.zip`.
