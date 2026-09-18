# Submission Log

## Provenance note

The repository was forked from a teammate's earlier competition repository on 2026-08-29. Entries before the user's own 2026-09-03 P0-A + N2 submission are historical team results reconstructed from the inherited repository and recorded Codabench scores. Their exact submission ZIPs, wrapper code, checkpoint assets, and bounds settings may be incomplete or unavailable.

Therefore:
- treat the pre-2026-09-03 scores as valid online result anchors;
- do not infer an exact historical bounds formula or full experiment recipe from a ZIP filename alone;
- historical names such as `bounds_rel00` are clues, not reproducible configuration evidence;
- new SPS/bounds decisions should be rebuilt from the current frozen validation protocol and official scorer unless the original package/artifact is independently recovered and verified.

## Offline overnight execution — 2026-09-06

The A2 Multi-scale screen completed from the frozen 50/16 protocol but failed
its final gate (`NO_GO`). The queued Horizon Curriculum screen stopped in
preflight because adapted P0-A initialization differed from raw CNO by
`0.0002474784851074219`; it produced no training result. See the complete
[overnight review](sota迭代/reviews/idle_gpu_overnight_20260906/README.md).

## Codabench successful submissions

| ID | File | Date | Final | rel_l2 | TKE | MVPE | Time | SPS | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 897948 | `submission_cno_baseline.zip` | 2026-08-23 13:37 | 64.52290 | 88.581214 | 65.718217 | 81.677206 | 88.663276 | 5.192310 | Historical teammate submission; exact package provenance may be incomplete. |
| 900896 | `submission_cno_realft_4700_20260825.zip` | 2026-08-25 15:27 | 70.04628 | 93.258780 | 68.856072 | 92.016287 | 88.618744 | 10.578370 | Historical teammate submission; exact package provenance may be incomplete. |
| 903976 | `submission_cno_tke1200_bounds_rel00.zip` | 2026-08-27 15:44 | 75.58455 | 93.542062 | 70.934325 | 92.167656 | 87.236663 | 27.780536 | Historical teammate online anchor; exact bounds recipe is not assumed from filename alone. |
| 907047 | `8-29提交.zip` | 2026-08-29 13:47 | 74.48384 | 91.868766 | 66.666667 | 89.885887 | 91.959120 | 27.374631 | Historical teammate submission; UNet local-proxy candidate. |
| — | `submission.zip` (P0-A + N2 full, 15,300 updates) | 2026-09-03 | 71.153839 | 93.023539 | 78.355520 | 91.894417 | 88.430528 | 11.431650 | User-owned all-82-trajectory P0-A + N2 refit. TKE improved strongly versus the best prior CNO, but SPS fell sharply. See [detailed handoff](coordination/CHATGPT_HANDOFF_T1_P0A_N2_FULL15300_CODABENCH.md). |
| — | `full43260_abs0075_rel002/submission.zip` | 2026-09-04 | 76.149726 | 93.434384 | 77.588799 | 92.519561 | 86.998134 | 27.545059 | Previous SOTA. P0-A + N2 full@43,260 with explicit static SPS bounds `half_width = 0.0075 + 0.02*abs(prediction)`. Checkpoint SHA256 `50b692e236d5df9285a5cee976a51e3457a7eeed0f87d55b6568745077645d71`; ZIP SHA256 `f8a79ec1114b4e7f05edc9bc95c6810c5250ca27b5044ca387044c0671d9fc98`. |
| — | `full43260_adaptive_package_20260905/submission.zip` | 2026-09-05 | 76.694784 | 93.434384 | 77.588799 | 92.519563 | 87.066646 | 29.519724 | Previous online SOTA. Same full@43,260 backbone prediction with v5 base Adaptive Uncertainty Head@1400 and `half_width_uv = 0.0025 + sigma`; pressure half-width `0`. Package prediction parity vs previous backbone was `max_abs_diff=0.0`. ZIP SHA256 `3285ad3a424988ab35061337ca836c23b5f7db04773246167da3a9f8eaa2178a`. Relative to 2026-09-04 SOTA: Final `+0.545058`, SPS `+1.974665`, Time `+0.068512`, physical prediction scores essentially unchanged. |
| — | `sota_v2_adaptive_20260916/package_clean/submission.zip` | 2026-09-16 | 77.314446 | 93.816645 | 79.164203 | 93.411176 | 86.898836 | 30.319572 | SOTA-V2 full@53,582 (`Dense-All + P0-A + MF-CNO + N2 + vorticity + Stage-B`) with fresh Adaptive Uncertainty Head@1400. Clean package SHA256 `9cfc055c6232d2b0aef9f88f1cb3de2aae883b7cff7f02659ed8b9cf85b3ed55`; prediction parity `0.0`. See [SOTA-V2 review](sota迭代/reviews/sota_v2_adaptive_20260916/README.md). |\n| — | `sps_teammate_final_20260917/package_clean/submission.zip` | 2026-09-17 | **77.732796** | **93.816645** | **79.164203** | **93.411176** | **86.699342** | **31.961724** | **Current online SOTA.** Same frozen full@53,582 point predictor, with full-specific teammate35 uncertainty head @1600 and `floor=0.0025, mult=1.0`. Point scores unchanged; online gain came from SPS. ZIP SHA256 `cc236a4926d36568ea9eb2d4c770f08b0a1c058ec3e85f82256d70d3df8db685`. See [review](sota迭代/reviews/sps_teammate_final_20260917/README.md). |\n| — | `sps_teammate_exact_submit_20260918/package/submission.zip` | 2026-09-18 | 77.728857 | 93.816645 | 79.164203 | 93.411176 | 86.828909 | 31.899537 | Exact teammate uncertainty-recipe transfer on the same frozen full@53,582 predictor: 35ch, h32/b2, masked Gaussian NLL, seed41, 2000-update budget, eval every200, 28-row SPS grid; selected @1800 with `floor=0.0025, mult=1.0`. Point scores unchanged, but SPS `-0.062187` vs 2026-09-17, so no online gain. ZIP SHA256 `bb1a8804276767362e835ad4ae15e4c558153c2510efdea7f1d75ec8063a8e71`. See [review](sota迭代/reviews/sps_teammate_exact_submit_20260918/README.md). |

## Local packages prepared on 2026-08-29

These are inherited historical CNO candidates. Their filenames and recorded summaries are useful evidence, but package-level reproducibility is not assumed unless the original ZIP/checkpoint is recovered and verified.

| File | Role |
|---|---|
| `submission_cno_tke4100_bounds_abs0075_rel000_flat_20260829.zip` | Historical candidate; filename suggests abs=0.0075, rel=0.0. |
| `submission_cno_tke4100_bounds_abs0075_rel010_flat_20260829.zip` | Historical candidate; filename suggests abs=0.0075, rel=0.01. |
| `submission_cno_tke4100_bounds_abs0075_rel020_flat_20260829.zip` | Historical candidate; filename suggests abs=0.0075, rel=0.02. |

Additional CNO-only low-learning-rate continuation from `tke4100`:

| File | Role |
|---|---|
| `submission_cno_tke4100_cont600_balanced_abs0075_rel000_flat_20260829.zip` | Historical continuation candidate. |
| `submission_cno_tke4100_cont600_balanced_abs0075_rel010_flat_20260829.zip` | Historical continuation candidate. |
| `submission_cno_tke4100_cont600_balanced_abs0075_rel020_flat_20260829.zip` | Historical continuation candidate. |

Local continuation best summary:

```text
run: cno_tke4100_cont_lr5e7_balanced_20260829
best_iter: 600
rel_l2: 0.10214869
tke: 0.71349053
mvpe: 0.10154706
local_mean5_proxy: 79.09720
```

Single-model interpolation/extrapolation between original `tke4100` and `cont600`:

| File | Role |
|---|---|
| `submission_cno_tke4100_continterp_lam125_abs0075_rel000_flat_20260829.zip` | Historical interpolation candidate. |
| `submission_cno_tke4100_continterp_lam125_abs0075_rel010_flat_20260829.zip` | Historical interpolation candidate. |
| `submission_cno_tke4100_continterp_lam125_abs0075_rel020_flat_20260829.zip` | Historical interpolation candidate. |

Local interpolation best summary:

```text
run: cno_tke4100_to_cont600_weight_interp_scan_20260829
lambda_cont600: 1.25
rel_l2: 0.10238736
tke: 0.70794994
mvpe: 0.10147392
local_mean5_proxy: 79.16723
best_local_bound: abs=0.0075, rel=0.01
```

## Lessons learned

- Do not optimize a self-written final-score composite when the official leaderboard composite may differ or evolve.
- UNet postprocessing can look strong on released validation data but generalize poorly on hidden data.
- CNO currently has better hidden physical scores, especially Rel-L2, TKE, and MVPE.
- P0-A + N2 full demonstrates a strong online TKE gain, but SPS must be protected independently; physical subscore gains alone do not guarantee a better final score.
- Pre-fork historical bounds information is weak evidence unless the original artifact is recovered; rebuild current SPS calibration from the frozen validation protocol and official scorer.
- The 2026-09-04 submission confirmed explicit static SPS calibration could recover SPS from `11.431650` to `27.545059` while preserving strong physical scores.
- The 2026-09-05 submission confirms learned adaptive uncertainty transfers online: with backbone predictions unchanged, SPS improved from `27.545059` to `29.519724` and Final from `76.149726` to `76.694784`. Adaptive uncertainty is therefore a validated SOTA component, not only an offline calibration signal.
- The 2026-09-16 SOTA-V2 submission confirms the integrated predictive recipe transfers online: versus the previous online SOTA, Rel-L2, TKE and MVPE all improve, SPS also rises, and Final reaches `77.314446`. The gain is therefore not an uncertainty-only effect.\n- The 2026-09-17 teammate35 full-specific head raised SPS to `31.961724` and Final to `77.732796` with identical point scores, establishing it as the current online SOTA.\n- The 2026-09-18 exact teammate uncertainty-recipe transfer did not improve online SPS (`31.899537` vs `31.961724`) despite closer alignment on masked NLL, seed, 2000-step selection and calibration. This closes further SPS-head-only micro-tuning unless new evidence appears; the remaining teammate gap should be investigated together with the pre-correction base / residual-corrector / final-prediction structure.
- Prefer simple, auditable SOTA merges; use online results to validate combined recipes without requiring perfect offline evidence first.
