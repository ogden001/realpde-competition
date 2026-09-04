# Submission Log

## Provenance note

The repository was forked from a teammate's earlier competition repository on 2026-08-29. Entries before the user's own 2026-09-03 P0-A + N2 submission are historical team results reconstructed from the inherited repository and recorded Codabench scores. Their exact submission ZIPs, wrapper code, checkpoint assets, and bounds settings may be incomplete or unavailable.

Therefore:
- treat the pre-2026-09-03 scores as valid online result anchors;
- do not infer an exact historical bounds formula or full experiment recipe from a ZIP filename alone;
- historical names such as `bounds_rel00` are clues, not reproducible configuration evidence;
- new SPS/bounds decisions should be rebuilt from the current frozen validation protocol and official scorer unless the original package/artifact is independently recovered and verified.

## Codabench successful submissions

| ID | File | Date | Final | rel_l2 | TKE | MVPE | Time | SPS | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 897948 | `submission_cno_baseline.zip` | 2026-08-23 13:37 | 64.52290 | 88.581214 | 65.718217 | 81.677206 | 88.663276 | 5.192310 | Historical teammate submission; exact package provenance may be incomplete. |
| 900896 | `submission_cno_realft_4700_20260825.zip` | 2026-08-25 15:27 | 70.04628 | 93.258780 | 68.856072 | 92.016287 | 88.618744 | 10.578370 | Historical teammate submission; exact package provenance may be incomplete. |
| 903976 | `submission_cno_tke1200_bounds_rel00.zip` | 2026-08-27 15:44 | 75.58455 | 93.542062 | 70.934325 | 92.167656 | 87.236663 | 27.780536 | Current best known online score; historical teammate package, so exact bounds recipe is not assumed from filename alone. |
| 907047 | `8-29提交.zip` | 2026-08-29 13:47 | 74.48384 | 91.868766 | 66.666667 | 89.885887 | 91.959120 | 27.374631 | Historical teammate submission; UNet local-proxy candidate. |
| — | `submission.zip` (P0-A + N2 full, 15,300 updates) | 2026-09-03 | 71.153839 | 93.023539 | 78.355520 | 91.894417 | 88.430528 | 11.431650 | User-owned all-82-trajectory P0-A + N2 refit. TKE improved strongly versus the best prior CNO, but SPS fell sharply and final score is lower. See [detailed handoff](coordination/CHATGPT_HANDOFF_T1_P0A_N2_FULL15300_CODABENCH.md). |

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
- Prefer simple CNO packages until a new candidate improves the official final score or demonstrates a better protected multi-metric trade-off.
