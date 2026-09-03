# Submission Log

## Codabench successful submissions

| ID | File | Date | Final | rel_l2 | TKE | MVPE | Time | SPS | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 897948 | `submission_cno_baseline.zip` | 2026-08-23 13:37 | 64.52290 | 88.581214 | 65.718217 | 81.677206 | 88.663276 | 5.192310 | Official CNO baseline package. |
| 900896 | `submission_cno_realft_4700_20260825.zip` | 2026-08-25 15:27 | 70.04628 | 93.258780 | 68.856072 | 92.016287 | 88.618744 | 10.578370 | Real-finetuned CNO. |
| 903976 | `submission_cno_tke1200_bounds_rel00.zip` | 2026-08-27 15:44 | 75.58455 | 93.542062 | 70.934325 | 92.167656 | 87.236663 | 27.780536 | Current best known Codabench result. |
| 907047 | `8-29提交.zip` | 2026-08-29 13:47 | 74.48384 | 91.868766 | 66.666667 | 89.885887 | 91.959120 | 27.374631 | UNet local-proxy candidate; hidden physical scores worse than CNO. |
| — | `submission.zip` (P0-A + N2 full, 15,300 updates) | 2026-09-03 | 71.153839 | 93.023539 | 78.355520 | 91.894417 | 88.430528 | 11.431650 | All-82-trajectory competition refit. TKE improved strongly versus the best prior CNO, but SPS fell sharply and final score is lower. See [detailed handoff](coordination/CHATGPT_HANDOFF_T1_P0A_N2_FULL15300_CODABENCH.md). |

## Local packages prepared on 2026-08-29

These are flat-clean CNO `tke4100` single-model candidates. They were smoke-tested locally on the remote machine and downloaded to the local workspace.

| File | Role |
|---|---|
| `submission_cno_tke4100_bounds_abs0075_rel000_flat_20260829.zip` | Recommended next one-shot candidate; closest to the known good CNO `rel00` route. |
| `submission_cno_tke4100_bounds_abs0075_rel010_flat_20260829.zip` | Local SPS proxy best among scanned simple bounds. |
| `submission_cno_tke4100_bounds_abs0075_rel020_flat_20260829.zip` | Matches earlier CNO package style with relative bound. |

Additional CNO-only low-learning-rate continuation from `tke4100`:

| File | Role |
|---|---|
| `submission_cno_tke4100_cont600_balanced_abs0075_rel000_flat_20260829.zip` | More aggressive CNO candidate; local TKE/MVPE improved, Rel-L2 slightly worse. |
| `submission_cno_tke4100_cont600_balanced_abs0075_rel010_flat_20260829.zip` | Same checkpoint with local simple-bound variant. |
| `submission_cno_tke4100_cont600_balanced_abs0075_rel020_flat_20260829.zip` | Same checkpoint with relative bound variant. |

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
| `submission_cno_tke4100_continterp_lam125_abs0075_rel000_flat_20260829.zip` | Aggressive CNO-only candidate, using lambda=1.25 and rel=0.0 bounds. |
| `submission_cno_tke4100_continterp_lam125_abs0075_rel010_flat_20260829.zip` | Local best among the CNO-only candidates; recommended if explicitly trying to beat current score. |
| `submission_cno_tke4100_continterp_lam125_abs0075_rel020_flat_20260829.zip` | Same checkpoint with relative bound variant. |

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

- Do not optimize the self-written equal-weight final estimate. Codabench states the `final_score` combination is not published.
- UNet postprocessing can look strong on released validation data but generalize poorly on hidden data.
- CNO currently has better hidden physical scores, especially Rel-L2, TKE, and MVPE.
- P0-A + N2 full demonstrates a strong online TKE gain, but SPS must be protected independently; physical subscore gains alone do not guarantee a better final score.
- Prefer simple CNO packages until a new candidate improves the official final score or demonstrates a better protected multi-metric trade-off.
