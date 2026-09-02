# ChatGPT/Sol Handoff — Point-V0 Clean Screening

Read this file first for the completed Point-V0 result. The implementation is in `tools/realpde_point_v0.py`; the frozen protocol and experiment registry are in `docs/track1_experiment_registry.md`.

## Decision

`STOP_PURE_POINT`. No Point-V1 was launched.

This was a standalone **Point MLP with normalized grid positional coordinates and no spatial-field context**, not a strict causal comparison with CNO.

## Protocol

- 82 PIV trajectories; fixed trajectory-disjoint 50 train / 16 dev / 16 locked-final manifest; manifest SHA-256 `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`.
- Seed `20260901`; batch 8 complete windows; 7,500 optimizer updates per learned variant; AdamW `lr=1e-5`; last@7500 only.
- Identity/raw velocity normalization. Point input is normalized grid indices plus the point's 20-frame raw `u/v` history; residual is raw future-minus-last-frame velocity. Output is `(N,20,32,64,3)` with `p=0`.
- Loss is raw-space `MSE + 0.05*TKE`; full field is restored before TKE. Official v9 scorer SHA-256 `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`; loss implementation SHA-256 `6ded951b3aea29152ca1f75b82bdc009e79b1233ac5b31beb1bdd636c13070ce`.
- `sim_real_ft`, CFD, Re/AoA, neighborhood, global features and locked-final were not used. No Codabench submission.

## Dev results (official v9 raw errors; lower is better)

| model | Rel-L2 | TKE | MVPE |
|---|---:|---:|---:|
| PERSIST | 0.135578 | 1.000000 | 0.132707 |
| POINT-DIRECT | 0.226415 | 0.930389 | 0.282945 |
| POINT-RESIDUAL | 0.147084 | 0.917402 | 0.132756 |

POINT-RESIDUAL vs PERSIST: Rel-L2 `-8.49%`, TKE improvement `8.26%`, MVPE `-0.04%`; it failed the required ≥5% Rel-L2 and MVPE improvements. POINT-DIRECT also failed.

## Stability and engineering findings

- Paired trajectory-macro bootstrap covers all 16 dev trajectories; it is evidence, not a veto. Horizon metrics use fixed window-micro aggregation. Spatial map has 2,040/2,048 valid pixels.
- Each learned variant took about 1,468–1,472 seconds for 7,500 updates. CPU/HDF5 loading dominated wall time while GPU utilization was low; optimize the data pipeline before the next point-modeling experiment.

## Review artifacts

The complete non-Git review package is at `artifacts/point_v0_v2_s20260901_review/` locally and in `artifacts/point_v0_v2_s20260901_review.zip`. It contains `summary.json`, `report.md`, `metrics.csv`, `paired_trajectory_bootstrap.csv`, `horizon_metrics.csv`, `spatial_error_e2.csv`, figures, run metadata, and the TKE replay check. It intentionally excludes checkpoints and prediction arrays.

## Runner protocol reminder

Long CPU/GPU tasks are detached Runner work. Codex should implement and smoke-test, start the detached job, confirm it once, and stop; later status/result recovery is user-triggered.
