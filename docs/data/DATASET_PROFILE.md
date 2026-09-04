# Track 1 Dataset Profile

Status: `COMPLETE_INITIAL_PROFILE`

Trajectory-level, model-independent reference for the frozen Track 1 PIV
split. It uses only 50 Train and 16 Dev trajectories; the 16 locked-final
trajectories were not read.

## Provenance and protocol

- Manifest: `artifacts/loss_optimization_v9_20260901_run1/evidence/manifests/id_seed20260901.json`
- Manifest SHA-256: `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`
- Archive: `data/train_real.tar.gz`; archive SHA-256 is in `artifacts/dataset_profile_20260904/profile_summary.json`.
- Analysis script: `tools/profile_track1_dataset.py`.
- Analysis code commit: `e52ff5bbd2a0b77b555983f3c1d30c6a80e1897e` (script and generated profile protocol).
- Split: 50 Train / 16 Dev / 16 locked-final; locked-final access: `false`.
- Raw trajectory shape: `[frames, 64, 128]`, channels `u, v`; coordinates `x, y`; metadata `aoa, re, t` retained.
- Window protocol: non-overlapping `T_in=20`, `T_future=20`, `stride=20`, valid when `start + 40 <= frames`.
- Aggregation: arithmetic mean of valid window descriptors per trajectory; 2,052 Train and 659 Dev windows.

Complete outputs: [trajectory_descriptors.csv](../../../artifacts/dataset_profile_20260904/trajectory_descriptors.csv) and [profile_summary.json](../../../artifacts/dataset_profile_20260904/profile_summary.json).

## Exact input-side descriptors

All quantities are computed per Past20 input window and averaged per trajectory.
Spatial derivatives use median `x/y` spacing and centered finite differences.
`fluctuation_rms = sqrt(mean((u-mean(u))² + (v-mean(v))²) / 2)`.
`delta_*` describe frame-to-frame speed-vector differences. `grad_mag_mean`
is the velocity-gradient Frobenius norm; `vorticity_abs_mean` is
`abs(dv/dx - du/dy)`; `strain_mag_mean` is
`sqrt((du/dx-dv/dy)² + (du/dy+dv/dx)²)`. `high_energy_area_ratio` is the
spatial fraction whose temporal local fluctuation energy exceeds 1.5 times
the spatial median. Spectrum ratios are non-DC temporal FFT energy in the
low/mid/high thirds of available frequency bins.

Descriptors: `u_mean`, `u_std`, `v_mean`, `v_std`, `speed_mean`, `speed_std`,
`speed_p95`, `delta_mean`, `delta_std`, `fluctuation_rms`, `grad_mag_mean`,
`vorticity_abs_mean`, `strain_mag_mean`, `high_energy_area_ratio`,
`spectrum_low_ratio`, `spectrum_mid_ratio`, `spectrum_high_ratio`.

## Train / Dev distribution summary

Values are `min / p10 / median / p90 / p95 / max`; full `p25 / p75` values
are in `profile_summary.json`. Every descriptor has overlapping raw Train/Dev
ranges; this does not imply identical joint distributions.

| Descriptor | Train | Dev |
|---|---:|---:|
| u_mean | 0.0447 / 0.0601 / 0.1498 / 0.2370 / 0.2589 / 0.2818 | 0.0396 / 0.0686 / 0.1435 / 0.2367 / 0.2579 / 0.2920 |
| u_std | 0.0214 / 0.0288 / 0.0781 / 0.1353 / 0.1436 / 0.1644 | 0.0216 / 0.0352 / 0.0692 / 0.1403 / 0.1458 / 0.1520 |
| v_mean | -0.0102 / -0.0060 / -0.0023 / 0.0010 / 0.0017 / 0.0031 | -0.0098 / -0.0057 / -0.0024 / 0.0005 / 0.0012 / 0.0027 |
| v_std | 0.0069 / 0.0079 / 0.0095 / 0.0116 / 0.0121 / 0.0134 | 0.0070 / 0.0081 / 0.0093 / 0.0121 / 0.0128 / 0.0140 |
| speed_mean | 0.0452 / 0.0603 / 0.1502 / 0.2372 / 0.2593 / 0.2821 | 0.0416 / 0.0703 / 0.1439 / 0.2371 / 0.2585 / 0.2923 |
| speed_std | 0.0209 / 0.0285 / 0.0782 / 0.1354 / 0.1436 / 0.1644 | 0.0212 / 0.0349 / 0.0694 / 0.1404 / 0.1458 / 0.1520 |
| speed_p95 | 0.0567 / 0.0766 / 0.1924 / 0.3018 / 0.3278 / 0.3570 | 0.0525 / 0.0892 / 0.1837 / 0.3017 / 0.3265 / 0.3700 |
| delta_mean | 0.0018 / 0.0022 / 0.0028 / 0.0042 / 0.0047 / 0.0056 | 0.0018 / 0.0024 / 0.0026 / 0.0044 / 0.0048 / 0.0053 |
| delta_std | 0.0022 / 0.0028 / 0.0038 / 0.0065 / 0.0073 / 0.0090 | 0.0022 / 0.0032 / 0.0035 / 0.0071 / 0.0078 / 0.0086 |
| fluctuation_rms | 0.0155 / 0.0211 / 0.0565 / 0.0962 / 0.1026 / 0.1190 | 0.0156 / 0.0255 / 0.0495 / 0.1000 / 0.1037 / 0.1086 |
| grad_mag_mean | 1.5676 / 2.2116 / 4.3175 / 7.4152 / 8.2113 / 9.5333 | 1.5105 / 2.4262 / 4.2702 / 7.6887 / 8.1355 / 9.0269 |
| vorticity_abs_mean | 1.1062 / 1.5885 / 3.2767 / 5.7077 / 6.3612 / 7.3123 | 1.0894 / 1.7647 / 3.1771 / 5.9834 / 6.2690 / 6.9035 |
| strain_mag_mean | 1.3005 / 1.9776 / 3.8836 / 6.7500 / 7.3642 / 8.6175 | 1.2596 / 2.1584 / 3.7752 / 7.0125 / 7.3960 / 8.2001 |
| high_energy_area_ratio | 0.3646 / 0.3894 / 0.4235 / 0.4464 / 0.4474 / 0.4495 | 0.3635 / 0.3953 / 0.4223 / 0.4491 / 0.4494 / 0.4500 |
| spectrum_low_ratio | 0.8707 / 0.8933 / 0.9165 / 0.9460 / 0.9525 / 0.9604 | 0.8765 / 0.8980 / 0.9200 / 0.9503 / 0.9548 / 0.9585 |
| spectrum_mid_ratio | 0.0311 / 0.0390 / 0.0528 / 0.0747 / 0.0821 / 0.0963 | 0.0307 / 0.0387 / 0.0512 / 0.0730 / 0.0788 / 0.0890 |
| spectrum_high_ratio | 0.0195 / 0.0252 / 0.0328 / 0.0474 / 0.0504 / 0.0527 | 0.0225 / 0.0268 / 0.0309 / 0.0462 / 0.0484 / 0.0485 |

## Dev coverage / OOD-like assessment

Labels use Train-only statistics. A descriptor is an exceedance when outside
the Train `[p5, p95]` interval. Inputs are standardized with the Train mean
and population standard deviation; `nearest_train_distance` is Euclidean
distance to the closest standardized Train trajectory. `OOD_LIKE` means at
least 3 exceedances or distance >4; `BOUNDARY` means at least 1 exceedance or
distance >2; otherwise `IN_DISTRIBUTION`. No Dev target/error is used.

Dev labels: 9 `IN_DISTRIBUTION`, 3 `BOUNDARY`, 4 `OOD_LIKE`.

| Dev trajectory | Label | NN distance | Exceedances |
|---|---|---:|---:|
| 10125_0 | BOUNDARY | 1.149 | 1 |
| 3750_0 | OOD_LIKE | 0.860 | 13 |
| 26700_0 | OOD_LIKE | 0.989 | 6 |
| 13950_5 | IN_DISTRIBUTION | 0.681 | 0 |
| 20325_5 | IN_DISTRIBUTION | 0.617 | 0 |
| 8850_5 | BOUNDARY | 0.904 | 1 |
| 8850_10 | IN_DISTRIBUTION | 0.672 | 0 |
| 11400_10 | IN_DISTRIBUTION | 0.582 | 0 |
| 24150_10 | IN_DISTRIBUTION | 0.726 | 0 |
| 16500_10 | IN_DISTRIBUTION | 1.129 | 0 |
| 22875_15 | IN_DISTRIBUTION | 0.813 | 0 |
| 11400_15 | IN_DISTRIBUTION | 0.775 | 0 |
| 25425_15 | OOD_LIKE | 0.945 | 9 |
| 20325_20 | BOUNDARY | 0.683 | 1 |
| 8850_20 | IN_DISTRIBUTION | 0.743 | 0 |
| 3750_20 | OOD_LIKE | 1.005 | 7 |

All NN distances are below 1.15, so OOD-like labels are marginal/joint tail
flags rather than isolated points far from Train. The recurring separation is
in velocity fluctuation, temporal delta and gradient/vorticity/strain,
especially for low-Re and some high-Re cases. These are distribution
observations only and do not imply model failure or justify model selection.

## Future20 target descriptors (analysis-only)

`future_u_mean`, `future_v_mean`, `future_speed_mean`,
`future_fluctuation_rms`, and `future_tke` are in the CSV solely for
retrospective analysis. They were excluded from labels, distances, split
decisions, and model inputs, and are not valid inference-time features.

## Reuse and refresh

Future experiments should load this profile before interpreting bad cases and
report both metric/horizon behavior and the case's position in this
Train-derived distribution. Refresh only when the manifest, window protocol,
input definition, or a materially important descriptor definition changes.
