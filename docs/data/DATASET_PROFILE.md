# Track 1 Dataset Profile

Status: `COMPLETE_FULL_SPLIT_AUDIT`

Trajectory-level, model-independent reference for the frozen Track 1 PIV split.
The initial 50 Train / 16 Dev profile remains preserved for comparison; the
full audit below adds all 16 locked-final trajectories using input-side data only.

## Provenance and protocol

- Manifest: `artifacts/loss_optimization_v9_20260901_run1/evidence/manifests/id_seed20260901.json`
- Manifest SHA-256: `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`
- Archive: `data/train_real.tar.gz`; archive SHA-256 is in `artifacts/dataset_profile_20260904/profile_summary.json`.
- Analysis script: `tools/profile_track1_dataset.py`.
- Analysis code commit: `e52ff5bbd2a0b77b555983f3c1d30c6a80e1897e` (script and generated profile protocol).
- Split: 50 Train / 16 Dev / 16 locked-final; full-audit locked-final access: `input-side only`.
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

## Full 82-Trajectory Split Audit

Conclusion: `SPLIT_OK`

This is a distribution-only audit. It did not run a model, read or slice
Future20 for locked-final, calculate Rel-L2/TKE/MVPE, select a checkpoint, or
modify the manifest. Although an H5 member contains a complete trajectory,
the audit opens only the `u/v` dataset slices for Past20 windows and reads
`x/y`, `aoa`, and `re` metadata. No target descriptor is constructed for any
split, and the generated CSV has no Future20, prediction, or metric fields.

The full artifact is [dataset_profile_20260904_full](../../../artifacts/dataset_profile_20260904_full/), with the manifest and archive hashes recorded in `profile_summary.json`. Counts are 50 / 16 / 16 trajectories and 2,052 / 659 / 672 valid windows for Train / Dev / Final. AoA counts remain the frozen `11/8/10/11/10`, `3/3/4/3/3`, and `3/3/4/3/3` across 0°/5°/10°/15°/20°.

### Basic and input-descriptor distributions

Each cell is `p10 / median / p90 / p95`; descriptor values are trajectory means over Past20 windows. The 17 input descriptors and the original `fluctuation_rms` definition are unchanged.

| Variable | Train | Dev | Final |
|---|---:|---:|---:|
| Re | 5028/13977/24204/24906.9 | 6306.5/12698.5/24843/25801.8 | 6945.5/15255.5/24204/26761 |
| frames | 868/868/868/868 | 868/868/868/868 | 868/868/868/868 |
| windows | 42/42/42/42 | 42/42/42/42 | 42/42/42/42 |
| u_mean | 0.06009/0.14977/0.23696/0.25890 | 0.06860/0.14352/0.23667/0.25792 | 0.07299/0.15496/0.22083/0.23674 |
| u_std | 0.02882/0.07807/0.13527/0.14357 | 0.03517/0.06917/0.14034/0.14581 | 0.03319/0.08849/0.13615/0.15185 |
| v_mean | -0.00352/-0.00048/0.00124/0.00204 | -0.00163/-0.00043/0.00098/0.00112 | -0.00192/-0.00044/0.00143/0.00180 |
| v_std | 0.00574/0.01157/0.02168/0.03187 | 0.00556/0.01158/0.02154/0.02425 | 0.00698/0.01208/0.02544/0.02849 |
| speed_mean | 0.06035/0.15021/0.23715/0.25933 | 0.07034/0.14385/0.23708/0.25849 | 0.07333/0.15751/0.22201/0.23751 |
| speed_std | 0.02811/0.07625/0.13508/0.14322 | 0.03422/0.06883/0.14010/0.14572 | 0.03317/0.08798/0.13602/0.15169 |
| speed_p95 | 0.08052/0.21517/0.36160/0.38340 | 0.09894/0.19362/0.37791/0.39072 | 0.09701/0.24229/0.36716/0.40085 |
| delta_mean | 0.00167/0.00358/0.00846/0.01001 | 0.00167/0.00348/0.00912/0.01033 | 0.00172/0.00398/0.00858/0.01109 |
| delta_std | 0.00178/0.00464/0.01132/0.01341 | 0.00174/0.00407/0.01224/0.01297 | 0.00183/0.00538/0.01230/0.01516 |
| fluctuation_rms | 0.02110/0.05655/0.09625/0.10258 | 0.02553/0.04953/0.10000/0.10366 | 0.02398/0.06364/0.09707/0.10818 |
| grad_mag_mean | 2.21158/4.31754/7.41523/8.21127 | 2.42616/4.27024/7.68868/8.13552 | 2.25017/4.60691/7.48998/8.44306 |
| vorticity_abs_mean | 1.58851/3.27674/5.70767/6.36116 | 1.76468/3.17709/5.98339/6.26902 | 1.63091/3.54472/5.81454/6.53268 |
| strain_mag_mean | 2.10413/4.12246/7.12525/7.87826 | 2.30325/4.01948/7.32636/7.86643 | 2.12723/4.42301/7.15889/8.21668 |
| high_energy_area_ratio | 0.38943/0.42350/0.44638/0.44744 | 0.38577/0.42887/0.44200/0.44394 | 0.39089/0.42911/0.44592/0.44752 |
| spectrum_low_ratio | 0.87685/0.91995/0.93975/0.94480 | 0.88816/0.92657/0.93329/0.93641 | 0.91189/0.93193/0.94603/0.95064 |
| spectrum_mid_ratio | 0.03516/0.04646/0.07279/0.08115 | 0.03956/0.04314/0.06555/0.07213 | 0.03166/0.04007/0.05176/0.05656 |
| spectrum_high_ratio | 0.02518/0.03276/0.04743/0.05038 | 0.02685/0.03089/0.04622/0.04841 | 0.02185/0.02762/0.03635/0.03985 |

Main differences are modest and overlapping: Final has a higher Re median (15,255.5 versus 13,977 Train and 12,698.5 Dev); higher central fluctuation, velocity spread, gradient/vorticity/strain descriptors; and higher low-band but lower mid/high-band spectrum ratios. Final contains no short-frame case (all 868 frames), while Train has one 282-frame case and Dev has one 607-frame case. These are tail/coverage differences, not a separate physical region.

### Joint distribution and Train coverage

PCA uses all 82 trajectory input vectors after one common all-82 standardization. The [PCA plot](../../../artifacts/dataset_profile_20260904_full/pca_split.png) shows Train spanning the occupied region, with Dev and Final interleaved inside it rather than forming a split-exclusive island.

Using Train-only standardization and Train p5/p95 descriptor bounds, Dev has 9 `IN_DISTRIBUTION`, 3 `BOUNDARY`, and 4 `OOD_LIKE` trajectories; Final has 11, 1, and 4 respectively. Dev nearest-Train distances range from 0.582 to 1.149 (median 0.813); Final ranges from 0 to 1.775 (median 0.880). The four Final OOD-like flags are descriptor-tail combinations, not large nearest-Train gaps; no Final trajectory exceeds the distance-2 boundary. Therefore Final has no clear Train coverage gap, and the comparable Dev/Final tail fractions do not support a split-bias conclusion.

The original 50/16 input descriptor rows were compared against the full-audit rerun: maximum per-trajectory absolute difference was below `2.4e-7` for both Train and Dev, confirming that the original statistical convention was preserved.

## Future20 target descriptors (historical initial profile only)

`future_u_mean`, `future_v_mean`, `future_speed_mean`,
`future_fluctuation_rms`, and `future_tke` are in the CSV solely for
retrospective analysis. They were excluded from labels, distances, split
decisions, and model inputs, and are not valid inference-time features.

## Reuse and refresh

Future experiments should load this profile before interpreting bad cases and
report both metric/horizon behavior and the case's position in this
Train-derived distribution. Refresh only when the manifest, window protocol,
input definition, or a materially important descriptor definition changes.
