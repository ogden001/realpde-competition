# CNO Future20 Cliff Data Collection

## 1. Checkpoint / SHA

| model | checkpoint | SHA-256 |
|---|---|---|
| Plain PIV-CNO | `long_E0_s20260901/model_best.pth` | `5d02c8da5bcbcbcc47917b0021b1007b2036931a3a51483772f7607deeb4aff6` |
| P0-A CNO | `model_last.pth` @ update 30900 | `e3d5faaf1a71e121b09077dd7dd7d0456a617e2916b8a671986f412fb54f6388` |

Manifest SHA-256: `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`.

## 2. Data and shape checks

Prediction and target shape were both `(659, 20, 32, 64, 3)`. The replay used 16 dev trajectories, 659 fixed windows, and 20 Future20 horizons. Every u/v target window was checked against raw `trajectory[start+20:start+40]`; the recorded frame indices are t+1=`start+20`, t+19=`start+38`, and t+20=`start+39`.

## 3. Executed command

```text
PYTHONPATH=/deps python /deps/horizon_cliff_closeout.py --data-root /data/p0ab_real_h5_20260830 --manifest /data/id_seed20260901.json --kit-root /kit --plain-checkpoint /runs/loss_opt_v9_20260901_run1/long_E0_s20260901/model_best.pth --p0a-checkpoint /runs/p0a_n2_simreal_validation_20260903/continuation_10300_to30900/run/model_last.pth --out-dir /out/results --batch-size 8
```

Executed in `realpde-pytorch-h5py:0831`; inference and statistics only.

## 4. Output files

- `horizon_metrics_closeout.csv`: model/horizon aggregate Rel-L2, MVPE, scorer singleton-horizon TKE, velocity squared error, and speed moments.
- `per_window_horizon_metrics.csv`: all 659 windows × 20 horizons × 2 models.
- `tail_contribution.csv`: per-horizon squared-error sums and fractions.
- `spatial_residuals.npz`: 32×64 mean absolute and mean squared velocity-error maps for t18/t19/t20.
- `compute_evidence.json`: shapes, counts, SHA values, Full20/First18 metrics, and tail fractions.

For the per-horizon TKE column, the supplied scorer's `tke_rel_l2_per_sample` is called with a one-frame time axis; its temporal fluctuation is therefore zero, and the recorded `tke_error` is `0.0`. Full20 and First18 TKE use the supplied scorer unchanged on their respective multi-frame slices.

## 5. Core numeric tables

| model | t18 Rel-L2 | t19 Rel-L2 | t20 Rel-L2 | SE19+SE20 / SEtotal |
|---|---:|---:|---:|---:|
| Plain PIV-CNO | 0.17068414 | 0.18490623 | 0.24056078 | 0.18691608 |
| P0-A CNO | 0.11934058 | 0.15308243 | 0.17540927 | 0.21071215 |

| model | range | Rel-L2 | TKE | MVPE |
|---|---|---:|---:|---:|
| Plain PIV-CNO | Full20 | 0.16892298 | 0.53847533 | 0.13614585 |
| Plain PIV-CNO | First18 | 0.16267683 | 0.61084205 | 0.13686641 |
| P0-A CNO | Full20 | 0.11284460 | 0.50010282 | 0.08728255 |
| P0-A CNO | First18 | 0.10539774 | 0.54849982 | 0.08783761 |

## 6. Previous t18/t19/t20 check

Passed exactly for both models against the prior recorded values: Plain `0.17068414 / 0.18490623 / 0.24056078`; P0-A `0.11934058 / 0.15308243 / 0.17540927`.

## 7. Calculation failures or missing data

None.
