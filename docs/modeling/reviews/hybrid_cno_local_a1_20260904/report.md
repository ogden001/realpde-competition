# A1 Evaluation / Error Anatomy

## Verified facts

- A1 overall relative deltas vs matched Direct are {'rel_l2': -0.35966253977229196, 'tke': 0.6612003826572548, 'mvpe': -2.494454006333056} (negative error delta is improvement).
- Trajectory wins are {'rel_l2': 2, 'tke': 7, 'mvpe': 1} out of 16.
- Top good cases: ['3750_0.h5', '11400_10.h5', '11400_15.h5']; top bad cases: ['8850_20.h5', '25425_15.h5', '24150_10.h5'].
- Results use only the frozen 16 Dev trajectories and official v9 raw metrics; no custom final score is constructed.

## Overall

| metric | matched Direct | A1 | delta % | wins |
|---|---:|---:|---:|---:|
| rel_l2 | 0.17582881 | 0.17646120 | -0.360 | 2/16 |
| tke | 0.59464884 | 0.59071702 | +0.661 | 7/16 |
| mvpe | 0.15163144 | 0.15541382 | -2.494 | 1/16 |

Metric-vs-update values are in `update_curve.csv`; horizon values are in `horizon_metrics.csv`.

## Parameters and runtime

- Direct CNO parameters: `7,989,843`; Local branch parameters: `914`.
- Dev inference timing on 659 windows, GPU forward path only: matched Direct `59.41 ms/window`; A1 `58.13 ms/window`.
- Timing excludes HDF5/DataLoader I/O and scorer execution.

## Case linkage

See `trajectory_case_table.csv`; descriptors are runtime-safe input-side quantities joined from the existing Dataset Profile. No model change was selected from an individual case.

## Spatial / branch evidence

Representative maps are `case_good_*.png` and `case_bad_*.png`. Local residual statistics are in `local_residual_windows.csv` and `local_residual_summary.npz`.

## Mechanism hypothesis

- Case linkage is descriptive: input-side fluctuation, temporal variation, gradients, vorticity/strain and Train-tail labels are not causal controls.
- A local-branch mechanism is supported only where residual magnitude and error-reduction maps spatially overlap; this does not prove the branch specifically models local physics.

## Level 2 trigger

- TKE clearly worsened: NO; severe trajectory collapse: NO; deeper diagnosis: NO.

## Architecture verdict

`WEAK_SIGNAL_PARKED`

The Local + Global result is interpreted only through the three official raw metrics, trajectory evidence, horizon behavior and maps; it is not converted into a custom final score.
