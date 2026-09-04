# A1 Evaluation / Error Anatomy

## Verified facts

- A1 overall relative deltas vs matched Direct are {'rel_l2': -0.28570285682517277, 'tke': 0.49807789304360867, 'mvpe': -2.2771053008634} (negative error delta is improvement).
- Trajectory wins are {'rel_l2': 2, 'tke': 10, 'mvpe': 1} out of 16.
- Top good cases: ['3750_0.h5', '11400_10.h5', '11400_15.h5']; top bad cases: ['8850_20.h5', '20325_5.h5', '26700_0.h5'].
- Results use only the frozen 16 Dev trajectories and official v9 raw metrics; no custom final score is constructed.

## Overall

| metric | matched Direct | A1 | delta % | wins |
|---|---:|---:|---:|---:|
| rel_l2 | 0.17582881 | 0.17633116 | -0.286 | 2/16 |
| tke | 0.59464884 | 0.59168702 | +0.498 | 10/16 |
| mvpe | 0.15163144 | 0.15508425 | -2.277 | 1/16 |

Metric-vs-update values are in `update_curve.csv`; horizon values are in `horizon_metrics.csv`.

## Case linkage

See `trajectory_case_table.csv`; descriptors are runtime-safe input-side quantities joined from the existing Dataset Profile. No model change was selected from an individual case.

## Spatial / branch evidence

Representative maps are `case_good_*.png` and `case_bad_*.png`. Local residual statistics are in `local_residual_windows.csv` and `local_residual_summary.npz`.

## Mechanism hypothesis

- Case linkage is descriptive: input-side fluctuation, temporal variation, gradients, vorticity/strain and Train-tail labels are not causal controls.
- A local-branch mechanism is supported only where residual magnitude and error-reduction maps spatially overlap; this does not prove the branch specifically models local physics.

## Level 2 trigger

- TKE clearly worsened: NO; severe trajectory collapse: NO; deeper diagnosis: NO.

## Review status

`REVIEW_REQUIRED`

This report provides evidence only. It does not make a KEEP/PARK/STOP research decision and does not construct a custom final score.
