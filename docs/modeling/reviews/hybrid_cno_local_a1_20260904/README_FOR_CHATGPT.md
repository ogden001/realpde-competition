# A1 Review Bundle

Experiment: `T1-ID-HYBRID-CNO-LOCAL-A1-S20260904`

This bundle contains the completed Dev-only Evaluation / Error Anatomy evidence for:

`Direct CNO global predictor + lightweight Local residual branch`

with `prediction = global_cno(x) + local_residual(x)`.

## Included

- `report.md`: final report with Verified facts, Mechanism hypothesis and Architecture verdict.
- `summary.json`: official v9 overall raw metrics, deltas, wins and verdict.
- `trajectory_case_table.csv`: all 16 Dev trajectory comparisons and input-side descriptor linkage.
- `horizon_metrics.csv`: t+1 ... t+20 analysis.
- `update_curve.csv`: saved update convergence curve; values are trajectory-macro summaries.
- `local_residual_windows.csv` and `local_residual_summary.npz`: residual magnitude/RMS/relative-to-global statistics.
- `case_*.png`: representative good/bad spatial maps.
- `run_metadata.json`: architecture, protocol, hashes, parameter counts and split provenance.
- `runtime.json`: matched Direct/A1 GPU forward timing.

## Key boundary

Only the frozen 16 Dev trajectories were used for this review. Locked final, private test and Codabench were not accessed. No custom final score was constructed.

The full A1 prediction artifact is intentionally excluded from this ChatGPT bundle because it is approximately 478 MB. It remains at:

`/home/chyfuture/realpde_runs/hybrid_cno_local_a1_20260904/eval_03000/predictions.npz`

The A1 checkpoint remains at:

`/home/chyfuture/realpde_runs/hybrid_cno_local_a1_20260904/model_update_03000.pth`

