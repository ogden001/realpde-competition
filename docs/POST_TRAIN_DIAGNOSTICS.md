# Default Post-Train Diagnostics

Status: DEFAULT_REQUIRED for RealPDE point-prediction experiments.

Every point-prediction training experiment must run the standard diagnostics on
its frozen dev protocol after the primary final checkpoint is evaluated. These
diagnostics are explanatory evidence, not extra competition metrics and not a
license to access locked-final data.

The standard bundle is written under an experiment's diagnostics/ directory:

- by_horizon.csv: Future1-Future20 Rel-L2/RMSE plus residual correction behavior.
- by_trajectory.csv: one row per trajectory, including Rel-L2, TKE diagnostic,
  mean-field error, fluctuation error and energy ratio.
- by_trajectory_horizon.csv: per-window, per-horizon evidence for case x time analysis.
- spatial_maps.npz: u/v/velocity RMSE, mean-field error, fluctuation error,
  TKE absolute-error and target-TKE maps.
- summary.json: mean/fluctuation decomposition and, when a base prediction is
  available, residual before/after statistics such as correction-help fraction.

For residual-corrector experiments, tools/colleague_80pt/residual_multi.py invokes
this bundle automatically at the end of every training run.

For historical checkpoints or experiments completed before this rule, use
tools/colleague_80pt/analyze_checkpoints.py to replay multiple checkpoints on one
frozen dev split and produce comparison_summary.csv plus comparison_by_horizon.csv.

New point-prediction runners should call
post_train_diagnostics.write_post_train_diagnostics() by default. Do not make
diagnostics opt-in unless a run is an explicitly documented smoke test.

User-facing reports must use descriptive experiment names. Internal IDs such as
R0/R1 are allowed in files, but reports should say, for example,
"original residual loss / TKE weight 0.06" rather than only "R0".
