# NEXT_ACTION - Execute CLEAN_BASELINE_FINAL_CAMPAIGN

Status: `READY_FOR_EXECUTION / REVIEW_REQUIRED`

Run exactly three frozen clean experiments and push lightweight review evidence to `origin/main` for Sol review.

Tasks: sync clean `main`; verify push permission, GPU/data/checkpoint assets and focused tests; run Strong Backbone Clean and Pareto-TKE Residual Clean on separate GPUs if available or serially otherwise; run Residual-aware SPS on frozen clean Stage2 best; generate `CAMPAIGN_REVIEW.md`; archive only lightweight JSON/CSV/MD/review logs; commit/pull-rebase/push `origin/main`; stop at `REVIEW_REQUIRED`.

Constraints: `main` only; no science changes; no parameter sweeps beyond frozen beta/calibration grids; no full-data, package, Codabench, locked-final/private; no automatic new experiment after any GO/NO_GO.

Deliverables include run configs, checkpoint hashes/provenance, training/eval milestones, sampling/init parity evidence, by-horizon/by-trajectory diagnostics from existing tools, Exp1/Exp2 gates and selected beta, SPS calibration/holdout evidence, `CAMPAIGN_REVIEW.md`, and the remote result commit SHA.
