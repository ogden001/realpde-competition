# NEXT_ACTION

## Goal
Complete the missing `trajectory × horizon` evidence for TMR-01 on the frozen SOTA-V2 `@32500` backbone and its existing teammate residual corrector.

## Tasks
1. Use `tools/realpde_tmr01a_trajectory_horizon_audit.py` and `tests/test_tmr01a_trajectory_horizon_audit.py`.
2. Replay Dev16 only for fixed `alpha = 0 / 0.5 / 1.0`; do not retrain anything.
3. Verify aggregate parity with TMR-01, then produce `by_horizon.csv`, `by_trajectory_horizon.csv`, and `horizon_trajectory_stability.csv`.
4. Commit lightweight evidence under `docs/modeling/reviews/tmr01a_trajectory_horizon_audit_20260917/` and push to `main`.

## Constraints
- Frozen 50/16 manifest; Dev16 canonical 659 windows only.
- Frozen SOTA-V2 `@32500` backbone and frozen TMR-01 corrector hashes enforced by the runner.
- No training, parameter sweep, TMR-02, SPS, uncertainty, full-data, locked-final/private, package, or Codabench.
- TKE/MVPE per-horizon quantities are diagnostic decompositions, not official per-frame scores.
- No automatic GO/NO-GO. Final scientific interpretation remains with ChatGPT/Sol.

## Deliverables
- `aggregate_parity.json`
- `by_horizon.csv` (20 rows)
- `by_trajectory_horizon.csv` (16 × 20 × 3 = 960 rows)
- `horizon_trajectory_stability.csv` (20 × 2 = 40 rows)
- `run_metadata.json`
- concise `README.md`

## Stop
After all replay evidence is committed and pushed to `origin/main`, return `REVIEW_REQUIRED`. Do not start the next experiment.
