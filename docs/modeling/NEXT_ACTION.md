# NEXT_ACTION

## Goal
Run one final analysis-only merge calibration for the fully trained residual corrector. Determine whether deterministic energy preservation can retain most of the corrector's Rel-L2/MVPE gains while removing the remaining TKE degradation.

## Frozen assets
- Backbone: SOTA-V2 `@32500`, frozen.
- Corrector: TMR-02 `@30000`, frozen.
- Dev: frozen Dev16 / 659 canonical windows.
- Alpha: `1.0` only.
- No training.

## Variants
Evaluate exactly four variants:
1. `base`
2. `corrected`
3. `window_energy`: keep corrected temporal mean; restore backbone total Future20 u/v fluctuation energy per window with one positive scalar.
4. `spatial_tke_map`: keep corrected temporal mean; restore backbone Future20 TKE magnitude per spatial point with one positive scalar shared across u/v and Future20.

No parameter sweep or extra projection variant is allowed.

## Execution
1. Use `tools/realpde_residual_corrector_projection.py` and `tests/test_residual_corrector_projection.py`.
2. Run focused tests, compile, and diff-check.
3. Replay the four frozen variants once on Dev16.
4. Produce official raw Rel-L2/TKE/MVPE and the existing trajectory × horizon diagnostics.
5. Verify projection invariants with `projection_audit.json`.
6. Commit lightweight evidence and push to `main`.

## Required evidence
- `physical_metrics.csv`
- `trajectory_metrics_long.csv`
- `by_horizon.csv` (80 rows)
- `by_trajectory_horizon.csv` (1280 rows)
- `horizon_trajectory_stability.csv` (60 rows)
- `trajectory_anatomy.json`
- `projection_audit.json`
- `run_metadata.json`
- `summary.json`
- `status.json`
- concise `README.md`

## Constraints
- Training: NOT PERFORMED.
- No alpha, loss, architecture, or projection-parameter sweep.
- No joint backbone tuning.
- No SPS, uncertainty, full-data, locked-final/private, package, or Codabench.
- Per-horizon TKE/MVPE remain diagnostics, not official per-frame scores.
- Do not auto-start another Corrector experiment.

## Stop
Return `REVIEW_REQUIRED`. ChatGPT/Sol makes the final binary decision: promote one merge candidate to full-data, or close the Residual Corrector direction.
