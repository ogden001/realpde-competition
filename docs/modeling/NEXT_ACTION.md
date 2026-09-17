# NEXT_ACTION

## Goal
Run one complete residual-corrector training campaign on the frozen SOTA-V2 `@32500` backbone to determine whether the strong TMR-01 Rel-L2 signal survives after proper multi-epoch training without the observed TKE trade-off.

## Frozen recipe
- Backbone: SOTA-V2 `@32500`, frozen.
- Corrector: existing 42-channel `ResidualCorrector3D`, `hidden=64`, `blocks=2`, `max_delta=0.04`.
- Features / loss / optimizer family / seed: identical to TMR-01.
- Batch size: `8`.
- Corrector updates: `30000`.
- LR: `1e-4`, AdamW, weight decay `1e-5`.
- Scheduler: one `CosineAnnealingLR(T_max=30000)` across the full campaign.
- Milestones: `6000 / 12000 / 20000 / 30000`.
- Evaluation alpha: `1.0` only.
- Dense-All Train50: `40488` windows; total exposure is about `5.93` dense epochs.

## Execution
1. Use `tools/realpde_residual_corrector_longtrain.py` and `tests/test_residual_corrector_longtrain.py`.
2. Run focused tests and compile checks before GPU execution.
3. Train through all `30000` updates without Dev evaluation or intermediate decision-making.
4. After training completes, replay `base + u6000 + u12000 + u20000 + u30000` once on frozen Dev16.
5. Produce overall physical metrics plus the existing TMR-01A trajectory × horizon diagnostics.
6. Commit lightweight evidence and push to `main`.

## Required evidence
- `training_summary.json`
- `physical_metrics.csv`
- `trajectory_metrics_long.csv`
- `by_horizon.csv` (`20 × 5 = 100` rows)
- `by_trajectory_horizon.csv` (`16 × 20 × 5 = 1600` rows)
- `horizon_trajectory_stability.csv` (`20 × 4 = 80` rows)
- `trajectory_anatomy.json`
- `replay_summary.json`
- `run_metadata.json`
- `summary.json`
- `status.json`
- concise `README.md`

## Constraints
- No backbone training or joint fine-tuning.
- No alpha sweep, loss-weight sweep, architecture sweep, or post-processing projection.
- No Dev access before the complete 30k training finishes.
- No SPS, uncertainty, full-data, locked-final/private, package, or Codabench.
- TKE/MVPE per-horizon quantities remain diagnostics, not official per-frame scores.
- No automatic follow-on experiment or SOTA merge.

## Stop
After all evidence is committed and pushed, return `REVIEW_REQUIRED`. ChatGPT/Sol owns the final scientific interpretation and merge decision.
