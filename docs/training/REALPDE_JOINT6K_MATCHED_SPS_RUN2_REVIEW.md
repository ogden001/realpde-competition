# REALPDE Joint@6k Matched SPS — Run 2 Review Evidence

Status: `REVIEW_REQUIRED`

## Execution

- Run: `/hy-tmp/realpde_runs/realpde_sota_merge_sps_joint6k_20260926_run2`
- Execution commit: `ffc704398c6e513f2dc45a28f71aa0522a333750`
- Hardware: RTX 4090
- Completion: 5,000 uncertainty-head optimizer updates; training wall time 3,038.19 s
- Point model: frozen; optimizer updates 0; point prediction parity max abs 0.0
- Backbone SHA256: `ca9d3efbcbe10a5b0190875c6bc749386002678e33b86d7ecda0fba55c306d86`
- Corrector SHA256: `a98b4eb048e193a6337d267fe54ead8645d77ddcfaad18321f0f6c07c76ebec8`
- Data root: prevalidated symlink-only adapter at `/hy-tmp/realpde_data/realpde_sps_joint6k_20260926_resolved_inputs`; manifest roles are Train51 / Seen-Dev12 / AoA10 Holdout18.
- DataLoader workers: 4; calibration workers: 12; parallel calibration backend: shared memory (`shm`).
- Seen-Dev gate: `GO`; selected head step: 5,000.
- Locked-final/private/Codabench accessed: no.

## Seen-Dev selection

Static: SPS 45.317705, coverage 0.789239, mean width 0.016219.

Adaptive at selected step 5,000: SPS 49.341143, gain +4.023438 vs static, coverage 0.864485, coverage gain +0.075246, mean width 0.019393, width ratio 1.195670, point parity max abs 0.0. Selected calibration: floor 0.0025, mult_u 1.0, mult_v 1.5, rel 0.0075. All recorded Seen-Dev gate checks passed.

## Constrained-best Seen-Dev curve

See [`REALPDE_JOINT6K_MATCHED_SPS_RUN2_CURVE.csv`](REALPDE_JOINT6K_MATCHED_SPS_RUN2_CURVE.csv) for steps 500–5,000, including constrained-best SPS, gain vs static, width, width ratio, coverage, and selected calibration.

## AoA10 one-shot audit

The run summary records AoA10 accessed after the Seen-Dev `GO`, with `aoa10_used_for_selection=false` and `aoa10_recalibrated=false`.

| Calibration | SPS | Coverage | Mean width uv |
| --- | ---: | ---: | ---: |
| Static | 43.680301 | 0.782807 | 0.016292 |
| Adaptive (Seen-Dev selected) | 47.270162 | 0.862619 | 0.019946 |

Adaptive gain vs static: +3.589861 SPS; coverage gain: +0.079812.

## Small artifact references

- `head_best.pth` SHA256: `b5c26f1a6ed929948717fb9cf68afb0ad558fdc20a23d92e97217d291ec0a445`
- Remote `summary.json`, `training_progress.json`, `static_calibration_grid.json`, and `calibration_grid_00500.json` through `calibration_grid_05000.json` remain in the run directory. Checkpoint files are not included in this evidence commit.
