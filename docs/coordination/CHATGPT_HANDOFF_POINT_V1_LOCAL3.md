# Point-V1 LOCAL3 handoff

Reference: `T1-ID-POINT-V1-LOCAL3-20260902`  
Execution state: **COMPLETED / STOP_LOCAL3_EARLY**

## Scope

Clean Track 1 experiment only. Train split was used for pipeline/LR/training; 16 dev trajectories were used once for the pre-registered Phase 3A gate. Locked-final was not accessed and Codabench was not called. Point-V0 was not retrained. Phase 3B was not entered.

## Phase 1 — frozen pipeline

`POINT_RAM_PIPELINE_V1 = B3_PACKED`.

Formal B3_PACKED training-path: `306.79 windows/s`, `26.08 ms/step`, data-wait ratio `5.77%`, cache build `31.17 s`, cache `1.04 GB`, process-tree peak RSS `2.06 GB`, host MemAvailable `57.2 GB`. Exact DATA_EQUIVALENCE passed on the fixed seeded shuffled train-window sequence. B3_RAM coarse was `305.03 windows/s` with `9.44%` data-wait ratio and `1.65 GB` process-tree RSS; its cache was `695 MB`.

## Phase 2 — LR sanity

Both 500-update runs used identical initialization, shuffled train-window order, pipeline and loss. Both were finite with no NaN/Inf.

| LR | final total loss | final MSE | final TKE | mean total loss |
|---:|---:|---:|---:|---:|
| 1e-5 | 0.0707491 | 0.0005456 | 1.40407 | 0.545632 |
| 1e-4 | 0.0419802 | 0.0003763 | 0.832077 | 0.102587 |

Frozen LR: `1e-4`.

## Phase 3A — LOCAL3 screening

Model: `3×3` replicate-padded local u/v history (`3*3*20*2+2=362` inputs), MLP `362→256→256→256→128→40`, raw-space residual output, loss `MSE + 0.05*TKE`, 1500 updates, `last@1500`.

| model | Rel-L2 | TKE | MVPE |
|---|---:|---:|---:|
| PERSIST | 0.135578 | 1.000000 | 0.132707 |
| LOCAL3 @1500 | 0.155252 | 0.832650 | 0.140822 |

Relative to PERSIST: Rel-L2 `-14.51%`, TKE `+16.74%`, MVPE `-6.12%`. The screening gate requires Rel-L2 > 0%, MVPE > 0%, and TKE degradation ≤ 10%; it failed on Rel-L2 and MVPE, so the runner wrote `STOP_LOCAL3_EARLY` and did not extend to 7500.

## Interpretation boundary

This is evidence that the first 1500-step LOCAL3 run did not beat Persistence under the frozen gate. It is not evidence that 3×3 context is useless after 7500 updates, because the protocol explicitly stopped at 1500. It also does not justify changing architecture, loss, normalization, or adding features in this experiment family.

## Evidence paths

- Remote run: `/home/chyfuture/realpde_runs/point_v1_local3_phase3_retry3_s20260901`
- Phase 1/2 source run: `/home/chyfuture/realpde_runs/point_v1_local3_s20260901`
- Retry3 gate: `phase3_local3/screening_gate.json`
- Dev evaluations: `phase3_local3/dev@1500_local3/` and `phase3_local3/dev@1500_persist/`
- Checkpoint: `phase3_local3/last@1500.pt`
- Runner: [`tools/realpde_point_v1_local3_runner.py`](../../tools/realpde_point_v1_local3_runner.py)

