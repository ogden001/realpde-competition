# Historical Future20 Frame-wise Audit

Status: `REVIEW_REQUIRED`.

This package contains 11 inventory records. Nine records have complete provenance and replayed on the frozen 50/16 development split: 16 trajectories, 659 ordered stride-20 windows, and Future20 horizons 1 through 20. Two records have `INCOMPLETE_PROVENANCE` and were not replayed.

## Replayed checkpoints

| Experiment | Checkpoint | SHA-256 |
|---|---|---|
| LOSS-E0-best | `/home/chyfuture/realpde_runs/loss_opt_v9_20260901_run1/long_E0_s20260901/model_best.pth` | `5d02c8da5bcbcbcc47917b0021b1007b2036931a3a51483772f7607deeb4aff6` |
| LOSS-E1-best | `/home/chyfuture/realpde_runs/loss_opt_v9_20260901_run1/long_E1_s20260901/model_best.pth` | `8b25200d157401e14eee39b8cf8f62379cc9d4c5afc0165ac51432962d14457d` |
| LOSS-E2-best | `/home/chyfuture/realpde_runs/loss_opt_v9_20260901_run1/long_E2_s20260901/model_best.pth` | `469add6d7bdd387ba635184e2f628b0ec4b1bcd6a519f1b2f2264bc972fa5c99` |
| LOSS-E3-best | `/home/chyfuture/realpde_runs/loss_opt_v9_20260901_run1/long_E3_s20260901/model_best.pth` | `d8b0f8bbb0de7a4966f2bf11d1669a65677161e19c126749495a24e487088742` |
| P0A-N2-2H-last | `/home/chyfuture/realpde_runs/b1_p0a_n2_20260901/b1_train_2h/model_latest.pth` | `56291d3f2afcf5a569ebea72ef2b68b9e95f16cd399ba48e293248e1ee4a1fb6` |
| MF01-1500 | `/home/chyfuture/realpde_runs/mf01_s20260904/model_update_01500.pth` | `488a8118f489789d385ec90e02856ef6a8482d6fa75c252e2e5d2d1f50e72226` |
| MF-C02-C0-1500 | `/home/chyfuture/realpde_runs/mf_energy_campaign02/c0/model_update_01500.pth` | `9f9f338582a653e47510d1ea577c56049bedb0ac1eebe87b7911e619538564bb` |
| RW-00-fixed-7500 | `/home/chyfuture/realpde_runs/rw00_rw01_random_phase_20260907_v8/output/RW-00_fixed/model_update_07500.pth` | `d18009c0efb48cabfd27f833e9ae927aa9bcd8e9e9cbdace6f4d14b766b76e49` |
| RW-01-random-phase-3000 | `/home/chyfuture/realpde_runs/rw00_rw01_random_phase_20260907_v8/output/RW-01_random_phase/model_update_03000.pth` | `bbc681f75b701e68b949e38bba106a56f3442822760ce7afb9db25cb7d017b6a` |

## Output files

- `experiment_inventory.csv`
- `frame_metrics.csv` — 180 rows (9 replays × 20 horizons).
- `per_window_frame_metrics.csv` — 118,620 rows (9 replays × 659 windows × 20 horizons).
- `aggregate_metrics.csv` — 9 rows.

## Not replayed

- `POINT-V0-direct`: `INCOMPLETE_PROVENANCE`; no independently reconstructable full checkpoint record is retained.
- `HYBRID-A1-rerun`: `INCOMPLETE_PROVENANCE`; the retained checkpoint and model-loader provenance do not match.

No locked-final data, model training, loss change, optimizer update, or Codabench access occurred.
