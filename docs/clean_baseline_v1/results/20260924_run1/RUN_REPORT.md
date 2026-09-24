# REALPDE Clean Baseline V1 — 2026-09-24 run

Status: `REVIEW_REQUIRED` (completed normally; review required by the frozen protocol)

## Execution provenance

- Execution commit: `65e4f0f1d029eea47c6ba96364ce889e22d437d8`, branch `main`, clean execution checkout.
- GPU: NVIDIA GeForce RTX 3090 Ti, 24 GB; `CUDA_VISIBLE_DEVICES=0`.
- Runtime: Python 3.11.10, Torch 2.4.0+cu121, CUDA runtime 12.1.
- Input checkpoint: `sim_real_cno.pth`, SHA-256 `82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61`.
- Data manifest SHA-256: `3612185c939f6c4cd4890afd3968d160a44e88d3e6f7733b8cb9ce71501d532c`.
- Canonical split SHA-256: `1a0935dd7fbc4860b0d2ca5d49e410dfe03191cf8177d9f8ced336cd69736e2a`.
- Split audit: 81 usable trajectories; Train 51, Seen Dev 12, AoA10 Holdout 18; no overlap; all available AoA10 trajectories held out; `7575_0.h5` excluded.
- Run directory: `/hy-tmp/runs/realpde_clean_baseline_v1_20260924`.
- Started 2026-09-24 01:25:07 and completed 06:28:08 (Asia/Shanghai), exit code 0. Stage 1 ran about 1:27, Stage 2 about 3:31, and final holdout evaluation about 4 minutes.
- OOM: no. Automatic environment recovery: none. Locked-final/private data: not accessed. Codabench: not accessed. Full-data refit and submission packaging: not started.
- Gate checks: 19 tests passed; requested `py_compile` and runner `--help` checks passed; split audit and strict checkpoint load passed.

## Seen-Dev results

Metrics are raw errors; lower is better. `point_score` is the mean of the v9 Rel-L2, TKE, and MVPE subscores and was the only checkpoint-selection metric.

| Stage/checkpoint | Update | Rel-L2 | TKE | MVPE | point_score |
|---|---:|---:|---:|---:|---:|
| Stage 1 best | 8,000 | 0.099606201 | 0.778031290 | 0.080289602 | 87.796616 |
| Stage 1 final | 8,723 | 0.099932298 | 0.792903721 | 0.080299616 | 87.663741 |
| Stage 2 step 0 | 0 | 0.099606197 | 0.778031262 | 0.080289606 | 87.796616 |
| Stage 2 best | 37,000 | 0.078363446 | 0.507731898 | 0.064407949 | 90.954326 |
| Stage 2 final | 38,400 | 0.078344605 | 0.508195901 | 0.064403135 | 90.949774 |

Stage 1 best mean-field and fluctuation errors were `0.059640504` and `0.995028794`; at final they were `0.059864625` and `0.998391330`.

## AoA10 Holdout results

Evaluated only after both training stages completed. These metrics are diagnostic and were not used for checkpoint selection.

| Checkpoint | Update | Rel-L2 | TKE | MVPE |
|---|---:|---:|---:|---:|
| Stage 2 step 0 | 0 | 0.096634969 | 0.714621186 | 0.100786760 |
| Stage 2 best | 37,000 | 0.085566938 | 0.523196816 | 0.101421528 |
| Stage 2 final | 38,400 | 0.085547566 | 0.523347020 | 0.101495080 |

## Checkpoint hashes

| Artifact | SHA-256 |
|---|---|
| Stage 1 best | `6aec4edb5e42746080e19baaafa2d6b719bb57befb1ff9035ece7889ec280036` |
| Stage 1 final | `4ffa498a9f9db682362f4f04ea090bc074e79d44f2439fe8a2658be96bcba858` |
| Stage 2 init | `a66cee1003f544ab392f8371b22a5c0022674f8bbf3a37141d19c85853cdb8ca` |
| Stage 2 best | `1536fe02e11fc0dc991a38cb1489d0be0752ec8149be7b9f1542a4cb81c2f975` |
| Stage 2 final | `f5b9352a7a7a1c939afa28fcee4c27df93c0cb508bbfea6dec1bbba2e54a4881` |

## Review evidence

The adjacent `evidence/` directory contains run and split manifests, the 82-row data inventory (filename, byte size, SHA-256), all Stage 1 evaluation metrics and horizon tables, all Stage 2 evaluation records and horizon milestones, sampling/window audits, sampled training logs, and per-trajectory/per-horizon AoA10 summaries.

Raw H5 training trajectories and checkpoint bytes remain on the GPU host and are deliberately excluded from Git. Their hashes and paths are recorded where needed for provenance. The evidence bundle excludes spatial-map archives and submission artifacts.
