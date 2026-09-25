# REALPDE Experiment 1 — Strong Backbone Clean

Status: `REVIEW_REQUIRED`

## Result

The frozen Seen-Dev gate passed. Mean raw-error change versus the frozen Clean Baseline residual @22k was **-1.2407%**; all three metrics improved, and TKE improved. The allowed AoA10 Holdout18 diagnostic was run once after the GO gate. No further training or submission action was started.

| Metric | Clean Baseline @22k | Candidate | Relative change |
|---|---:|---:|---:|
| Rel-L2 | 0.07969018 | 0.07953662 | -0.1927% |
| TKE | 0.50590988 | 0.49601257 | -1.9563% |
| MVPE | 0.06544801 | 0.06441840 | -1.5732% |
| Mean raw-error change | — | — | **-1.2407%** |

Gate: **GO** (mean <= -1%, at least 2/3 improve, no metric worsens by >1%, TKE does not worsen).

AoA10 Holdout18 diagnostic: Rel-L2 `0.08554702`, TKE `0.51909405`, MVPE `0.09412055`.

## Selected checkpoints

- Strong Backbone: update `32500`; Seen-Dev Rel-L2 `0.10122272`, TKE `0.47076225`, MVPE `0.07291063`, point score `90.87068`.
  - Remote path: `/hy-tmp/realpde_runs/strong_backbone_clean_20260924_run1/exp1_strong_backbone_clean/strong_backbone/checkpoints/model_best.pth`
  - SHA256: `d340effd68031aba1cd2bc610c676878fb9105dd7f81b9ce15767280e665a5c0`
- Residual corrector: update `19000`; Seen-Dev Rel-L2 `0.07953662`, TKE `0.49601257`, MVPE `0.06441840`, point score `91.06089`.
  - Remote path: `/hy-tmp/realpde_runs/strong_backbone_clean_20260924_run1/exp1_strong_backbone_clean/residual_22k/model_best.pth`
  - SHA256: `d2b4ddd0df6064d053ec918c2671a93b7d9c2a40a15afe854bb3a5badf1438fc`

## Protocol and provenance

- Execution commit: `d8e6000ec08b39b0e853f03028e13488c2d95a0e` (`main` at launch).
- One GPU: NVIDIA GeForce RTX 3090 Ti; seed `41`; effective batch `8`.
- Backbone recipe and schedule: official `sim_real` CNO init; Dense-All + P0-A + MF-CNO + N2 + Vorticity + Stage-B extra Rel; Stage A 30,000 updates at `1e-5`, Stage B 5,000 updates at `3e-6`.
- Residual: frozen backbone; `ResidualCorrector3D`, hidden `96`, blocks `2`, max delta `0.04`; 22,000 updates, batch `8`, seed `41`, scalar gradient mode, frozen loss weights. Check `run_config.json` and the residual run config for the full provenance.
- Split: Train51 / Seen-Dev12 / AoA10 Holdout18; disjoint; `7575_0.h5` excluded; 41,317 dense legal train windows and 491 Seen-Dev windows. The launch record notes Holdout HDF5 metadata was read for split auditing; Holdout fields and metrics were not read until after GO.
- Preflight hashes for Clean Stage1, Clean Stage2, official sim_real CNO, data manifest, and starting kit are recorded in `run_config.json` and `preflight.json`.
- Tests: 23 passed; requested py_compile, `git diff --check`, and batch-8 finite-loss/gradient smoke check passed. Smoke check performed zero optimizer updates.
- End-to-end wall time from launch to final DONE marker: approximately `9h 55m 11s`; Strong Backbone training wall time: `5h 51m 47s`; Residual phase plus its evaluation/selection ran from `2026-09-24 23:42` to `2026-09-25 03:44` local time.
- Runner exit code: `0`; campaign, selected Seen-Dev evaluation, and conditional Holdout evaluation all have completion markers.
- Operational note: another independent GPU workload was present on the same single-GPU host during part of the Residual phase. This may have affected throughput; the fixed update count and configured scientific parameters completed, with no detected OOM/NaN or protocol failure.
- Scientific protocol changed: `NO`. Codabench: `NO`. Locked-final/private: `NO`. Full-data refit: `NO`. Submission packaging: `NO`.

## Contents

This directory contains run/preflight/configuration provenance, split manifests, backbone milestones, all Residual Seen-Dev milestone primary metrics (sanitized to update/Rel-L2/TKE/MVPE/point score), selected Seen-Dev and permitted Holdout metrics, and per-trajectory/per-horizon CSV evidence. It contains no checkpoint weights, HDF5 data, or full training console logs.
