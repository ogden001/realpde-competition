# Joint training run2 — Sol review evidence (interim)

Status: `RUNNING / REVIEW_REQUIRED` at snapshot time.

Snapshot time: **2026-09-26 10:12 Asia/Shanghai**. Remote process PID 75089 was still running at approximately update **2,200 / 6,000** when stdout was copied. The latest completed evaluation in this package is update 2,000. This is an interim snapshot and contains no final result.

## Run identity and provenance

- Host: `I2b5847212d00701227`, NVIDIA GeForce RTX 4090 (24,564 MiB); snapshot utilization 87%, memory use 11,426 MiB.
- Run directory: `/hy-tmp/realpde_runs/realpde_joint_training_v1_4090_20260926_run2`.
- Execution commit: `cbf25c73400e753de5d3d9b701118f9b01954b3e`; branch `main`, clean worktree.
- Training source: [`tools/train_sota_merge_joint.py`](../../../../tools/train_sota_merge_joint.py), SHA-256 `30710c3e663b9bca2c2dd9b0806381d2d865268d18f8c25c19de437c914e5b60`. The source is tracked in this repository at the execution commit; it is not duplicated here.
- The full process command, tracked source/runtime SHA-256 values, run file inventory, and checkpoint hashes are in [`remote_snapshot.txt`](remote_snapshot.txt).

## Data scope

The recorded manifest has **51 Train / 12 Seen-Dev / 18 AoA=10 holdout** trajectories. The trainer resolves Train for updates and Seen-Dev for scheduled evaluation. The captured run contains Seen-Dev outputs only. Its run config records `locked_final_accessed=false`, `private_accessed=false`, and `codabench_accessed=false`. No H5 data or checkpoint payloads are included in this review evidence.

## Interim Seen-Dev curve

Metrics are over 491 Seen-Dev windows. `final_*` is after the residual corrector; `base_*` is backbone only.

| Update | Final Rel-L2 | Final TKE | Final MVPE | Final point score |
|---:|---:|---:|---:|---:|
| 0 | 0.0913941 | 0.4658932 | 0.0718891 | 91.088923 |
| 1,000 | 0.0892417 | 0.4628836 | 0.0680017 | 91.215287 |
| 2,000 | 0.0887024 | 0.4630233 | 0.0669236 | 91.238804 |

Descriptively, update 2,000 vs. update 0 has Rel-L2 -2.95%, TKE -0.62%, MVPE -6.90%, and point score +0.150 on Seen-Dev. These are not official composite scores and are not presented as an unbiased final selection result.

## Protocol review flag

[`CLEAN_CAMPAIGN_BOUNDARIES.md`](CLEAN_CAMPAIGN_BOUNDARIES.md) records the current Clean campaign rules, including the ban on end-to-end joint fine-tuning. The remote trainer updates backbone and residual corrector in one optimizer, applies Stage-A, Stage-B, and residual losses in each update, and sets `base_pred_detached=false`. The residual checkpoint is marked `warm_start_cross_backbone`; it began from a different backbone digest and is being jointly retrained. Sol should classify this run before using it for any Clean claim or merge decision. This archive records evidence only; it does not make a scientific GO/NO-GO decision.

## Evidence files

- `run_config.json`, `clean_baseline_v1_split.json`: protocol/configuration and data roster.
- `stdout.log`: complete captured raw log at snapshot time (23 lines, 13,568 bytes).
- `aggregate_metrics.csv` and `eval_seen_dev_*`: curve and milestone metrics/horizon summaries.
- `remote_snapshot.txt`, `stdout_snapshot_meta.txt`: execution, host, Git, file, and digest provenance.
- `evidence_manifest.json`: per-file sizes and SHA-256 digests for this archive.
