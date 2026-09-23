# REALPDE Clean Baseline V1

Status: REVIEW_REQUIRED

## Purpose

This is the frozen research baseline for the remaining RealPDE Track 1 experiments.

The model architecture and losses are copied from the colleague 80.078849 solution. The research protocol is intentionally different from the colleague submission workflow: it removes train/dev overlap, holds out one entire angle of attack, fixes reproducible dense temporal sampling, and selects checkpoints using point-prediction metrics only.

## Data split

The canonical manifest is:

`configs/clean_baseline_v1_split.json`

The complete 81 usable real-PIV trajectories are partitioned as:

- Train51: AoA 0/5/15/20 only.
- Seen-Dev12: three trajectories each at AoA 0/5/15/20.
- Unseen-AoA Holdout18: every available AoA=10 trajectory.
- `7575_0.h5` remains excluded by `BAD_TRAIN_FILES`.

Important correction: an earlier draft said 55/12/14. That count came from the older frozen 50/16 subset. Auditing all 81 usable trajectories shows 18 AoA=10 trajectories, so a genuinely unseen-10-degree protocol must be 51/12/18.

The runner refuses to start unless all 81 usable files are accounted for, all 18 ten-degree files are in holdout, no ten-degree file is in Train/Dev, Dev is balanced 3/3/3/3 across 0/5/15/20, and the three sets are disjoint.

## Frozen temporal sampling protocol

Training, for both Stage 1 and Stage 2:

- Past20 -> Future20.
- sub_sample=2, P00 only.
- stride=1.
- enumerate every legal temporal start.
- global deterministic shuffle with seed 41.
- without replacement inside an epoch.
- batch size 8.

Validation and holdout:

- stride=20.
- start=0.
- shuffle=False.

This sampling protocol is part of the baseline definition and should not change in later experiments unless sampling itself is the explicit experimental variable.

## Stage 1: CNO

Initialization:

`sim_real_cno.pth`, expected SHA256
`82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61`.

Architecture: colleague-80 CNO3d unchanged.

Loss: colleague-80 Stage-1 decomp loss unchanged:

- mean-field weighted MSE;
- fluctuation weighted MSE;
- 0.05 TKE term;
- 0.01 pressure-zero term;
- same late-horizon ramp;
- same wake-region weighting.

Optimization:

- AdamW, lr=1e-4.
- batch=8.
- FP32.
- CosineAnnealingLR.
- max 8723 updates.
- Seen-Dev evaluation at step 0, every 1000 updates, and final step.

Stage-1 best is selected by point_score = mean of the official-v9 Rel-L2, TKE, and MVPE subscores. SPS and runtime are excluded.

## Stage 2: residual corrector

The Stage-1 best CNO is frozen.

Architecture and objective are copied from colleague-80:

- ResidualCorrector3D.
- hidden=96.
- blocks=2.
- dropout=0.
- max_delta=0.04.
- history_context=False.
- alpha fixed to 1.0.
- final Conv3d zero-init, so step 0 equals the Stage-1 CNO.

Loss weights remain:

- point 1.0
- MSE 0.05
- TKE 0.06
- temporal 0.03
- spatial gradient 0.015
- pressure-zero 0.01
- residual-target MSE 0.25
- delta penalty 0.02

Optimization:

- AdamW, lr=2e-4, weight_decay=1e-5.
- batch=8.
- gradient clip=1.0.
- FP32.
- max 38400 updates.
- Seen-Dev evaluation every 1000 updates.

Unlike the colleague submission recipe, Stage 2 also uses the frozen research stride=1 training protocol. Dev remains stride=20.

Residual alpha is fixed to 1.0 during evaluation. Best checkpoint is selected by the same point-only score; SPS/time cannot influence the result.

## Holdout discipline

AoA10 Holdout18 is never available to training or checkpoint selection.

After Stage 1 and Stage 2 are completely finished, the runner evaluates only:

- Stage2 step 0, equivalent to the selected Stage-1 CNO.
- Stage2 best.
- Stage2 final.

The holdout result is diagnostic only. It must not trigger automatic retuning or reruns.

## Outputs

The runner writes:

- campaign_manifest.json
- train_dev_manifest.json
- holdout_eval_manifest.json
- Stage1 run_config, learning curve, per-horizon metrics, best/final checkpoints
- Stage2 run_config, learning curve, sampling audit, best/final checkpoints
- final AoA10 holdout diagnostics
- SHA256 evidence
- REVIEW_REQUIRED status

It does not train SPS, refit all released data, package a submission, access locked-final/private data, or access Codabench.


## 3090 / 3090 Ti runtime engineering profile

The clean baseline preloads released Train/Dev PIV tensors into host RAM after P00 subsampling and float32 conversion. Window semantics are unchanged: the same legal starts, stride, shuffle order and training batch are used; only repeated HDF5 opens/reads are removed.

For DataLoader workers > 0, clean training uses persistent workers plus prefetching. RAM-cache size, worker count and prefetch settings are written into run/runtime evidence.

Optional `--runtime-profile` on `run_clean_baseline_v1.py` performs disposable batch=8 versus batch=16 benchmarks for Stage1 and Stage2. Selection uses throughput and VRAM only, via the existing rule that b16 must achieve at least 1.20x samples/sec while staying within the configured VRAM headroom.

The recommendation is evidence for future experiment families only. It never overrides REALPDE_CLEAN_BASELINE_V1, whose actual training remains batch=8.
