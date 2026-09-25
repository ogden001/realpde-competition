# Experiment 3 — Residual-aware SPS Clean

Status: `REVIEW_REQUIRED`; primary Seen-Dev gate: `NO_GO`.

## Frozen protocol

- Execution commit: `396dc89dec46ac1118a9b1dd33709784903d8e98`.
- GPU/runtime: NVIDIA GeForce RTX 3090 Ti, PyTorch 2.4.0+cu121, CUDA 12.1.
- Split: Train51 / Seen-Dev12 / AoA10 Holdout18; trajectory-disjoint. Training stride 5; Seen-Dev stride 20, start 0.
- Frozen point predictor: Clean Stage1 + Stage2. Only the uncertainty head was trained; the point predictor remained frozen.
- Head: h64, 2 blocks, dropout 0; masked log-MAE objective; batch 16; AdamW LR 1e-3 with warmup/cosine schedule; 5,000 updates.
- Calibration: original 300-combination family. Static calibration and adaptive grids at every 500-update evaluation are included.
- Runtime: 17,987.97 seconds of training (about 5 hours).

## Final Seen-Dev result

- Selected head step: 4,500.
- Static SPS: 48.41636; adaptive SPS: 52.23542; gain: +3.81906 (gate requires at least +1.0).
- Static mean interval width: 0.01243318; adaptive: 0.01855358; ratio: 1.49226 (gate allows at most 1.20).
- Point prediction parity max absolute difference: 0 (gate allows at most 1e-7).
- Gate: `NO_GO` because the interval-width ratio exceeds its limit. AoA10 Holdout was not accessed; Codabench and locked-final/private data were not accessed.

## Evidence files

- `summary.json`: final metrics, gate, selection, runtime and access flags.
- `training_progress.json`: loss samples and Seen-Dev evaluation records across training.
- `static_calibration_grid.json`: frozen static baseline calibration sweep.
- `calibration_grid_00500.json` through `calibration_grid_05000.json`: all ten intermediate adaptive calibration sweeps.
- `preflight.json` and `train_dev_manifest.json`: environment, source asset hashes and split provenance.
- `checkpoint_sha256.json`: selected checkpoint hash only. The checkpoint binary is intentionally not included.
- `EXP3_RESIDUAL_AWARE_SPS_REVIEW.md`: compact human review summary.
- `SHA256SUMS.txt`: integrity hashes for all evidence files except this checksum list.
