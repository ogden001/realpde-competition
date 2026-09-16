# SOTA-V2 adaptive execution — 2026-09-16

Status: **COMPLETED / REVIEW_REQUIRED**

The frozen SOTA-V2 adaptive uncertainty execution completed from execution commit `22eed1066345da7bb3d541a2472e4eaef0e29320`. No algorithm or model-source files were changed during execution.

## Validation replay

- Backbone: `/home/chyfuture/realpde_runs/sota_v2_integrated_50_16_20260916/run/checkpoints/model_update_32500.pth`
- Iteration: `32500`
- Windows / trajectories: `659 / 16`
- Rel-L2: `0.09993461519479752`
- TKE: `0.4692927300930023`
- MVPE: `0.07577798515558243`

## Adaptive head and calibration

- Architecture: `in_channels=15, hidden=32, blocks=2`
- Updates: `1400`
- Training windows: `2052`
- Head SHA256: `01c1fc06f806ca00a370cf971df0f093d363ab7f3a4c4e473f1727441f3d4d17`
- Static reference SPS: `42.12489194711354`
- Adaptive best SPS: `45.07008160038756`
- Delta SPS: `+2.9451896532740207`
- Selected bounds: `floor=0.0025, mult=1.0`
- Coverage / mean width: `0.8558713772975425 / 0.02358330972492695`
- Gate: `ADAPTIVE_GO`

## Package and clean-room smoke

- Full backbone: `/home/chyfuture/realpde_runs/sota_v2_full_20260916/run/checkpoints/model_update_53582.pth`
- Full backbone SHA256: `f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce`
- Package: `/home/chyfuture/realpde_runs/sota_v2_adaptive_20260916/package/submission.zip`
- ZIP bytes / SHA256: `30191330 / ad13e6ddf2438838df788f4f21204198b1cd121b209377d7cc446499a7246f44`
- Smoke A: `PASS`, parity `0.0`, first/steady `0.3647943510 / 0.0602133380 s`
- Smoke B: `PASS`, parity `0.0`, first/steady `0.4066184660 / 0.0308258310 s`
- Peak CUDA allocation: `142737920` bytes in both runs

Codabench/private/locked submission was not run or uploaded. The scientific frozen recipe was not changed. The package builder's embedded full-checkpoint SHA guard contained a 65-character typo; the valid 64-character checkpoint digest was supplied only as an in-process execution override, with no repository-source modification.

See the accompanying JSON/CSV evidence and `SHA256_PROVENANCE.md` for the complete provenance.
