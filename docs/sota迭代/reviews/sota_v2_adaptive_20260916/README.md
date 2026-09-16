# SOTA-V2 adaptive execution — 2026-09-16

Status: **COMPLETED / REVIEW_REQUIRED**

The clean SOTA-V2 adaptive package rebuild completed from `REQUIRED_COMMIT=8b2e74a8dc4bde1ca7f7f8b755b3d902d7a740d9`. No retraining or recalibration was performed, and no algorithm or model-source files were changed during execution.

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
- Final candidate package: `/home/chyfuture/realpde_runs/sota_v2_adaptive_20260916/package_clean/submission.zip`
- ZIP bytes / SHA256: `30191330 / 9cfc055c6232d2b0aef9f88f1cb3de2aae883b7cff7f02659ed8b9cf85b3ed55`
- Smoke A: `PASS`, parity `0.0`, first/steady `0.3303255550 / 0.0307885470 s`
- Smoke B: `PASS`, parity `0.0`, first/steady `0.3408626430 / 0.0308223500 s`
- Peak CUDA allocation: `142737920` bytes in both runs

The package builder SHA guard is now fixed in the required source commit, and this clean rebuild passed without monkeypatch or in-process SHA override. The prior `package/submission.zip` evidence remains historical; the final candidate is `package_clean/submission.zip`. Codabench/private/locked submission was not run or uploaded.

See the accompanying JSON/CSV evidence and `SHA256_PROVENANCE.md` for the complete provenance.
