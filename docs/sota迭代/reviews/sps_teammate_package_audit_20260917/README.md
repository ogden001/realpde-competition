# Teammate Codabench SPS package audit — 2026-09-17

Status: **SOURCE AUDIT / REVIEWED BY SOL**

Source artifact inspected outside Git:

`submission_all81_probe_fp16_20260915.zip`

This document records only facts recoverable from the submitted package plus explicitly marked inference. It is the source basis for `tools/sps_teammate_uncertainty_runtime.py` and `tools/realpde_sps_teammate_final.py`.

## Package-confirmed uncertainty recipe

The submitted `uncertainty_head.pt` records:

- architecture: hidden `32`, blocks `2`, dropout `0.0`, include pressure `true`;
- sigma limits: `1e-4 .. 1.0`;
- `sigma0 = 0.02`;
- training budget configured as `2000` updates;
- evaluation interval `200`;
- batch size `8`;
- LR `0.001`;
- weight decay `1e-5`;
- training windows `2701`, validation windows `640`;
- `train_on_all = false`;
- floor grid `[0, 0.0025, 0.005, 0.0075]`;
- multiplier grid `[0.5, 1, 1.5, 2, 2.5, 3, 4]`;
- packaged best checkpoint iteration `1800`;
- packaged local best SPS `48.95572376852203`;
- packaged best bounds `(floor=0.0025, mult=1.0)`.

The saved state loads strictly into the reproduced architecture with `141448` parameters.

## Exact 35-channel inference features

The submitted `submission.py` constructs future-aligned features as:

1. augmented point prediction: `13` channels;
2. augmented last observed frame repeated over Future20: `13` channels;
3. linear extrapolation from the final two observed frames: `3` channels;
4. prediction minus last observation: `3` channels;
5. prediction minus linear extrapolation: `3` channels.

Total: `35` channels.

Each 13-channel augmented block contains:

`u, v, p, speed, kinetic_energy, du_dt, dv_dt, vorticity, divergence, strain_magnitude, x_coord, y_coord, t_coord`.

The head is:

`LayerNorm(35) -> Conv3d(35,32,3) -> GroupNorm -> SiLU -> 2 residual blocks -> Conv3d(32,2,1)`.

The teammate package applies this head to the base CNO prediction before its residual corrector. Our SOTA-V2 has no equivalent separable direct-CNO + corrector inference path, so the controlled transfer experiment applies the same uncertainty feature/head recipe to the frozen final SOTA-V2 prediction. Point prediction itself is unchanged.

## Bounds

The submitted runtime uses symmetric UV intervals:

`half_u = 0.0025 + 1.0 * sigma_u`

`half_v = 0.0025 + 1.0 * sigma_v`

Pressure interval width is zero. No asymmetric, horizon-specific, conformal, or region-specific calibration appears in the package.

## Training-loss evidence boundary

The package does **not** contain the uncertainty-head training script. Therefore the exact training-loss source code cannot be claimed as package-confirmed.

The stored head emits `log_std`, the artifact records `sigma0=0.02`, and its saved training losses are negative (approximately `-3.6` to `-4.8`), which is consistent with Gaussian NLL in log-standard-deviation form. The controlled transfer runner therefore uses Gaussian NLL on bounded `log_std`. This is an explicit reconstruction choice, not a claim that the missing teammate training source was independently verified.

## Controlled transfer protocol

The new experiment deliberately keeps the user's frozen protocol:

- Phase A: 50 Train / 16 Dev, canonical fixed stride-20 windows;
- frozen SOTA-V2 validation point predictor `@32500`;
- 35-channel teammate uncertainty features + teammate head;
- maximum `2000` updates, evaluation every `200`, best Dev SPS checkpoint selected;
- same frozen 28-row calibration grid;
- GO only if SPS improves by at least `+1.5`, mean UV width <= `1.15x` baseline, and point prediction parity <= `1e-7`.

If GO, Phase B trains a fresh full-specific teammate-style head on all 82 released trajectories using canonical fixed windows. The number of Phase-B updates is frozen to the Phase-A selected checkpoint iteration; Phase-B does not recalibrate on full-data labels. A candidate package may then be built and clean-room verified, but Codabench submission remains manual after Sol review.
