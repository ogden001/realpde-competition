# Teammate Adaptive Probe Reference — 2026-09-05

## Provenance

本文件只记录用户提供的历史 teammate submission `submission_probe_adaptive_sps_20260903.zip` 中可直接核验的结构和 checkpoint metadata。原 ZIP / checkpoint 不提交到 Git，也不直接复用其权重。

该 teammate submission 的线上结果：

| Metric | Score |
|---|---:|
| Rel-L2 | 94.367021 |
| TKE | 77.166035 |
| MVPE | 93.551235 |
| Time | 86.006050 |
| SPS | 35.596250 |
| Final | 78.685445 |

本项目只移植已经可审计的 **Residual Corrector + Adaptive Uncertainty** 思路到当前 P0-A + N2 SOTA，不替换当前 backbone。

## Residual Corrector

结构：
- backbone 冻结；
- corrector hidden `64`；
- residual blocks `2`；
- dropout `0`；
- output `3` channels，pressure delta 强制为 `0`；
- `max_delta=0.04`，使用 `0.04 * tanh(raw_delta / 0.04)`；
- correction alpha `1.0`。

Future feature 输入：
- backbone future prediction 的 deterministic flow features；
- Past20 最后一帧 flow features，broadcast 到 Future20；
- 基于最后两帧的简单线性外推；
- `base_pred - last_frame`；
- `base_pred - linear_extrapolation`。

Flow feature helper 使用 raw u/v（corrector 不含 pressure）并附加：speed、kinetic energy、du/dt、dv/dt、vorticity、divergence、strain magnitude，以及由 tensor shape 生成的 normalized index x/y/t。这里的 x/y/t 不是 HDF5 physical coordinates，也不使用 Re/AoA/metadata。

Teammate checkpoint metadata：
- updates `2400`；
- eval interval `600`；
- batch `8`；
- lr `1e-4`；
- weight decay `1e-5`；
- cosine LR decay over 2400 updates；
- trainable parameters `500,613`；
- loss weights：`point=1.0, mse=0.05, tke=0.12, temporal=0.04, grad=0.02, p_zero=0.01`；
- residual MSE coefficient `0.15`；
- delta penalty coefficient `0.05`。

Loss semantics for the port are frozen as：
- point: relative L2 on future u/v field；
- mse: u/v MSE；
- tke: relative L2 on differentiable future TKE map；
- temporal: relative L2 on first temporal differences of future u/v；
- grad: relative L2 on spatial finite-difference gradients of future u/v；
- p_zero: predicted pressure MSE to zero；
- residual_mse: MSE between predicted correction delta and `target - backbone_prediction` on u/v；
- delta_penalty: mean squared correction delta on u/v。

## Adaptive Uncertainty Head

结构：
- hidden `32`；
- residual blocks `2`；
- dropout `0`；
- input uses the same future-feature recipe, with pressure channel included in deterministic feature construction；
- output `2` channels of `log_std` for u/v；
- `sigma = exp(log_std)`；
- clamp sigma to `[1e-4, 1.0]`；
- initial sigma `0.02`。

Training metadata：
- batch `8`；
- lr `1e-3`；
- weight decay `1e-5`；
- evaluated every `200` updates；
- historical best update `1400`；
- Gaussian NLL semantics: `mean(log_sigma + 0.5 * ((target_uv - final_prediction_uv) / sigma)^2)`，不含常数项。

Bounds：
- `half_width_uv = floor + mult * sigma`；
- pressure half-width `0`；
- historical best `floor=0.0025, mult=1.0` at update 1400；
- calibration grid recorded in teammate artifact：
  - floor: `0 / 0.0025 / 0.005 / 0.0075`
  - mult: `0.5 / 1 / 1.5 / 2 / 2.5 / 3 / 4`

Important implementation detail from the submitted wrapper：the uncertainty head receives **raw Past20 + uncorrected backbone future prediction features**; its NLL target is the residual uncertainty of the final prediction. The head does not require future truth or metadata at inference.

## Porting Rule

- Do not reuse teammate weights.
- Current backbone remains P0-A + N2 CNO.
- Validation experiment uses frozen 50/16 protocol and current P0-A validation checkpoint.
- Only if Residual Corrector passes the pre-registered TKE-protection gate is it refit on all 82 released trajectories.
- Adaptive uncertainty is trained/calibrated only from the clean validation-family training protocol, not by fitting uncertainty residuals on the all-82 in-sample full model.
- No locked-final/private-test/Codabench tuning during the overnight run.
