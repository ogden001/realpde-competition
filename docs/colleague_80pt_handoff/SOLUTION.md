# SOLUTION

本文件描述 80.078849 分线上方案的完整预测与区间 pipeline。所有内容来自实际运行代码
（`tools/colleague_80pt/residual_multi.py`、`tools/colleague_80pt/submission/submission.py`、
`tools/colleague_80pt/final_all81.py`）。

## 1. 总体结构

```text
Past 20 frames (u, v, p=0)  ->  CNO3d backbone  ->  Base future 20 frames
                                                        |
                                                        v
                            ResidualCorrector3D  <--  features(x, base)
                                                        |
                                                        v
                                            prediction = base + 1.0 * delta
                                                        |
                                                        v
                            UncertaintyHead3D    <--  features(x, base)
                                                        |
                                                        v
                                  lower / upper = prediction +/- half_width
```

没有 ensemble，没有 TTA，没有 AoA 旋转。最终 submission 是单模型。

## 2. Backbone：CNO3d

官方 baseline 类，构造参数与 submission 内 `rpde_baselines/cno.py` 完全一致：

```python
CNO3d(
    in_dim=3,        # u, v, p；p 实际恒为 0
    out_dim=3,
    out_dim_mult=1,
    in_size=64,
    N_layers=3,
    activation="LeakyReLU",
)
```

Stage 1 从官方 `sim_real_cno.pth` 初始化，在全部 81 条可用真实轨迹上微调，得到
`cno_final_all81.pt`。Stage 2/3 中该 backbone 完全冻结（`requires_grad_(False)`，eval 模式）。

## 3. Residual corrector

`ResidualCorrector3D` 是 3D 卷积残差网络：

```text
input  = build_future_features(x, base_pred)          # (B, 20, 32, 64, C_in)
LayerNorm(C_in)
permute -> (B, C_in, 20, 32, 64)
Conv3d(C_in, 96, kernel=3, padding=1)
GroupNorm(8, 96) + SiLU
ResidualBlock3D(96) x 2      # 每个 block: Conv3d-GN-SiLU-Dropout3d-Conv3d-GN + residual
Conv3d(96, 3, kernel=1)
permute -> (B, 20, 32, 64, 3)
delta = 0.04 * tanh(raw_delta / 0.04)
delta[..., 2] = 0
```

配置（来自 `evidence/run_config.json`）：

```text
hidden            = 96
blocks            = 2
dropout           = 0.0
include_pressure  = true
max_delta         = 0.04
history_context   = false
train_alpha       = 1.0
correction_alpha  = 1.0
```

`trainable_parameters = 1,087,849`，`total_parameters = 9,048,316`。

### 3.1 build_future_features

```python
base            = zero_pressure(base_pred)
last_raw        = zero_pressure(ensure_three_channels(x[:, -1:])).expand(-1, 20, ...)
linear          = future_linear_extrapolation(x, 20)
base_features   = augment_torch(base, include_pressure=True)
past_features   = augment_torch(x, include_pressure=True)
last_features   = past_features[:, -1:].expand(-1, 20, ...)
pieces = [base_features, last_features, linear, base - last_raw, base - linear]
features = cat(pieces, dim=-1)
```

`linear` 用最后一帧加线性趋势外推，步长从 `1/20` 到 `1.0`。`augment_torch` 来自
`realpde_feature_engineering.py`，是确定性的工程特征（不含随机增强）。

## 4. Uncertainty head

`UncertaintyHead3D` 与 corrector 共享同一组特征构造，但更小：

```text
LayerNorm(C_in)
Conv3d(C_in, 64, kernel=3, padding=1)
GroupNorm(8, 64) + SiLU
ResidualBlock3D(64) x 2
Conv3d(64, 2, kernel=1)          # 输出 u/v 两个通道的 log_std
log_std = clamp(log_std, log(1e-4), log(1.0))
sigma   = exp(log_std)
```

配置：

```text
head_hidden          = 64
head_blocks          = 2
head_dropout         = 0.0
head_include_pressure= true
head_min_sigma       = 1e-4
head_max_sigma       = 1.0
```

## 5. 区间生成（SPS）

```python
half_width[..., 0] = 0.0025 + 1.25 * sigma[..., 0] + 0.005 * |pred[..., 0]|
half_width[..., 1] = 0.0025 + 1.50 * sigma[..., 1] + 0.005 * |pred[..., 1]|
half_width[..., 2] = 0.0
lower = pred - half_width
upper = pred + half_width
```

逐元素（per-pixel）、逐帧（20 帧）、逐通道（u/v）粒度，没有 per-window 全局缩放。
详细审计见 `SPS.md`。

## 6. 推理入口

`submission.predict(input_array, metadata=None)`：

1. 惰性加载 `model.pth` + `uncertainty_head.pt`。
2. 校验输入形状与输出形状 `(N, 20, 32, 64, 3)`。
3. 任何异常都回退到 persistence（`repeat(input[-1], 20)`），但最终 80 分包 bench 结果是
   `not_fallback True`，即正常走模型。
4. `model.pth` 加载失败时打印完整 traceback 并回退。

`model.pth` 内部 key 为 `model_state_dict`，同时含 `base_model.*` 与 `corrector.*`；
`uncertainty_head.pt` 内部 key 为 `head_state_dict`（或直接 state dict）。

## 7. 未使用的模块（明确记录）

以下内容在本方案中**不存在或未使用**，不要误加：

```text
- ensemble
- test-time augmentation / TTA
- AoA / 攻角旋转增强
- 历史上下文特征 (history_context=False)
- per-horizon SPS 校准
- AMP / bf16 训练（尝试过基准，因精度顾虑未采用）
- 多 backbone 融合
```

## 8. 与 baseline 的差异总结

```text
官方 baseline CNO (sim_real_cno.pth)
  + Stage1 all81 fine-tune (stride=1, 8723 updates)
  + Stage2 residual corrector h96/b2/38400 updates
  + Stage3 uncertainty head h64/b2/logmae/6000 updates
  + SPS interval calibration (floor=0.0025, mult_u=1.25, mult_v=1.5, rel=0.005)
  = 80.078849 online final
```
