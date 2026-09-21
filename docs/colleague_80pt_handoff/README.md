# 80-Point Solution Executive Summary

```text
Online score: 80.078849
Rel-L2: 94.739714
TKE: 76.892538
MVPE: 93.845797
time: 86.542490
SPS: 40.209117
final: 80.078849

Git commit: UNKNOWN for the server-side /runs scripts (they were not under version control).
            Repository baseline before handoff: a928355 (branch codex/feature-engineering).
            This handoff commit: see `git log -1`.
Final config: configs/colleague_80pt_final.yaml

Checkpoint(s):
  base CNO (all81):   /runs/cno_final_all81.pt
                      sha256 ff28aaf0114d57e320e5ea9cf9945d874a7482f22e3f0aeaed5ebbe286573f8a
  residual corrector: /runs/residual_h96x8_all81_20260920/model_best.pth
                      sha256 909fdc7f8a6a42335e4ea4ce7471fc9a507135e9a29a4927bc7b7318be9c85b2
  uncertainty head:   /runs/head_h96x8_h64logmae/head_5000.pth
                      sha256 1dc6b56cf29b62a0e017cb0b434262215f81d94df22b964a04e1898720aa0522

Train command:     scripts/colleague_80pt_train.sh
Eval command:      scripts/colleague_80pt_eval.sh
Inference command: scripts/colleague_80pt_infer.sh
SPS command:       scripts/colleague_80pt_sps.sh
Package command:   scripts/colleague_80pt_package.sh
```

## 1. 最终方案是什么？

在官方 CNO3d baseline 之上做两阶段残差修正，最后接一个逐像素不确定度头生成 SPS 区间：

1. 用全部 81 条可用真实轨迹把 CNO 从 `sim_real_cno.pth` 全量微调成 `cno_final_all81.pt`（stride=1，8723 updates）。
2. 冻结 CNO，训练一个 3D 残差修正器 `ResidualCorrector3D`（hidden=96, blocks=2, max_delta=0.04），输入为过去 20 帧 + CNO base 预测 + 线性外推，输出对 base 的修正量。
3. 冻结 CNO + 残差修正器，用 stride=5 缓存的训练窗口训练 `UncertaintyHead3D`（hidden=64, blocks=2, logmae loss, 6000 updates），逐像素预测 log sigma。
4. 推理时区间半宽 `h = floor + mult * sigma + rel * |pred|`，参数由 dev 集网格扫描得到。
5. 打包为 submission zip，只用包内 checkpoint 与依赖，官方 bench 通过 `not_fallback True`。

## 2. 最核心的 3~5 个技术点

```text
1. CNO 全量 all81 微调（stride=1, 8723 updates, 带 wake/ramp 加权的 decomp loss）
2. 冻结 backbone 的 3D residual corrector（h96/b2, max_delta=0.04, alpha=1.0）
3. 残差头容量 × 训练步数匹配：h96 x4/x8 长训（19200/38400 updates）持续提升
4. 逐像素 uncertainty head（h64/b2, logmae loss）预测 SPS 区间半宽的 sigma 项
5. 区间参数网格扫描 floor/mult_u/mult_v/rel，选用 dev SPS 最优组合
```

## 3. 从输入到输出的完整 pipeline

```text
input (B, 20, 32, 64, 3)  # u, v, p(=0)
  -> CNO3d backbone (frozen after stage 1)
  -> base future (B, 20, 32, 64, 3)
  -> ResidualCorrector3D(features(x, base))
       delta = 0.04 * tanh(raw_delta / 0.04), delta[...,2] = 0
  -> prediction = base + 1.0 * delta, prediction[...,2] = 0
  -> UncertaintyHead3D(features(x, base))
       log_std = clamp(log(sigma), log(1e-4), log(1.0))
  -> half_width_u = 0.0025 + 1.25 * sigma_u + 0.005 * |pred_u|
     half_width_v = 0.0025 + 1.50 * sigma_v + 0.005 * |pred_v|
  -> lower = prediction - half_width
     upper = prediction + half_width
  -> {"prediction": ..., "lower": ..., "upper": ...}
```

`features(x, base)` 由 5 部分组成：

```text
augment(base)                    # base 预测的工程特征
augment(x)[:, -1:].expand(20)    # 最后一帧过去帧特征
future_linear_extrapolation(x)   # 线性外推
base - last_raw                  # base 相对最后一帧的增量
base - linear                    # base 相对线性外推的增量
```

## 4. 训练需要几个阶段

三个阶段，严格串行：

```text
Stage 1: CNO all81 fine-tune      -> cno_final_all81.pt
Stage 2: residual corrector       -> residual_h96x8_all81_20260920/model_best.pth
Stage 3: uncertainty head         -> head_h96x8_h64logmae/head_5000.pth
```

## 5. 最终 submission 使用哪些 checkpoint

```text
model.pth             = residual_h96x8_all81_20260920/model_best.pth
uncertainty_head.pt   = head_h96x8_h64logmae/head_5000.pth
```

`model.pth` 内部同时包含 CNO backbone 与 residual corrector 的权重（`model_state_dict` 同时有 `base_model.*` 和 `corrector.*`）。

## 6. SPS 是怎么做的？

逐像素、逐帧、逐通道（u/v）预测 sigma，区间半宽线性组合 floor、sigma、|pred|：

```text
h_ch = floor + mult_ch * sigma_ch + rel * |pred_ch|
```

最终参数：

```text
floor  = 0.0025
mult_u = 1.25
mult_v = 1.50
rel    = 0.005
sigma clamp = [1e-4, 1.0]
```

详见 `SPS.md`。

## 7. 完整复现需要执行哪些命令

```bash
# Stage 1: CNO all81
bash scripts/colleague_80pt_train.sh --stage cno

# Stage 2: residual corrector
bash scripts/colleague_80pt_train.sh --stage residual

# SPS feature cache + uncertainty head
bash scripts/colleague_80pt_train.sh --stage head

# dev evaluation
bash scripts/colleague_80pt_eval.sh

# SPS parameter scan
bash scripts/colleague_80pt_sps.sh

# packaging
bash scripts/colleague_80pt_package.sh
```

## 8. 当前还有哪些 UNKNOWN？

```text
1. 4 相位 downsampling augmentation（P00/P01/P10/P11）是否有效：对照实验进行中，尚未定论。
2. h96x16（76800 updates）是否继续提升：训练进行中，尚未定论。
3. AoA augmentation：未实现，未验证；本方案完全未使用 AoA 增强。
4. 官方评测机的精确硬件/时间计分基准未知，本地 time_score 与线上存在小幅波动。
5. `sim_real_cno.pth` 的官方训练细节不在本仓库范围内（官方 baseline 权重）。
```
