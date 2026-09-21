# SPS

本文件是 80 分方案 uncertainty / interval 部分的完整审计。所有数值来自实际代码与
`evidence/scan_bounds_h96x8.json`、`evidence/eval_log.json`。

## 0. 摘要（按 handoff 模板）

```text
SPS / uncertainty method: learned per-pixel sigma (logmae head) + linear interval family
                          h = floor + mult * sigma + rel * |pred|

Calibration split:        dev 16 trajectories from split_paths(seed=41, val_fraction=0.2)
Calibration sample count: 640 windows x 20 frames x 32 x 64 pixels x 2 channels

quantile:                 N/A  (not a conformal / quantile method)
alpha:                    N/A  (no conformal alpha; note residual correction alpha = 1.0)
scale:                    mult_u = 1.25, mult_v = 1.50
clip:                     sigma clamped to [1e-4, 1.0]
min_sigma:                1e-4
max_sigma:                1.0

granularity:
  - global:    NO
  - per-frame: YES (20 output frames independently)
  - per-channel: YES (u, v separately)
  - per-pixel: YES (each HxW location independently)

post-processing:          none; half_width[...,2] = 0

implementation file:      tools/colleague_80pt/submission/submission.py
implementation functions: UncertaintyHead3D.forward, _with_bounds
scoring implementation:   tools/colleague_80pt/realpde_sps_scoring.py
scan implementation:      tools/colleague_80pt/scan_bounds.py

final submission behavior: predict() returns {"prediction", "lower", "upper"}
                           with lower = pred - h, upper = pred + h
```

## 1. 官方 SPS 评分定义（用于 dev 选择）

来自 `realpde_sps_scoring.py` 的模块 docstring：

```text
SPS = 0.5 * sps_dm + 0.3 * sps_tke + 0.2 * sps_mvpe

each branch averages over scored (target != 0) elements:

    (1 - accuracy(err)) * exp(-(upper - lower) / sigma_global) * inside

where
    accuracy(err) = err / (0.5 + err)
    (1 - accuracy) = 0.5 / (0.5 + err)
    inside = 1 if lower <= target <= upper else 0

sigma_global = 0.0563870259
```

时间分：

```text
time_score = 100 / (1 + sqrt(t_neural / T_NUMERICAL_SEC))
T_NUMERICAL_SEC = 0.72896
```

说明：SPS 只统计 `target != 0` 的元素；`lower/upper` 对第 3 通道（压力）无效，因为
`measured_channels` 只保留非零通道（实测为 u/v 两通道）。

## 2. 我们的 uncertainty 方法

不是 quantile regression，也不是 conformal prediction。做法是：

1. 训练一个 3D head 逐像素预测 `log_std`。
2. 运行时 `sigma = exp(log_std)`。
3. 区间半宽是 floor、sigma、预测幅值三者的线性组合。

```python
# submission.py
half_width[..., 0] = _UNC_FLOOR_U + _UNC_MULT_U * sigma[..., 0] + _UNC_REL * abs(pred[..., 0])
half_width[..., 1] = _UNC_FLOOR_V + _UNC_MULT_V * sigma[..., 1] + _UNC_REL * abs(pred[..., 1])
half_width[..., 2] = 0.0
lower = pred - half_width
upper = pred + half_width
```

对应官方 SPS 中的 `(upper - lower) = 2 * half_width`，所以指数项为
`exp(-2 * half_width / sigma_global)`。

## 3. Sigma head

```text
module        : UncertaintyHead3D
hidden        : 64
blocks        : 2
dropout       : 0.0
input         : build_future_features(x, base_pred, include_pressure=True, history_context=False)
output        : 2 channels (u, v) log_std, shape (B, 20, 32, 64, 2)
clamp         : log_std in [log(1e-4), log(1.0)]
training loss : logmae = mean(|log_std - log(|error| + 1e-6)|)
```

其中 `base_pred` 是冻结 CNO 的输出，`error = target - (base + residual_delta)`，即最终
corrected prediction 的误差。head 的输入与 residual corrector 使用同一组特征，但 head 不参与
prediction 的生成。

## 4. 校准数据

```text
source        : frozen_h96x8 cache (tools/colleague_80pt/cache_frozen.py)
champion used : /runs/residual_h96x8_all81_20260920/model_best.pth
train split   : all 81 trajectories, stride=5  -> 13202 windows
dev split     : 16 trajectories, stride=20     -> 640 windows
dev split def : split_paths(list_h5(real_root, BAD_TRAIN_FILES), 0.2, seed=41)
dtype         : float16
```

dev 相位固定为 P00（`[::2, ::2]`），与线上评测一致。扫描只在 dev 上进行，没有用 train 或
线上反馈。

## 5. 参数扫描

脚本：`tools/colleague_80pt/scan_bounds.py`（在 h96x8 上运行时把 `CACHE` 指向
`/runs/frozen_h96x8`，并把 `--head` 指向 `head_5000.pth`）。

扫描网格（共 300 组）：

```python
for floor in (0.0, 0.0025, 0.005):
    for mult_u in (0.5, 0.75, 1.0, 1.25, 1.5):
        for mult_v in (0.5, 0.75, 1.0, 1.25, 1.5):
            for rel in (0.0, 0.0025, 0.005, 0.0075):
                ...
```

评分函数为 `fast_sps`，与官方 SPS 公式一致（`(0.5*dm + 0.3*tke + 0.2*mvpe)`，
`exp(-(hi-lo)/SIGMA_GLOBAL)`，`inside` 指示函数）。

### 扫描结果（evidence/scan_bounds_h96x8.json）

| rank | SPS | coverage | floor | mult_u | mult_v | rel |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 51.6332 | 0.8853 | 0.0025 | 1.25 | 1.50 | 0.0050 |
| 2 | 51.6232 | 0.8978 | 0.0025 | 1.50 | 1.50 | 0.0025 |
| 3 | 51.5985 | 0.8915 | 0.0025 | 1.25 | 1.50 | 0.0075 |
| 4 | 51.5955 | 0.8906 | 0.0025 | 1.50 | 1.50 | 0.0000 |

第 1 与第 2 名只差 0.010，说明最优区域是**平坦**的：参数微调已经榨干，不要期待
通过继续调 floor/mult/rel 拿分。

### 最终选择

```text
_UNC_FLOOR_U = 0.0025
_UNC_MULT_U  = 1.25
_UNC_FLOOR_V = 0.0025
_UNC_MULT_V  = 1.50
_UNC_REL     = 0.005
```

## 6. 从 dev SPS 到线上 SPS

记录几次线上提交的真实 SPS（用于判断 dev 与线上关系）：

| 版本 | dev SPS (proxy) | online SPS | online final |
|---|---:|---:|---:|
| E | ~50.2 | 39.467 | 79.877 |
| H (final) | 51.6332 | **40.209** | **80.078849** |

观察：dev SPS 提升约 1.43，线上 SPS 提升 0.742，**约 52% 兑现**。SPS 权重约 0.311，
因此线上 final 提升约 0.20。

## 7. 明确未使用的方法（避免误复现）

```text
- conformal / quantile calibration: NOT USED
- per-horizon sigma 校准: NOT USED (验证后 +0.001，判定为冗余)
- pinball loss: 试过，未采用
- sps surrogate loss: 试过，未采用
- reward-weighted loss: 试过，与基线逐位相同，未采用
- global scalar sigma: NOT USED
- 运行时 recalibration: NOT USED
- test-time augmentation: NOT USED
```

## 8. 代码索引

| 功能 | 文件 | 关键函数/常量 |
|---|---|---|
| 推理区间生成 | `tools/colleague_80pt/submission/submission.py` | `_with_bounds`, `UncertaintyHead3D` |
| sigma 训练 | `tools/colleague_80pt/train_head_fast.py` | `logmae` 分支 |
| 冻结特征缓存 | `tools/colleague_80pt/cache_frozen.py` | `dump`, `windows` |
| 参数扫描 | `tools/colleague_80pt/scan_bounds.py` | `fast_sps` |
| 官方 SPS 评分 | `tools/colleague_80pt/realpde_sps_scoring.py` | `aggregate_sps`, `_acc_factor`, `SIGMA_GLOBAL` |
