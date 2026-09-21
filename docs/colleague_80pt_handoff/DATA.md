# DATA

## 1. 数据来源

官方提供的真实轨迹 HDF5 数据集，服务器路径：

```text
/data/p0ab_real_h5_20260830/*.h5
```

共 82 个 H5 文件，其中 1 个被显式排除：

```python
BAD_TRAIN_FILES = {"7575_0.h5"}
```

排除后可用轨迹 **81 条**（`list_h5` 会跳过该文件）。不要使用 81 条以外的额外 CFD/PIV 数据；
本方案**没有使用任何额外数据源**。

## 2. H5 结构（实测）

```text
u   (T, 64, 128) float64
v   (T, 64, 128) float64
x   (64, 128)    float64
y   (64, 128)    float64
t   (T,)         float32
aoa ()           int32
re  ()           int32
```

- `T` 随轨迹变化（实测 607、868 等）。
- `x` 方向（列，128 点）为流向，`y` 方向（行，64 点）为法向。
- `x`/`y` 为**均匀网格**：`dx = dy = 0.00171083`，`x` 范围约 `[0.00171, 0.21899]`，
  `y` 范围约 `[0.08212, 0.18990]`。
- 没有顶层 `p` 字段；压力通道在训练/提交里恒置 0。

`h5_field(handle, key)` 兼容顶层与 `measured_data/<key>` 两种布局。

## 3. 空间下采样（关键）

所有阶段统一使用 `sub_sample = 2`，实现为纯切片下采样：

```python
u = h5_field(handle, "u")[sl, ::2, ::2]   # (T, 64, 128) -> (T, 32, 64)
v = h5_field(handle, "v")[sl, ::2, ::2]
```

这不是裁剪，而是**整个 64x128 流场的 2x 均匀下采样**。因此 64x128 网格存在 4 种相位
（P00/P01/P10/P11），但最终 80 分方案**只使用 P00 = `[::2, ::2]`**。
4 相位增强属于 FUTURE_RESEARCH_NOTES，见 `EXPERIMENT_HISTORY.md`。

## 4. 时间窗口

统一参数：

```text
in_steps  = 20
out_steps = 20
total     = 40
```

各阶段 stride 不同（见 `TRAINING.md`）：

| 阶段 | stride | 每轨迹窗口数 | 说明 |
|---|---:|---:|---|
| Stage 1  CNO all81 | 1 | 约 814 | 65926 总窗口 |
| Stage 2  residual   | 20 | 约 41 | 3341 总窗口（81 轨迹） |
| Stage 3  head cache | 5 (train) / 20 (val) | — | 13202 train / 640 val |

窗口起点：`starts = range(0, T - 40 + 1, stride)`。
输入为 `full[:20]`，target 为 `full[20:40]`。

## 5. Train / dev split

```python
paths = list_h5(real_root, BAD_TRAIN_FILES)          # 81 条
train_paths, val_paths = split_paths(paths, val_fraction=0.2, seed=41)
```

`split_paths` 用 `np.random.default_rng(seed=41)` 打乱轨迹顺序，取前 20% 作为 dev：

```text
train_trajectories = 65
val_trajectories   = 16
```

但 Stage 2 / Stage 3 的**最终训练都使用全部 81 条轨迹**（`--train-on-all` 或
`cache_frozen.py --all-data`）：

```text
fit_trajectories = 81
```

dev（16 条 / 640 窗口）只用于：

- 残差头训练过程中的 alpha / bound 选择（`evaluate_alphas`）
- SPS 参数网格扫描（`scan_bounds.py`）
- 探针头训练过程中的评估（`eval_log.json`）

`cache_frozen.py` 的 val 固定使用同样的 16 条轨迹、`stride=20`、P00 相位。

## 6. Normalization

数据加载层（`H5WindowDataset`）**没有做任何显式归一化**：直接读取原始 `u`、`v`，
转 float32，stack 成 `(T, H, W, 3)`，压力通道置 0。

```text
input normalization: NONE at the dataset level
target normalization: NONE
```

backbone 内部可能包含自身的归一化/缩放层（见 `rpde_baselines/cno.py`），本 handoff 不做修改。
如果复现时出现尺度差异，优先核对官方 `sim_real_cno.pth` 的训练约定。

## 7. Sampling

- Stage 1：`DataLoader(shuffle=True, num_workers=0, drop_last=True)`。
- Stage 2：`DataLoader(shuffle=True, num_workers=2, drop_last=True)`。
- Stage 3（head）：基于预生成 `frozen_{tr,va}_{x,b,p,y}.npy` memmap，float16，shuffle=True。

训练循环在每个 epoch 用完后重新 `iter(loader)`（隐式多 epoch）。

## 8. Augmentation

**最终 80 分方案没有使用任何数据增强。** 逐项确认：

| Augmentation | 使用情况 | 说明 |
|---|---|---|
| AoA / 攻角旋转 | **NOT USED** | 未实现、未验证；metadata `aoa/re` 不作为输入 |
| 随机裁剪 | NOT USED | 空间处理是固定 P00 下采样 |
| 随机相位下采样 | **NOT USED in final** | P00/P01/P10/P11 对照实验进行中，未进入 80 分包 |
| 时间抖动 | NOT USED | 窗口起点固定 stride |
| 噪声 / dropout | NOT USED | dropout=0.0 |
| flip / rotation | NOT USED | — |
| 额外 CFD / PIV 数据 | NOT USED | — |

### AoA augmentation 详细说明（按 handoff 模板要求）

```text
AoA augmentation

probability: N/A (not implemented)
angle distribution: N/A
angle range: N/A
coordinate rotation formula: N/A
input transform: N/A
target transform: N/A
u/v transform: N/A
interpolation: N/A
boundary handling: N/A
random seed: N/A
consistency train/dev/inference: N/A
```

结论：本方案不包含 AoA 增强。若后续要做，必须先证明 dev SPS 提升，并保证 input/target 同步旋转、
边界条件不被破坏。

## 9. 数据不可用项

用户/团队曾提到“有一个 H5 数据不可用”。实际代码中对应：

```text
BAD_TRAIN_FILES = {"7575_0.h5"}
```

三个训练路径（`final_all81.py`、`residual_multi.py`、`cache_frozen.py` 经 `list_h5`）都会排除它。
