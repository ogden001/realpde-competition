# RealPDE Track 1 — CFD / PIV Data Inventory

状态：`COMPLETED`。本报告只做正式 `train_sim` / `train_real` 归档盘点和 phase-free 描述统计；没有训练、没有读取 locked-final Future20、没有 Codabench。

## 1. 数据范围与读取方式

- 数据：`data/train_sim.tar.gz`（正式 competition CFD）和 `data/train_real.tar.gz`（正式 competition PIV）。
- 读取：tar 流式读取 HDF5，未将归档完整解压到磁盘。
- CFD 100 条 trajectory；PIV 82 条 trajectory。
- 文件名条件键为 `<Re>_<AoA>.h5`。CFD/PIV 不按 trajectory 视为唯一配对；本轮只用条件键筛选共同 condition。
- 分析分辨率为 raw `64×128` 的 `::2` 下采样，即 `32×64`；descriptor 只使用 `u/v`。

## 2. 逐条结构

| dataset | trajectory | u/v shape | p | frame 数 | dt | duration |
|---|---:|---|---|---:|---:|---:|
| CFD | 100 | `(1000,64,128)` | `(1000,64,128)`，float32 | 1000（全体） | `0.02002002002` | 20.00 s |
| PIV | 82 | `(T,64,128)` | 不存在 | 282 / 492 / 607 / 868 | `0.01999998093` | 5.62 / 9.84 / 12.14 / 17.34 s |

PIV frame 分布为：79 条 `T=868`，以及 `24150_20:282`、`21600_5:492`、`10125_0:607`。全部 trajectory 的 raw 空间网格均为 `64×128`，下采样后为 `32×64`。

## 3. 空间与时间分辨率

- raw `dx≈0.00171085`、`|dy|≈0.00171083`；32×64 后 `dx≈0.00342166`、`|dy|≈0.00342166`，坐标单位沿用 HDF5 `x/y`，未擅自标注物理单位。
- CFD 起始时间为 `0.08`，末时刻 `20.08`；PIV 起始时间为 `0.08`，常规末时刻 `17.42`。
- CFD 与 PIV 的 dt 接近但不完全相同；本轮 temporal descriptor 使用各自记录的时间向量/各自 dt，不进行逐帧 phase 对齐。

## 4. Re / AoA coverage

- AoA：两域均覆盖 `0°, 5°, 10°, 15°, 20°`。
- CFD Re：`3750, 5025, 6300, 7575, 8850, 10125, 11400, 12675, 13950, 15225, 16500, 17775, 19050, 20325, 21600, 22875, 24150, 25425, 26700, 27975`。
- PIV Re：同一序列，但没有 `15225` 和 `27975`。
- Condition intersection：`82` 个。
- CFD-only：`18` 个：
  `3750_15`、`15225_{0,5,10,15,20}`、`17775_0`、`22875_{5,20}`、`24150_5`、`25425_5`、`26700_{5,20}`、`27975_{0,5,10,15,20}`。
- PIV-only：`0` 个。

## 5. 可构造窗口数

按完整 trajectory、`Past20→Future20`、要求 `start+40≤T`：

| dataset | stride=20 | stride=1 |
|---|---:|---:|
| CFD | 4,900 | 96,100 |
| PIV | 3,383 | 66,755 |

CFD/PIV 窗口规模比为 `1.448×`（stride20）和 `1.440×`（stride1）；trajectory 数量比为 `1.22×`。CFD 的新增量主要是更长轨迹和 18 个 CFD-only conditions，而不是 PIV 的同条件重复观测。

## 6. 初步价值判断

- `CFD_DATA_HIGH_VALUE`：条件覆盖和 phase-free representation/OOD 资产。CFD 提供 18 个 PIV 未覆盖 condition，以及每条更长的 1000-frame dynamics。
- `CFD_DATA_MODERATE_VALUE`：作为经过明确尺度/观测退化校准后的 mean-flow 或辅助表征 pretrain 数据。直接 raw-unit 混用不成立。
- `CFD_DATA_LOW_VALUE`：未经校准的 direct PIV forecast transfer，尤其是 TKE/高频 temporal transfer；B 审计显示这些量的跨域差异远大于 PIV 邻近条件 variation。

## 7. 机器证据与复现

机器输出：`artifacts/sim2real_round1/{inventory.csv,pairs.csv,piv_neighbor_variation.csv,summary.json}`。

本轮 descriptor 统计在同一分析命令中完成，使用 NumPy FFT fallback；本机 SciPy/NumPy ABI 不兼容，因此没有假装使用 Welch。详细 domain-gap 解释见 `CHATGPT_HANDOFF_SIM2REAL_DOMAIN_GAP.md`。

限制：PIV 没有同一 Re/AoA 的 exact replicate，因此 PIV↔PIV 参考是同 AoA 最近 Re 的邻近-condition variation，不是纯自然重复实验方差。
