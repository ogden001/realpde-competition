# Colleague SOTA 增量实验系统诊断

状态：`REVIEW_REQUIRED`。本报告只使用恒源云 `I2b421f852d05501958` 上已有资产做 inference/replay；没有训练、访问 locked-final/private、访问 Codabench、打包或上传 submission。

## 协议与资产

- run：`full_ef9c54f`，Dev16，640 windows，train/dev overlap 按原实验设计保留，因此不是 clean generalization estimate。
- 代码基线：`582b420637779156d2883669656e500ca3ac8cf3`；指定测试 21 passed，py_compile 通过。
- 输出：`/hy-tmp/realpde_runs/colleague_diagnostics_20260922/replay_final`；本地副本见同目录下的 CSV/JSON/NPZ。
- 当前同事 80 分原始 residual（`current_80pt_residual`）：checkpoint SHA `909fdc7f8a6a42335e4ea4ce7471fc9a507135e9a29a4927bc7b7318be9c85b2`。
- 原训练目标继续训练（TKE 0.06，固定 phase）：SHA `e26afa8d472fa2fe87779099d33ee237d3d70da30ce2bc5ba93c862a372fdd22`。
- 温和 TKE（0.09）：SHA `9810776f8d7dc386cc97aeb4758b56519fd1d1b0490b72e981b1ab1564a46f9e`。
- 较强 TKE（0.12）：SHA `aae2e2d7ccfdafc819e1fd930d8d9cab4e3137f36d44a2b2c4f33829193f7bda`。
- 随机时间起点（TKE 0.06）：SHA `ae66081b8b144f6b484e315d22ed1e50fe7a3c517a79c771c690d840e734f2e4`。
- `longtrain_h96x16`：`MISSING`。当前恒源云只有 `h96x16 v2` 的 summary/config，没有 checkpoint；未从其他机器传输。

## 总体与物理解剖

| 中文标签 | Rel-L2 | TKE | MVPE | mean-field Rel | fluctuation Rel | TKE energy ratio |
|---|---:|---:|---:|---:|---:|---:|
| 当前 80 分 residual | 0.080420 | 0.449158 | 0.071117 | 0.044050 | 0.744466 | 0.669431 |
| 原目标继续 5000 step | 0.080241 | 0.450366 | 0.071028 | 0.043984 | 0.742099 | 0.664581 |
| TKE weight 0.09 | 0.081179 | 0.435177 | 0.071261 | 0.044167 | 0.754063 | 0.705420 |
| TKE weight 0.12 | 0.082059 | 0.425257 | 0.071522 | 0.044346 | 0.764290 | 0.736455 |
| 随机时间起点 | 0.080721 | 0.463794 | 0.072026 | 0.044491 | 0.743918 | 0.651144 |

相对当前 residual 的 base→corrected 变化为：Rel-L2 `0.097971→0.080420`，mean-field Rel `0.056796→0.044050`，fluctuation Rel `0.890760→0.744466`，TKE field Rel `0.700314→0.449158`，energy ratio `0.453122→0.669431`。因此 residual 不是只改平均流场；均值和动态波动均受益，但预测能量仍系统性偏低于 1。

TKE reweighting 的收益是“能量幅值/统计量修正”，不是 fluctuation reconstruction 的全面改善：0.09/0.12 的 TKE raw 相对当前分别下降 `3.11%/5.32%`，但 fluctuation Rel 分别上升 `1.29%/2.66%`，Rel-L2 分别上升 `0.94%/2.04%`。这与逐 trajectory 结果一致：两者 TKE 都是 `16/16` 条改善，但 Rel-L2 和 fluctuation 都是 `0/16` 条改善。

## Future1–Future20

相对当前 residual，四段 horizon 的平均变化（百分比，负数更好）：

| 中文标签 | Future 1–5 Rel / TKE | Future 6–10 Rel / TKE | Future 11–15 Rel / TKE | Future 16–20 Rel / TKE |
|---|---:|---:|---:|---:|
| 原目标继续 | -0.487 / -0.822 | -0.261 / -0.101 | -0.198 / +0.041 | -0.150 / -0.034 |
| TKE 0.09 | +2.949 / +0.882 | +0.876 / -0.170 | +0.590 / -0.381 | +0.769 / -0.619 |
| TKE 0.12 | +5.907 / +1.884 | +1.897 / -0.185 | +1.299 / -0.614 | +1.736 / -1.044 |
| 随机时间起点 | -0.310 / +0.048 | +0.336 / +0.636 | +0.593 / +0.969 | +0.424 / +0.949 |

结论：TKE 0.09/0.12 的 TKE 改善在前 1–5 帧仍伴随 Rel 恶化，之后 TKE-contribution 改善而 Rel 恶化持续；不存在稳定的“同段同时改善 Rel 和 TKE”。随机 phase 只在最早 horizon 有微小 Rel 优势，从 Future6 起 Rel/TKE 均转差，后半段最明显。原目标继续训练的 Rel 改善贯穿全部 20 帧，主要集中在前 5 帧。

## Residual 与空间诊断

当前 residual 的 correction help fraction 为 `0.7159`、hurt fraction `0.2841`；TKE 0.09/0.12 的 help fraction 降至 `0.7098/0.7027`，delta RMS 升至 `0.007154/0.007423`，说明更强 TKE 权重以更大的修正幅度换取能量误差下降，并扩大逐点误差代价。

所有候选的最大 velocity/fluctuation/TKE 空间误差都集中在相近的翼型/尾迹区域，而不是出现新的全场失稳。TKE 0.09/0.12 的平均 TKE 绝对误差相对当前分别下降约 `3.6%/6.5%`，但平均 velocity RMSE 上升约 `0.9%/1.9%`；随机 phase 的平均 TKE 绝对误差上升约 `2.7%`。

## 决策

- `original_loss_continuation`：小而一致的 Rel/mean/fluctuation 改善，但 TKE 微弱恶化；不足以改变主线。
- `tke_weight_0p09`：`NO_GO`；TKE 改善真实且普遍，但 Rel guard 与 fluctuation 保护失败。
- `tke_weight_0p12`：`NO_GO`；TKE 改善更强，但 Rel/fluctuation 代价更大。
- `random_temporal_phase`：`NO_GO`；从 Future6 起出现 broad late-horizon degradation。
- 不建议基于本轮结果单独继续 TKE 权重扫描或随机 phase 长训。若要推进，下一项最小区分实验应是“保留 temporal mean、只对 fluctuation 做受限能量校正”，并同时看 Rel/TKE/MVPE，而不是继续增大 TKE 权重。

## 归档说明

- `comparison_by_horizon.csv` 保留 5 个模型 × Future1–Future20 的 100 行原始逐帧结果，并包含相对同事当前 80 分原始残差模型的变化百分比。
- 每个模型目录保留 16 条 trajectory、640 个 trajectory×horizon 记录及原始 `spatial_maps.npz`；每个 NPZ 约 52 KB，已纳入 Git。
- `training_progress.csv` 的 step 1000/2000/3000/4000/5000 来自恒源云原始 `eval_step_*.json` 的 `alpha=1.0` 记录；step 0 是共享起始 checkpoint 的确定性 replay，`final_est` 按原始数据缺失时留空。
- `h96x16@73500` 在当前恒源云只有 summary/config，没有 checkpoint，manifest 中记录为 `MISSING`。
