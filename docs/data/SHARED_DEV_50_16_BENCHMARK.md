# Track 1 共享 50/16 Dev 基准与关键实验结果

> 用途：与并行参赛同事对齐同一个 Dev 数据集、评分口径和当前参考结果。本文只汇总可用于共同复现实验的冻结协议与少量高价值结果，不替代完整实验注册表。
>
> 状态：`FROZEN_SHARED_DEV_REFERENCE`  
> 更新时间：2026-09-16

## 1. 先统一口径

### 1.1 冻结数据划分

当前所有 Track 1 日常离线实验继续使用同一份冻结 manifest：

- 数据总池：82 条本地 PIV trajectories；**按完整 trajectory 划分，不按 window 随机拆分**。
- Split：`50 Train / 16 Dev / 16 locked-final`。
- 日常训练与模型选择：只访问 `50 Train + 16 Dev`；locked-final 不参与调参、early stop 或架构选择。
- Seed：`20260901`。
- Manifest：`artifacts/loss_optimization_v9_20260901_run1/evidence/manifests/id_seed20260901.json`。
- Manifest SHA-256：`42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`。

**不要重新执行一次随机 split 来“复现 50/16”。** 对齐的关键是复用上述 frozen membership。即便随机种子相同，只要候选文件集合、排序或 split 实现不同，也可能得到另一组数据。

### 1.2 16 条 Dev trajectory membership

下面 16 条就是当前共同 Dev 集。协作者至少应先核对这份 membership：

```text
10125_0.h5
3750_0.h5
26700_0.h5
13950_5.h5
20325_5.h5
8850_5.h5
8850_10.h5
11400_10.h5
24150_10.h5
16500_10.h5
22875_15.h5
11400_15.h5
25425_15.h5
20325_20.h5
8850_20.h5
3750_20.h5
```

数据画像中，这 16 条 Dev 按 Train-only 统计被标记为：`9 ID / 3 BOUNDARY / 4 OOD_LIKE`。这只是输入分布诊断，不用于选模或重新划分。

### 1.3 Dev window / tensor 协议

当前 SOTA-V2 runner 的 Dev 评估协议：

- `T_in = 20`
- `T_out = 20`
- `stride = 20`
- `window_mode = fixed`
- 起点从 `start=0` 开始，合法条件 `start + 40 <= frames`
- `sub_sample = 2`
- `include_pressure = false`；正式预测中的 pressure 通道固定为 0
- Dev 不 shuffle

冻结 16 Dev 最终产生 **659 个 scored windows**。

作为一致性检查：

- canonical Train fixed-window pool：`2052` windows
- Dense-All Train pool：`40488` windows
- Dev fixed-window pool：`659` windows

如果这三个数字对不上，应先停止模型比较，排查数据或窗口实现。

### 1.4 评分口径

统一使用 Track 1 starting kit v9 官方 scorer：

- scorer SHA-256：`a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`
- 本文报告的是 Dev **raw errors**：`Rel-L2 / TKE / MVPE`
- **三项均为越低越好**

注意：这里的 raw error 不能和 Codabench 上 0–100 的 `rel_l2_score / tke_score / mvpe_score` 直接混为一谈；后者是变换后的 leaderboard subscore，越高越好。

---

## 2. 当前共同 Dev SOTA

当前用于全量 refit 映射的 SOTA-V2 50/16 配方为：

`Dense-All + P0-A + MF-CNO + N2 + Vorticity + Stage-B extra Rel`

主要训练配置：

- warm-start：官方 `sim_real_ft/sim_real_cno.pth`
- warm-start SHA-256：`82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61`
- effective batch：8
- Stage A：update `1..30000`，LR `1e-5`
- Stage B：update `30001..35000`，LR `3e-6`，额外增加一份 `0.027514 * Rel-L2`
- N2：`MSE=1.0, TKE=0.05, Rel=0.027514, MVPE=0.009757`
- Vorticity weight：`15.5385751724`

### 2.1 SOTA-V2 Dev curve

| Update | Stage | Rel-L2 ↓ | TKE ↓ | MVPE ↓ | 说明 |
|---:|:---:|---:|---:|---:|---|
| 7,500 | A | 0.125148 | 0.510103 | 0.108036 | early |
| 15,000 | A | 0.111479 | 0.484161 | 0.087232 | strong |
| 20,000 | A | 0.110924 | 0.485559 | 0.091191 | MVPE 回摆 |
| 25,000 | A | 0.106151 | 0.471133 | 0.082209 | strong |
| 30,000 | A | 0.105989 | 0.474373 | 0.090337 | Stage-A end |
| 31,000 | B | 0.100937 | 0.470932 | 0.078285 | Stage-B gain |
| **32,500** | **B** | **0.099935** | **0.469293** | **0.075778** | **当前 primary Dev sweet spot** |
| 35,000 | B | 0.099978 | 0.471441 | **0.075749** | MVPE 略低，但 Rel/TKE 略差 |

因此，与同事共享时建议把 **SOTA-V2 @32,500** 作为当前统一 benchmark：

> **Rel-L2 = 0.099935 / TKE = 0.469293 / MVPE = 0.075778**

项目当前 full-data refit 的 primary checkpoint 训练强度也是由 Dev `32.5k` sweet spot 按 Dense epoch 映射后预先冻结，而不是使用 35k 继续挑点。

相对两个历史内部 anchor，SOTA-V2@32.5k 的 raw error 改善为：

| Reference | Rel-L2 改善 | TKE 改善 | MVPE 改善 |
|---|---:|---:|---:|
| historical balanced anchor `0.112925 / 0.494840 / 0.084671` | +11.50% | +5.16% | +10.50% |
| Dense-All@30k `0.110092 / 0.479399 / 0.080831` | +9.23% | +2.11% | +6.25% |

这里的百分比只是 performance delta，不代表可以把 SOTA-V2 的总收益归因给某一个组件。

---

## 3. 同一 50/16 Dev 上值得参考的关键实验

下面只列对后续研究最有信息量的实验。**不同 cleanliness、初始化、训练预算的行可以作为性能参考，但不能直接当作单变量因果 A/B。** 真正判断某个策略是否有效，应优先看其 matched control。

| 实验 / checkpoint | Cleanliness / 对照 | Rel-L2 ↓ | TKE ↓ | MVPE ↓ | 关键结论 |
|---|---|---:|---:|---:|---|
| Historical E0 @7498 | `OFFICIAL_WARM_START`，`sim_real_ft`，E0 loss | 0.168923 | 0.538475 | 0.136146 | 早期 competition-oriented 历史参考，不作为 clean 方法基线 |
| P0-A + N2 @约4100 | `CLEAN`，`sim_pretrain` | 0.156106 | 0.553362 | 0.125436 | 证明 P0-A + N2 路线有价值；训练预算/架构不同，禁止跨 family 做因果归因 |
| DW-01 Dense-All @30k | `CLEAN`，`sim_pretrain`，P0-A+N2 | **0.110092** | **0.479399** | **0.080831** | 强信号。训练窗口从 2052 扩到 40488（19.731×）；后来进入 SOTA-V2 集成配方 |
| MF-01 @1500 | matched Direct control | 0.188409 | 0.645156 | 0.160552 | 对照 Direct `0.193675/0.633786/0.165178`：Rel/MVPE 改善约 2.72%/2.80%，TKE 恶化约 1.79%，standalone `NO_GO` |
| Vorticity V1 @12000 | matched C0 low-memory | **0.125012** | **0.510238** | **0.108193** | 对照 C0 `0.130150/0.518528/0.110468`：+3.95%/+1.60%/+2.06%；long-run gate 因 MVPE 稳定性不足而 `PARK` |
| SOTA-V2 @32500 | `OFFICIAL_WARM_START` integrated | **0.099935** | **0.469293** | **0.075778** | 当前共同 Dev benchmark / primary sweet spot |

### 3.1 Dense-All：当前最明确的数据训练策略信号之一

DW-01 唯一改变 train window pool：

- canonical：2052 windows
- Dense-All：40488 windows，`19.731×`
- Dev 始终固定为同一 659 windows

DW-01@30k 达到 `0.110092 / 0.479399 / 0.080831`。逐 horizon replay 显示收益不是只来自早期帧：从 7.5k 到 30k，20/20 horizon 的 frame Rel-L2 都改善；h19/h20 尾部误差也明显下降。

重要限制：DW-01 与部分 historical long-run reference 的初始化 provenance 不一致，因此不能仅靠跨 run 数值断言“Dense-All 单独贡献了多少”。但 Dense-All 后续已在 competition-oriented SOTA-V2 集成中保留。

### 3.2 Vorticity Supervision：真实小幅信号，但 standalone 不够强

严格 matched long-run @12000：

- Control：`0.130150 / 0.518528 / 0.110468`
- Vorticity：`0.125012 / 0.510238 / 0.108193`
- 改善：Rel `+3.948%`，TKE `+1.599%`，MVPE `+2.059%`

但 late checkpoints `6000/9000/12000` 的 median MVPE improvement 只有 `+2.059%`，低于预注册 `>=4%` gate；因此 standalone 方向结论是 `PARK`，不是独立 SOTA merge candidate。

SOTA-V2 中虽然仍包含固定 Vorticity supervision，但应理解为**集成配方的一部分**，不能把 SOTA-V2 总收益反向归因给 Vorticity。

### 3.3 Mean / Fluctuation decomposition：standalone 混合信号

MF-01@1500 相对 matched Direct@1500：

- Rel-L2：改善约 `2.72%`
- TKE：恶化约 `1.79%`
- MVPE：改善约 `2.80%`

因此 standalone MF-01 当时判定 `NO_GO`。后续 SOTA-V2 使用 MF-CNO 是集成验证后的 recipe，不能把早期 MF-01 的单因素结论和最终 integrated result 混成一个结论。

### 3.4 Structured Temporal Dynamics：大方向已收口

R2@3000 matched screening：

| Arm | Rel-L2 ↓ | TKE ↓ | MVPE ↓ | 结论 |
|---|---:|---:|---:|---|
| C0 Direct | 0.161343 | 0.557971 | 0.131496 | control |
| V1 Vorticity | 0.156023 | 0.555572 | 0.121585 | short-run promising |
| A1 Latent Temporal Conv | 0.159602 | 0.556817 | 0.127943 | 约 1%–3% weak signal，parked |
| A2 Latent Temporal Attention | 0.161400 | 0.559184 | 0.131574 | 基本无增益，NO_GO |

总体结论：当前没有证据表明“缺少显式 Future20 temporal architecture”是 CNO 的主要瓶颈，不继续投入 Temporal Transformer / SSM / autoregressive 深挖。

### 3.5 Random Phase：旧结果不能作为有效 A/B

旧 RW-01 random-phase run 在 @3000 显著差于 RW-00，但事后发现 RW-00 使用 global shuffle，而旧 RW-01 的 sampler 以 trajectory/time block 顺序输出，存在 **shuffle confound**。

因此旧 RW-01 数值只保留为执行证据，状态为：

`INVALID_COMPARISON / SHUFFLE_CONFOUND`

不要向协作者转述成“Random Phase 已经验证无效”。如果未来重新研究该方向，必须重新做 matched shuffle 对照。

---

## 4. 建议双方以后怎样报实验

为了让两个人的实验能直接拼到一张表里，建议每个候选至少报告：

1. manifest SHA；
2. 16 Dev membership 是否完全一致；
3. Dev window count 是否为 `659`；
4. initialization checkpoint / SHA；
5. model / feature / loss；
6. optimizer、LR、effective batch、seed、updates；
7. `Rel-L2 / TKE / MVPE` raw errors；
8. 若是 matched A/B，再报告相对 improvement；
9. 最好补充 16 条 trajectory 的 win count，避免 aggregate 被少数 case 主导；
10. 明确 `CLEAN` 还是 `OFFICIAL_WARM_START`，不要跨 family 做因果结论。

最简共享结果格式可以直接写成：

```text
Experiment:
Manifest SHA: 42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347
Dev: 16 trajectories / 659 windows
Init:
Recipe:
Budget:
Rel-L2:
TKE:
MVPE:
Matched baseline:
Delta vs baseline:
Trajectory wins (Rel/TKE/MVPE):
Notes:
```

如果只是快速判断一个新方案是否值得继续，可同时给出相对当前 SOTA-V2@32.5k 的**性能差距**；但若要证明某个技术因素有效，必须使用相同 family 的 matched control，而不是强行和 SOTA-V2 做单变量归因。

---

## 5. 关键证据入口

- 数据划分 / Dev membership：[`DATASET_PROFILE.md`](DATASET_PROFILE.md)
- 数据侧总体规则：[`data概要.md`](data概要.md)
- 全局实验注册表：[`../track1_experiment_registry.md`](../track1_experiment_registry.md)
- Dense-All：[`../experiments/dw01_dense_all_20260907/README_FOR_CHATGPT.md`](../experiments/dw01_dense_all_20260907/README_FOR_CHATGPT.md)
- Structured Temporal / Vorticity closeout：[`../modeling/structured_temporal_dynamics_closeout.md`](../modeling/structured_temporal_dynamics_closeout.md)
- Vorticity long-run handoff：[`../coordination/CHATGPT_HANDOFF_VORTICITY_LOWMEM_FINAL.md`](../coordination/CHATGPT_HANDOFF_VORTICITY_LOWMEM_FINAL.md)
- Random Phase historical review：[`../experiments/rw00_rw01_random_phase_20260907/README_FOR_CHATGPT.md`](../experiments/rw00_rw01_random_phase_20260907/README_FOR_CHATGPT.md)
- SOTA-V2 50/16：[`../sota迭代/reviews/sota_v2_integrated_50_16_20260915/README.md`](../sota迭代/reviews/sota_v2_integrated_50_16_20260915/README.md)
- SOTA-V2 runner：[`../../tools/realpde_sota_v2_integrated.py`](../../tools/realpde_sota_v2_integrated.py)

## 6. 一句话版本

> **共同 Dev = frozen manifest 的 16 trajectories / 659 fixed windows；当前共同性能锚点 = SOTA-V2@32.5k：Rel-L2 `0.099935` / TKE `0.469293` / MVPE `0.075778`。**
