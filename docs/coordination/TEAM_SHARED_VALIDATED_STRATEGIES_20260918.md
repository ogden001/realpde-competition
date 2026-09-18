# RealPDE Track 1：已验证有效策略、实验设置与数据结论

> 更新时间：2026-09-18  
> 用途：团队内部同步。本文只汇总**已经有实验证据支持、值得复用的策略与结论**，并明确区分“线上已验证”“离线已验证但有条件”“仅数据分析支持”。  
> 不把尚未完成的想法、存在实验 confound 的结果，或其他同事已验证但我们尚未复现的策略冒充成自己的有效实验。

---

## 0. 一页结论

目前最值得直接复用的主线不是某一个小技巧，而是下面这套组合：

`Dense-All + P0-A + N2 + MF-CNO + Vorticity + Stage-B extra Rel + Adaptive SPS`

其中证据强度不同：

| 策略 | 当前结论 | 证据强度 | 是否进入当前线上体系 |
|---|---|---|---|
| Dense-All 全合法时间窗训练 | 明确有效，是目前最强的数据利用策略之一 | 50/16 matched + horizon 分析 + SOTA-V2 线上 | 是 |
| P0-A runtime-safe 特征 + N2 多目标 Loss | 明确有效，且长训仍持续改善 | CLEAN/competition validation + 线上历史 | 是 |
| 充分训练至平台区 | 10k 远未收敛；约 22k 后进入平台/振荡 | 30.9k 完整 validation curve | 是 |
| Stage-B 低 LR + extra Rel | 明确有效，尤其显著改善 Rel/MVPE | 50/16 matched continuation | 是 |
| Adaptive uncertainty / SPS calibration | 明确有效，可在 point prediction 不变时提升 Final | Dev + 多次 Codabench online | 是 |
| Temporal + Spatial 信息 | 数据层面明确含有增量信息 | train-only ridge residual probe | 尚不能等价为神经 fusion 已验证 |
| Mean / Fluctuation | 短中程有明显收益，长程 standalone 不稳定 | matched + long convergence | 组合内保留，独立归因未解决 |
| Vorticity supervision | 有小幅一致信号，但 standalone 长程 gate 未过 | matched long validation | 组合内保留，独立归因未解决 |
| H1 / Point residual correction + scale | Rel/MVPE 很强，但 TKE 风险明显 | frozen-CNO matched offline | 有条件有效，暂未进入线上 SOTA |

截至 2026-09-18，当前线上最佳仍使用同一个 SOTA-V2 point predictor，SPS 使用 full-specific teammate35 uncertainty head：

- Rel-L2 score：`93.816645`
- TKE score：`79.164203`
- MVPE score：`93.411176`
- Time score：`86.699342`
- SPS score：`31.961724`
- Final：`77.732796`

---

# 1. 统一实验协议

## 1.1 Frozen 50/16/16 split

本项目离线方法实验统一使用：

- 总数据：82 条 released PIV trajectories
- Train：50 trajectories
- Dev：16 trajectories
- Locked Final：16 trajectories
- 按**完整 trajectory**划分，禁止按 window 随机拆分
- seed：`20260901`
- manifest SHA-256：  
  `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`

标准 fixed-window 口径：

- `T_in = 20`
- `T_out = 20`
- `stride = 20`
- Dev 不 shuffle
- spatial subsample = 2
- pressure 不作为真实物理输入，最终预测固定为 0

窗口数：

| Pool | Windows |
|---|---:|
| Train canonical fixed | 2,052 |
| Train Dense-All | 40,488 |
| Dev fixed | 659 |
| Full 82 Dense-All | 66,755 |

如果协作者的 Dev 不是 **16 trajectories / 659 windows**，则结果不能直接与本文数字比较。

## 1.2 Scorer

统一使用 Track 1 starting kit v9：

- scorer SHA-256：  
  `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`

离线报告 raw error：

- Rel-L2 ↓
- TKE ↓
- MVPE ↓

三项均为越低越好。不要与 Codabench 上 0–100、越高越好的 subscore 混用。

## 1.3 推理阶段硬约束

最终可部署策略只依赖官方 runtime 能保证拿到的：

- Past20 `u/v`
- tensor shape
- pressure 占位通道

不把以下信息作为正式推理输入：

- Re
- AoA metadata
- physical x/y
- CFD
- hidden/private mask
- sample id 解码
- future target

---

# 2. 策略一：Dense-All 全合法时间窗训练

## 2.1 做法

原始 canonical 训练只取固定 stride=20 的 Past20→Future20 窗口，一共 2,052 个 Train windows。

Dense-All 对每条 Train trajectory 枚举所有满足 Past20 + Future20 的合法起点：

- 2,052 → **40,488 windows**
- 数据量扩大 **19.731×**
- 不生成 synthetic 数据
- 不引入新的 domain
- Dev 仍固定 659 windows，不改变评估口径

这是“把已有真实 PIV trajectory 用透”，而不是传统意义上的噪声增强。

## 2.2 结果

Dense-All @30k：

| Rel-L2 ↓ | TKE ↓ | MVPE ↓ |
|---:|---:|---:|
| **0.110092** | **0.479399** | **0.080831** |

证据不只来自 aggregate：

- matched 7.5k 对 fixed-window control：三项 aggregate 均改善
- Future20 frame-wise Rel-L2：**18/20 horizons** 更好
- Dense-All 从 7.5k 继续到 30k：frame-wise Rel-L2 **20/20 horizons** 继续改善

## 2.3 结论

**Dense-All = DEFAULT_TRAINING_PROTOCOL / ONLINE_KEEP。**

这是目前最明确、最值得同事直接复用的策略之一。后续无需重复做“Dense-All 是否有效”的 A/B，研究重点应转向训练预算、objective 和模型结构。

证据入口：

- `docs/experiments/dw01_dense_all_20260907/README_FOR_CHATGPT.md`
- `docs/training/training概要.md`

---

# 3. 策略二：P0-A runtime-safe 特征 + N2 多目标 Loss

## 3.1 P0-A

P0-A 将从 Past20 `u/v` 内部构造的 runtime-safe 信息扩展到 CNO 输入，总计 20 个 feature channels。

原则比具体代号更重要：

- 只从当前输入窗口计算
- 不使用 Re/AoA/private metadata
- 强调 temporal mean、fluctuation、recent dynamics、spatial structure 等信息
- 保持 submission-compatible

实现以仓库中的 `realpde_p0_features` / P0-A runner 为准，不建议根据文档手工重写 20 通道定义。

## 3.2 N2 Loss

当前 N2 权重：

```text
MSE   = 1.0
TKE   = 0.05
Rel   = 0.027514
MVPE  = 0.009757
```

对应：

`L = MSE + 0.05*TKE + 0.027514*Rel-L2 + 0.009757*MVPE`

## 3.3 早期 CLEAN 结果

从 official sim-only `sim_pretrain` 初始化、仅使用本地 Train50：

- batch = 8
- 约 4,100 updates / 2h
- Dev replay：

| Rel-L2 ↓ | TKE ↓ | MVPE ↓ |
|---:|---:|---:|
| 0.156106 | 0.553362 | 0.125436 |

该实验是 CLEAN 证据，但由于输入架构和训练预算与其它 baseline 不完全一致，不应把绝对差异全部归因给单个因素。

## 3.4 长训练结果：10k 远未收敛

competition-oriented `sim_real_ft` warm-start 的同一 P0-A + N2 validation 被继续训练到 30,900 updates。

关键点：

| Update | Rel-L2 | TKE | MVPE |
|---:|---:|---:|---:|
| 10,300 | 0.123734 | 0.515453 | 0.096836 |
| 26,240 | 0.112925 | 0.494840 | **0.084671** |
| 27,880 | **0.112398** | 0.496896 | 0.088587 |
| 30,340 | 0.112939 | **0.492848** | 0.088154 |

相对 10,300：

- best Rel-L2 error：约 **-9.16%**
- best TKE error：约 **-4.39%**
- best MVPE error：约 **-12.56%**

约 22k 后进入宽平台/振荡区，但直到 30.9k 都没有出现明显 overfit collapse。

## 3.5 线上观察

2026-09-03 的 all-82 P0-A + N2 @15,300 submission：

- Rel-L2 score：93.023539
- TKE score：78.355520
- MVPE score：91.894417
- SPS：11.431650
- Final：71.153839

这里最重要的经验不是 Final 低，而是：

> **P0-A + N2 能显著改善 physical prediction，尤其 TKE；但 SPS 是独立问题，如果 interval 没有一起校准，Final 会被 SPS 拖垮。**

证据入口：

- `docs/coordination/CHATGPT_HANDOFF_P0A_N2_VALIDATION_30900.md`
- `docs/coordination/CHATGPT_HANDOFF_T1_P0A_N2_FULL15300_CODABENCH.md`

---

# 4. 策略三：Stage-B 低学习率 + extra Rel-L2

这是 SOTA-V2 中非常值得复用的一步。

## 4.1 设置

SOTA-V2 Stage A：

- Dense-All
- effective batch = 8
- LR = `1e-5`
- update 1 → 30,000

Stage B：

- 从 Stage A checkpoint 继续
- LR 降至 `3e-6`
- update 30,001 → 35,000
- 在原 objective 上增加额外 Rel-L2 强化项

当前主 sweet spot：`@32,500`

## 4.2 效果

Stage A @30k：

- Rel-L2 = 0.105989
- TKE = 0.474373
- MVPE = 0.090337

Stage B @32.5k：

- Rel-L2 = **0.099935**
- TKE = **0.469293**
- MVPE = **0.075778**

相对 @30k raw error 约：

- Rel-L2：**-5.7%**
- TKE：**-1.1%**
- MVPE：**-16.1%**

这是一个很干净的 late-stage improvement，尤其 MVPE 改善幅度很大。

## 4.3 为什么做 Stage-B

逐帧分析发现误差明显集中在 late horizon。

Integrated Dev：

- @30k：h19 = 8.67% squared-error fraction
- @30k：h20 = 16.43%
- @30k：h19+h20 = **25.11%**
- @35k：h19+h20 仍占 **24.22%**

说明 Future20 尾部是结构性难点，而不是 overall 指标的小噪声。

Stage-B 的价值可以理解为：

> 主干先用大规模 Dense-All 学到总体动力学，再在较小 LR 下针对 relative / late-horizon error 做收口。

证据入口：

- `docs/sota迭代/reviews/sota_v2_integrated_50_16_20260915/README.md`

---

# 5. SOTA-V2 集成方案：组合效果已线上验证

## 5.1 50/16 recipe

`Dense-All + P0-A + MF-CNO + N2 + Vorticity + Stage-B extra Rel`

关键配置：

- warm-start：official `sim_real_ft/sim_real_cno.pth`
- warm-start SHA：`82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61`
- effective batch = 8
- Stage A：30k，LR=1e-5
- Stage B：5k，LR=3e-6
- N2：1.0 MSE + 0.05 TKE + 0.027514 Rel + 0.009757 MVPE
- Vorticity weight：`15.5385751724`

Dev curve：

| Update | Rel-L2 ↓ | TKE ↓ | MVPE ↓ |
|---:|---:|---:|---:|
| 7,500 | 0.125148 | 0.510103 | 0.108036 |
| 15,000 | 0.111479 | 0.484161 | 0.087232 |
| 25,000 | 0.106151 | 0.471133 | 0.082209 |
| 30,000 | 0.105989 | 0.474373 | 0.090337 |
| 31,000 | 0.100937 | 0.470932 | 0.078285 |
| **32,500** | **0.099935** | **0.469293** | **0.075778** |
| 35,000 | 0.099978 | 0.471441 | 0.075749 |

## 5.2 Full refit

Full released PIV：

- trajectories = 82
- Dense-All windows = 66,755
- Stage-A end = 49,461
- final update = 53,582
- checkpoint SHA：  
  `f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce`

## 5.3 Codabench

2026-09-16 SOTA-V2：

| Metric | Previous SOTA | SOTA-V2 | Delta |
|---|---:|---:|---:|
| Final | 76.694784 | **77.314446** | **+0.619662** |
| Rel-L2 | 93.434384 | **93.816645** | +0.382261 |
| TKE | 77.588799 | **79.164203** | +1.575404 |
| MVPE | 92.519563 | **93.411176** | +0.891613 |
| SPS | 29.519724 | **30.319572** | +0.799848 |

三个 physical subscore 同时上涨，因此这不是 SPS-only 提升。

**重要边界：** 这证明的是“集成 recipe 整体有效”，不能把线上 +0.62 Final 分别归因给 MF 或 Vorticity。

证据入口：

- `docs/sota迭代/reviews/sota_v2_integrated_50_16_20260915/README.md`
- `docs/sota迭代/reviews/sota_v2_full_20260916/README.md`
- `docs/sota迭代/reviews/sota_v2_adaptive_20260916/README.md`

---

# 6. 策略四：SPS 必须独立建模与校准

这一点已经被线上反复验证。

## 6.1 静态 interval 先救回 SPS

P0-A + N2 的 first full submission：

- SPS = 11.431650

后来仅使用显式 static bounds：

`half_width = 0.0075 + 0.02 * abs(prediction)`

point prediction 不变，线上：

- SPS → **27.545059**
- Final → **76.149726**

说明 SPS 与 point prediction 是相对独立的优化轴。

## 6.2 Adaptive uncertainty：线上有效

2026-09-05，在同一个 full@43,260 backbone 上：

- static SPS = 27.545059
- adaptive SPS = **29.519724**
- Final = 76.149726 → **76.694784**
- point prediction parity = 0.0

因此 learned adaptive uncertainty 是**真正有在线收益的组件**。

## 6.3 SOTA-V2 fresh head

50 Train canonical：

- 2,052 windows
- uncertainty head：`in_channels=15, hidden=32, blocks=2`
- 1,400 updates

Dev16 calibration：

- static SPS = 42.124892
- adaptive SPS = **45.070082**
- +2.945190
- selected：`floor=0.0025, mult=1.0`
- coverage ≈ 0.8559
- mean UV width ≈ 0.02358

## 6.4 当前线上最强 SPS 版本

2026-09-17 full-specific teammate35 head：

- full point predictor 不变
- head update = 1,600
- floor = 0.0025
- mult = 1.0
- point scores 完全不变
- online SPS = **31.961724**
- Final = **77.732796**

这是当前线上最佳。

## 6.5 SPS-only 微调已经接近收口

2026-09-18 又精确复制了更完整的 teammate recipe：

- 35-channel
- hidden=32 / blocks=2 / dropout=0
- masked Gaussian NLL
- seed=41
- AdamW lr=1e-3
- weight_decay=1e-5
- batch=8
- 2,000 updates
- every 200 eval
- selected @1,800
- fixed 28-row calibration grid
- floor=0.0025 / mult=1.0

线上：

- SPS = 31.899537
- Final = 77.728857

略低于 09-17 版本，因此：

> **不要继续在 seed / NLL / 1600 vs 1800 vs 2000 / floor-mult 小网格上消耗提交次数。**

剩余 SPS 差距更可能与“base prediction → residual corrector → final prediction → interval center”的结构有关，而不是 uncertainty head 本身的小参数。

证据入口：

- `docs/submission_log.md`
- `docs/sota迭代/reviews/sps_teammate_final_20260917/README.md`
- `docs/sota迭代/reviews/sps_teammate_exact_submit_20260918/README.md`

---

# 7. 数据分析结论：Temporal / Spatial 信息确实有增量价值

这是一个**信息价值实验**，不是“某个神经网络 fusion 已经成功”的证明。

## 7.1 实验

固定 PERSIST baseline，预测 residual；只用 Train50 拟合 ridge，Dev16 一次评估。

输入包：

- Raw-Control：last-frame u/v，2 dim
- Temporal：+ mean/std/delta，共 8 dim
- Spatial：+ 四个 pixel gradient，共 6 dim
- Temporal+Spatial：共 12 dim

Ridge：

- alpha = `1e-2`
- train-only normalization
- 4,202,496 train pixel rows
- 无 Dev 调参

## 7.2 结果

| Feature | Rel-L2 ↓ | TKE ↓ | MVPE ↓ |
|---|---:|---:|---:|
| Raw-Control | 0.131353 | 0.987575 | 0.130701 |
| + Temporal | 0.118340 | 0.940578 | 0.109454 |
| + Spatial | 0.130241 | 0.973685 | 0.129355 |
| **Temporal + Spatial** | **0.116346** | **0.936060** | **0.107228** |

联合包相对 Raw：

- Rel-L2：-0.015007
- TKE：-0.051515
- MVPE：-0.023473

Trajectory-macro win rate：

- Temporal：Rel/TKE/MVPE = 0.875 / 1.000 / 0.938
- Spatial：0.812 / 1.000 / 0.750
- Temporal+Spatial：**0.938 / 1.000 / 1.000**

所以：

> **Temporal 信息的价值最强，Spatial 也有独立但较弱的增量，二者联合最好。**

## 7.3 冻结 feature 定义

TEMPORAL6：

```text
mean_u_20, mean_v_20
std_u_20,  std_v_20
delta_u,   delta_v
```

其中：

`delta = last_frame - previous_frame`

SPATIAL4：

```text
du_dx_pixel
du_dy_pixel
dv_dx_pixel
dv_dy_pixel
```

采用 pixel spacing=1，内部 centered difference，边界 forward/backward。

## 7.4 冗余分析

已确认：

- `speed` 与 `abs(u)` Pearson ≈ **0.9999**
- `u2_prime_mean = std_u²`
- `v2_prime_mean = std_v²`
- `vorticity = dv_dx - du_dy` 是确定性 derived summary

因此不要把所有“有物理名字”的量都堆成独立 feature。

另外，直接把 RawSpatial8 注入一个小 residual head 已经失败，说明：

> **Feature 有信息 ≠ 任意 fusion 方法都有效。**

证据入口：

- `docs/coordination/CHATGPT_HANDOFF_FE_INCREMENTAL_PROBE.md`
- `docs/feature_engineering/feature_engineering概要.md`

---

# 8. Mean / Fluctuation：有价值，但不能单独夸大

## 8.1 matched @3000 的强信号

Direct@3000：

- 0.175829 / 0.594649 / 0.151631

MF@3000：

- **0.164327 / 0.582928 / 0.130374**

MF 相对 Direct raw-error 改善：

- Rel-L2：**6.541%**
- TKE：**1.971%**
- MVPE：**14.019%**

trajectory wins：

- Rel：16/16
- TKE：8/16
- MVPE：15/16

## 8.2 但长程收益不稳定

继续到 15k 后，优势 wash out / mixed：

相对 Direct@15k 最终约：

- Rel：+1.157%
- TKE：+0.207%
- MVPE：**-1.969%**

所以 standalone 状态是：

`WEAK_SIGNAL_PARKED`

但 MF-CNO 被保留在 SOTA-V2 集成 recipe 中，而整个集成 recipe 已线上成功。

正确表述：

> MF 有明确的早期/中程机制信号，集成后可用，但当前不能声称 SOTA-V2 的线上增益由 MF 单独贡献。

---

# 9. Vorticity supervision：有小幅一致收益，但 standalone 证据不够强

matched low-memory @12000：

Control：

- 0.130150 / 0.518528 / 0.110468

Vorticity：

- **0.125012 / 0.510238 / 0.108193**

改善：

- Rel-L2：**3.948%**
- TKE：**1.599%**
- MVPE：**2.059%**

但 late checkpoints 6000/9000/12000 的 median：

- Rel：+3.243%
- TKE：-0.564%
- MVPE：+2.059%

未通过预注册的强 gate，因此 standalone = `PARK`。

它最终被放进 SOTA-V2 integrated recipe，并随整体方案线上提升。

正确表述与 MF 相同：

> **组合内可用，独立贡献未被严格归因。**

---

# 10. Residual / Point correction：强信号，但 TKE 必须加护栏

这是很值得同事参考的“有条件有效”策略。

## 10.1 H1 frozen-CNO residual

Frozen CNO：

- Rel-L2 = 0.190821
- TKE = 0.644069
- MVPE = 0.144258

加 zero-init LOCAL3 Point residual head，1500 updates：

- Rel-L2 = **0.141102**
- TKE = 0.706906
- MVPE = **0.102705**

相对变化：

- Rel-L2：**+26.056%**
- MVPE：**+28.804%**
- TKE：**-9.756%**，即 TKE error 变坏 9.756%

这说明 residual corrector 对 pointwise bias / local error 很强，但会破坏 energy structure。

## 10.2 Train-only residual scale

不再训练网络，只扫描：

`Y = Y_CNO + alpha * residual`

alpha grid：0.0, 0.1, ..., 1.0。

alpha 只在 Train 上选择，规则：

1. TKE degradation <= 5%
2. 在满足条件的 alpha 中最大化 Rel-L2 improvement

得到：

`alpha* = 0.5`

Dev：

- Rel-L2 = 0.156427
- TKE = 0.669563
- MVPE = 0.115209

相对 CNO：

- Rel-L2：**+18.024%**
- MVPE：**+20.137%**
- TKE：-3.958%，在 5% protection line 内

## 10.3 稳定性风险

trajectory-level：

- Rel wins：16/16
- MVPE wins：16/16
- TKE wins：2/16
- TKE degradation <=5%：仅 3/16

所以不能只看 aggregate TKE gate。

结论：

> residual correction 的“Rel/MVPE 强、TKE 易坏”是非常稳定的现象。若以后回收这条线，必须把 TKE/energy protection 设计成结构的一部分，而不是训练后才看一眼 aggregate。

证据入口：

- `docs/coordination/CHATGPT_HANDOFF_HYBRID_CNO_POINT_H1.md`
- `docs/coordination/CHATGPT_HANDOFF_HYBRID_CNO_POINT_H1_SCALE.md`
- `docs/coordination/CHATGPT_HANDOFF_HYBRID_CNO_POINT_H1_SCALE_STABILITY.md`

---

# 11. 数据集画像与误差分析

## 11.1 Split 没有明显偏置

完整 input-side audit：

- Dev：9 ID / 3 Boundary / 4 OOD-like
- Final：11 ID / 1 Boundary / 4 OOD-like
- Dev/Final 在 PCA 空间与 Train 交错
- OOD-like 主要是 fluctuation / delta / gradient / vorticity / strain 的 tail combination
- 没有形成一个与 Train 分离的物理区域

结论：`SPLIT_OK`，不需要重划分。

## 11.2 AoA 分布本身是平衡的

0° / 5° / 10° / 15° / 20° trajectory count：

- Train：11 / 8 / 10 / 11 / 10
- Dev：3 / 3 / 4 / 3 / 3
- Final：3 / 3 / 4 / 3 / 3

这里仅用于 dataset audit。正式推理仍不能依赖 AoA metadata。

## 11.3 已知 exact duplicate

Past20 input 完全重复：

- Train `6300_0.h5`
- Final `7575_0.h5`

42 个有效 Past20 windows 全部 exact equal。

除该 pair 外没有第二个 <0.1 的 near-duplicate candidate。

这不改变当前 split。

证据入口：

- `docs/data/DATASET_PROFILE.md`
- `docs/data/DUPLICATE_AUDIT.md`

---

# 12. 当前不应被写成“我们已验证有效”的策略

为了避免团队信息污染，下面几项要特别说明。

## 12.1 Random Phase

旧实验存在 shuffle / sampler confound，目前状态是：

`INVALID_COMPARISON`

因此既不能写“有效”，也不能写“无效”。

Dense-All 已经解决了大部分“时间起点利用不足”的问题，而且证据更干净。

## 12.2 AoA augmentation

这是**同事已经验证、我们计划吸收/复现的策略**，不是本文项目目前已经完成验证的自有实验。

所以本文不把它列入“已验证有效”列表。

## 12.3 RawSpatial8 直接注入

虽然 Spatial 信息在 ridge probe 中有增量，但直接注入 residual head 的 RawSpatial8：

- Rel-L2/MVPE 没改善
- TKE 有 trade-off
- Gate failed

结论：不要把“Spatial feature 有信息”误写成“Spatial concat 已成功”。

## 12.4 Temporal Attention / 复杂显式时序模型

当前没有大台阶：

- Latent Temporal Conv 只有约 1%–3% 弱信号
- Temporal Attention 基本无增益
- ΔUV supervision 有 Rel/TKE trade-off

暂不作为主线。

## 12.5 CFD frozen representation transfer

official sim_pretrain frozen representation + tiny probe 甚至弱于 matched random frozen control：

- CFD rep：0.9026 / 32.0180 / 1.1509
- Random：0.7680 / 13.0809 / 0.7800

当前路线 STOP / PARKED。

---

# 13. 给协作者的复现实验建议

如果同事要与我们对齐，请至少固定这些信息：

```text
Manifest SHA:
42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347

Dev:
16 trajectories / 659 fixed windows

Scorer SHA:
a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39

Report:
Rel-L2 raw error
TKE raw error
MVPE raw error

Must also record:
init checkpoint
train window policy
effective batch
optimizer / LR
updates
checkpoint selection rule
code commit
```

建议实验结果统一写成：

```text
Experiment:
Init:
Train windows:
Budget:
Rel-L2:
TKE:
MVPE:
Matched baseline:
Delta vs baseline:
By-horizon:
Trajectory-level stability:
Notes:
```

**不要只报一个 aggregate 最优数字。** 本项目已经多次遇到：

- aggregate 看起来过 gate
- 但 trajectory-level TKE 广泛退化
- 或 h19/h20 极端拖尾

这类情况对 hidden set 泛化风险很高。

---

# 14. 最值得直接拿走的 recipe

如果只给同事五条：

1. **先用 Dense-All，把真实 PIV trajectory 的合法时间窗吃满。**
2. **P0-A + N2 是可靠的 prediction 主线，而且要充分训练，不要 5k/10k 就判断收敛。**
3. **30k 左右进入平台后，用更低 LR 的 Stage-B + extra Rel 做收口，收益很实。**
4. **SPS 独立优化。point model 再强，interval 没校准仍会丢大量 Final；adaptive uncertainty 已在线验证有效。**
5. **Residual corrector 对 Rel/MVPE 潜力很大，但必须加 TKE/energy protection；只追 pointwise error 会破坏能量。**

如果再加两条研究经验：

6. **Temporal feature 信息量 > Spatial feature 信息量；Feature Value 和 Fusion Value 必须分开验证。**
7. **按 horizon + trajectory 做分析。late horizon，尤其 h19/h20，是当前明确的结构性误差来源。**

---

# 15. 关键文档索引

- 共享 50/16 benchmark：  
  `docs/data/SHARED_DEV_50_16_BENCHMARK.md`
- 全局实验注册表：  
  `docs/track1_experiment_registry.md`
- 当前整体战略：  
  `docs/realpde整体优化概要.md`
- Dense-All / Training：  
  `docs/training/training概要.md`
- P0-A + N2 30.9k：  
  `docs/coordination/CHATGPT_HANDOFF_P0A_N2_VALIDATION_30900.md`
- SOTA-V2 50/16：  
  `docs/sota迭代/reviews/sota_v2_integrated_50_16_20260915/README.md`
- SOTA-V2 online：  
  `docs/sota迭代/reviews/sota_v2_adaptive_20260916/README.md`
- SPS 当前线上最佳：  
  `docs/sota迭代/reviews/sps_teammate_final_20260917/README.md`
- SPS exact replica：  
  `docs/sota迭代/reviews/sps_teammate_exact_submit_20260918/README.md`
- Feature incremental probe：  
  `docs/coordination/CHATGPT_HANDOFF_FE_INCREMENTAL_PROBE.md`
- H1 residual scale：  
  `docs/coordination/CHATGPT_HANDOFF_HYBRID_CNO_POINT_H1_SCALE.md`
- Dataset profile：  
  `docs/data/DATASET_PROFILE.md`
- Submission history：  
  `docs/submission_log.md`

---

## Final takeaway

目前最可靠的工程结论是：

> **更多真实时间监督（Dense-All） + runtime-safe input priors（P0-A） + multi-objective objective（N2） + 充分长训 + 低 LR Stage-B 收口，是 point prediction 的主干；Adaptive SPS 是与之并行的第二条主干。**

MF / Vorticity 可以保留在已成功的 integrated recipe 中，但不要过度做单模块归因；Residual correction 则是下一类最有潜力、同时最需要 TKE protection 的增量。
