# RealPDE Track 1 整体优化概要

> 本文只维护 **Track 1 当前战略地图、方向状态和下一步优先级**。具体实验过程、配置和指标统一回溯到各方向概要、`track1_experiment_registry.md`、SOTA review 和 coordination handoff。

- Deadline：**2026-09-28**
- 当前阶段：**SOTA 收口 / 提交冲刺期**。
- 当前线上 SOTA：**Dense-All + P0-A + MF-CNO + N2 + vorticity + Stage-B + full@53582 + fresh Adaptive Uncertainty Head@1400**。
- Codabench Final：**77.314446**。
- 线上子分：Rel-L2 `93.816645` / TKE `79.164203` / MVPE `93.411176` / Time `86.898836` / SPS `30.319572`。
- 相对上一线上 SOTA `76.694784`：Final `+0.619662`；Rel-L2 `+0.382261`；TKE `+1.575404`；MVPE `+0.891613`；SPS `+0.799848`；Time `-0.167810`。

---

## 0. 2026-09-24：科研基线重置

从 2026-09-24 起，项目明确分成两套 anchor：

### A. Predictive Research Anchor

统一使用：

`REALPDE_CLEAN_BASELINE_V1`

核心协议：

- 81 usable released real-PIV trajectories；
- Train51 / Seen-Dev12 / unseen-AoA10 Holdout18；
- Train/Dev/Holdout trajectory-level disjoint；
- Train stride=1，枚举所有合法 Past20→Future20 temporal windows；
- Dev/Holdout stride=20；
- Stage1 = colleague-80 CNO recipe；
- Stage2 = frozen-backbone ResidualCorrector3D h96/b2；
- checkpoint selection 只看 Rel-L2/TKE/MVPE 的 point score；
- SPS/runtime 不参与科研选模；
- AoA10 Holdout 只在方案训练完成后做诊断。

首个完整 run：

- Stage1 best@8000：Rel-L2 `0.099606` / TKE `0.778031` / MVPE `0.080290` / point_score `87.7966`；
- Stage2 best@37000：Rel-L2 `0.078363` / TKE `0.507732` / MVPE `0.064408` / point_score `90.9543`；
- unseen AoA10：Residual 将 Rel-L2 `0.096635 → 0.085567`、TKE `0.714621 → 0.523197`，MVPE `0.100787 → 0.101422`。

详细分析：

`docs/clean_baseline_v1/BASELINE_ANALYSIS_20260924.md`

### B. Online Submission Anchor

线上 SOTA / Codabench 结果继续独立维护，用于最终 merge、full-data refit、SPS、runtime 和 submission 判断。

**两套 anchor 不可混用。**

过去几天在 all81/all82、train/dev overlap 或其他非 matched protocol 下完成的实验：

- 结果保留；
- 仍可作为机制线索和方向优先级依据；
- 不再作为 clean generalization 结论；
- 若要进入后续主线，必须重新对比 `REALPDE_CLEAN_BASELINE_V1`。

因此从本节开始，本文旧的 50/16、SOTA-V2 等离线数字应理解为历史/competition evidence，不再是默认科研 control。

---

## 1. 默认阅读顺序

新开一级研究会话时先读：

1. `docs/realpde整体优化概要.md`
2. `docs/clean_baseline_v1/README.md`
3. `docs/clean_baseline_v1/BASELINE_ANALYSIS_20260924.md`
4. `docs/sota迭代/README.md`
3. `docs/submission_log.md`
4. `docs/sota迭代/reviews/sota_v2_adaptive_20260916/README.md`
5. 对应方向 `*概要.md` / closeout
6. `docs/track1_experiment_registry.md` 与相关 coordination handoff

默认科研对照已经切换为 `REALPDE_CLEAN_BASELINE_V1`；线上提交仍使用最新 Codabench SOTA 作为 submission anchor。任何后续实验必须先声明自己是在 Research 轨还是 Submission 轨。

---

## 2. 研发与资源原则

### 2.1 60 分原则

一级方向优先回答：

> **有没有大台阶？值不值得继续投资？**

默认先做 bounded screening，不在弱信号上持续扫参。

方向状态：

- `ONLINE_KEEP`：已通过正式 Codabench 验证，进入当前 SOTA recipe；
- `PROMISING`：有明显工程收益，等待合并时机；
- `WEAK_SIGNAL / PARKED`：有小幅或冲突信号，暂不继续；
- `NO_GO / STOP`：当前实现路线停止；
- `CLOSED`：该一级方向已完成本阶段探索。

### 2.2 SOTA Merge 门槛

完整 merge 成本高，任何下一轮 merge 仍必须先做 `Merge Worthiness Review`：

- `MERGE_WORTHY`：存在大台阶变量，或多个证据强且兼容的增量；
- `SKIP_MERGE`：只有孤立小收益、weak signal、线上迁移不确定，或不值得消耗完整 50/16 → full-data → package → Codabench 周期。

当前新的比较基线是 Final `77.314446`，因此过去相对弱 backbone 的小收益需要重新审视，不能机械沿用旧百分比。

---

## 3. 当前战略地图

| 一级方向 | 当前结论 | 状态 | 优先级 |
|---|---|---|---|
| **Data Regime / Dense-All** | 使用全部合法 Past20→Future20 temporal starts 明显扩大 PIV exposure；50-train Dense-All `40488` windows，all-82 `66755` windows。已进入 SOTA-V2 full refit并线上提升。 | **ONLINE_KEEP** | SOTA CORE |
| **P0-A Feature Engineering** | runtime-safe 20-channel P0-A 已稳定进入当前线上 SOTA。 | **ONLINE_KEEP** | SOTA CORE |
| **Mean / Fluctuation** | 单独 long-convergence 证据曾 mixed/wash-out，但在 Dense-All + N2 + vorticity + Stage-B integrated recipe 中随整体方案线上成功。当前只能确认“组合内可用”，不能独立归因其收益。 | **INTEGRATED_KEEP / ATTRIBUTION_UNRESOLVED** | SOTA CORE |
| **Vorticity Supervision** | 单独 long paired gate 曾 mixed；但已作为 SOTA-V2 integrated recipe 的组成部分并通过线上整体验证。不能将线上增益单独归因给 vorticity。 | **INTEGRATED_KEEP / ATTRIBUTION_UNRESOLVED** | SOTA CORE |
| **Loss / Objective** | N2 继续作为主 objective；Stage-B 低 LR + extra Rel 在 50/16 上带来关键 late improvement，并进入线上 SOTA。 | **ONLINE_KEEP** | SOTA CORE |
| **Adaptive Uncertainty / SPS** | fresh head 用 50 Train canonical 训练、16 Dev calibration；Dev SPS `42.124892 → 45.070082`，线上 SPS `29.519724 → 30.319572`。 | **ONLINE_KEEP** | SOTA CORE |
| **Local / Point / Residual Corrector** | Rel/MVPE 有信号，但 trajectory-level TKE 保护不稳；未进入 SOTA-V2。 | **PARKED** | PARKED |
| **Multi-scale / coarse+fine A2** | matched screen 三项无正收益。 | **NO_GO** | STOP |
| **Structured Temporal Dynamics** | ΔUV trade-off、Temporal Mixer/Attention 无稳定收益；显式 future temporal architecture 未发现大台阶。 | **CLOSED / WEAK_SIGNAL** | PARKED |
| **Backbone Family** | FNO / Transolver 尚未形成足以替换当前 SOTA-V2 的同协议证据。若再做，只做高信息增益 bounded screen。 | **OPEN** | P1 |
| **POD / Modal Dynamics** | 尚未形成可并入当前 SOTA 的强证据。 | **OPEN** | P1 |
| **Sim2Real / CFD** | raw transfer / frozen representation 均未形成强增益，CFD 主要保留 OOD/coverage 研究价值。 | **WEAK_SIGNAL / PARKED** | PARKED |
| **Data Split / Distribution Audit** | 50/16/16 `SPLIT_OK`；已完成 duplicate / OOD-like 审计。 | **CLOSED** | Support |

---

## 4. SOTA-V2 已验证事实

### 4.1 50/16 predictive result

固定 50 Train / 16 Dev 上，SOTA-V2 @32500：

- Rel-L2 `0.0999346152`
- TKE `0.4692927301`
- MVPE `0.0757779852`

相对 historical balanced anchor `0.112925 / 0.494840 / 0.084671`，raw error 分别下降约：

- Rel-L2 `11.50%`
- TKE `5.16%`
- MVPE `10.50%`

### 4.2 Full-data refit

- released PIV trajectories：`82`
- Dense-All windows：`66755`
- Stage A end：`49461`
- Final update：`53582`
- Full checkpoint SHA256：`f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce`

### 4.3 Adaptive uncertainty

- Head training：50 Train canonical `2052` windows，`1400` updates
- Dev calibration：16 Dev `659` windows
- Best bounds：`half_width_uv = 0.0025 + 1.0 * sigma`
- pressure half-width：`0`
- Dev adaptive SPS：`45.0700816004`
- same-backbone static SPS：`42.1248919471`

### 4.4 Online result

Final clean package SHA256：`9cfc055c6232d2b0aef9f88f1cb3de2aae883b7cff7f02659ed8b9cf85b3ed55`。

Codabench：

| Metric | Previous SOTA | SOTA-V2 | Delta |
|---|---:|---:|---:|
| Final | `76.694784` | **`77.314446`** | **`+0.619662`** |
| Rel-L2 | `93.434384` | **`93.816645`** | `+0.382261` |
| TKE | `77.588799` | **`79.164203`** | `+1.575404` |
| MVPE | `92.519563` | **`93.411176`** | `+0.891613` |
| Time | `87.066646` | `86.898836` | `-0.167810` |
| SPS | `29.519724` | **`30.319572`** | `+0.799848` |

结论：**SOTA-V2 = ONLINE_KEEP / NEW_ONLINE_SOTA**。这次提升不是单纯 SPS 校准收益，三个 physical prediction subscore 同时上涨。

---

## 5. 下一轮主要问题

> 本节中的研究优先级必须以 `REALPDE_CLEAN_BASELINE_V1` 为默认 control。旧协议的结果只能决定“值不值得重新验证”，不能直接证明在新 baseline 上有效。

下一轮先复盘而不是立即 full train：

1. **TKE 仍是相对最低的主要物理子分。** 已有诊断显示趋势/方向相关性高，但能量幅值存在系统性偏差；值得研究 amplitude calibration / energy-aware correction，但不能破坏 Rel/MVPE。
2. **SPS 仍有线上 calibration generalization gap。** Dev `45.07` 对线上 `30.32`，说明下一轮收益重点可能来自更稳健的 uncertainty calibration 泛化，而不是盲目加复杂 head。
3. **Late horizon 仍是结构性误差来源。** h19/h20 在 integrated Dev 中仍占较高 squared-error fraction。
4. **任何新 merge 都以 `77.314446` 为唯一线上 anchor。** 若预期只是小幅单指标改善，默认不值得再烧完整 merge 周期。

---

## 6. 当前明确不做

除非出现新的独立机制证据，否则不继续：

- Temporal Transformer / SSM / autoregressive rollout 大规模展开；
- Vorticity / ΔUV 权重扫描；
- MF 单独机制精修；
- Local / Point / coarse+fine residual 变体扫描；
- raw CFD transfer / 长 CFD pretraining / 复杂 Sim2Real campaign；
- Feature 21/22/23 式继续堆手工特征；
- 同 LR 的无目的超长训练；
- 直接在 Codabench 上高频搜索 SPS 参数。

---

## 7. 关键文档

- 当前线上：`docs/sota迭代/README.md`
- 最新线上 review：`docs/sota迭代/reviews/sota_v2_adaptive_20260916/README.md`
- 最新 handoff：`docs/coordination/CHATGPT_HANDOFF_SOTA_V2_ADAPTIVE_20260916.md`
- Submission：`docs/submission_log.md`
- Inference：`docs/inference/inference概要.md`
- Modeling：`docs/modeling/modeling概要.md`
- Feature Engineering：`docs/feature_engineering/feature_engineering概要.md`
- Sim2Real：`docs/sim2real/sim2real概要.md`
- Dataset：`docs/data/DATASET_PROFILE.md`
- Experiment registry：`docs/track1_experiment_registry.md`

后续战略决策优先以本文件和最新 SOTA review 为入口。
