# RealPDE Track 1 整体优化概要

> 本文只维护 **Track 1 当前战略地图、方向状态和下一步优先级**。具体实验过程、配置和指标不在这里展开，统一链接到各方向概要、`track1_experiment_registry.md` 和 coordination handoff。

- Deadline：**2026-09-28**
- 当前阶段：**探索收口期**；优先完成仍未覆盖的大方向，避免继续深挖已关闭弱信号。
- 当前线上 SOTA：**P0-A + N2 + CNO + full-data late continuation + learned adaptive uncertainty bounds**，Codabench Final **76.694784**。

---

## 1. 默认阅读顺序

新开一级研究会话时先读：

1. `docs/realpde整体优化概要.md`
2. `docs/sota迭代/README.md`
3. `docs/submission_log.md`
4. 对应方向 `*概要.md` / closeout
5. `docs/track1_experiment_registry.md` 与相关 coordination handoff

原则：不能因为当前 SOTA 是 CNO 就只围绕 CNO 微调；也不能因为做战略探索就忽略已获得的线上/离线证据。

---

## 2. 研发与资源原则

### 2.1 60 分原则

一级方向优先回答：

> **有没有大台阶？值不值得继续投资？**

默认先做 1～2 小时级粗筛或少量代表性方案，不在弱信号上连续做超参雕琢。

方向状态统一使用：

- `PROMISING`：有明显工程收益，记录后 PARK；
- `WEAK_SIGNAL / PARKED`：有小幅或冲突信号，暂不继续；
- `NO_GO / STOP`：当前实现路线停止；
- `CLOSED`：该一级方向已完成本阶段探索，不再自动追加实验。

### 2.2 SOTA Merge 门槛

一次完整 merge 通常消耗约 **4～6 轮 ChatGPT/Codex + 4～6 GPU 小时**，因此任何 merge 前必须先做 `Merge Worthiness Review`：

- `MERGE_WORTHY`：存在大台阶变量，或多个证据强且兼容的增量；
- `SKIP_MERGE`：只有孤立小收益、weak signal、线上迁移不确定或继续 exploration 的信息收益更高。

`+1%` 左右的小离线收益默认不足以单独触发完整 merge。

---

## 3. 当前战略地图

| 一级方向 | 当前结论 | 状态 | 优先级 |
|---|---|---|---|
| **Data Regime / Random Window** | 完整 PIV trajectory 仍可能存在大量未利用的 temporal phase / window coverage；随机窗口实验正在验证。 | **RUNNING** | **P0** |
| **Backbone Family** | 项目长期主要围绕 CNO 展开，FNO / Transolver 尚未完成同协议 family screen。 | **OPEN** | **P0** |
| **POD / Modal Dynamics** | 尚未回答“高维流场是否主要由低维 coherent modes 驱动”；信息增益高、实验成本低。 | **OPEN** | **P0** |
| **Structured Temporal Dynamics** | ΔUV 有 trade-off；latent Temporal Conv 仅弱信号；Temporal Attention 无收益；显式 Future20 temporal architecture 未发现大台阶。 | **CLOSED / WEAK_SIGNAL** | PARKED |
| **Vorticity Supervision** | long paired final：late median Rel `+3.243%`、TKE `-0.564%`、MVPE `+2.059%`；MVPE gate 与 2-of-3 rule 失败。 | **PARK** | PARKED |
| **Mean / Fluctuation** | @3000 曾有强收益，但 long-convergence 后 wash out / mixed；@15000 未保持强 gate。 | **WEAK_SIGNAL_PARKED** | PARKED |
| **Local / Point / Hybrid residual** | Point/LOCAL3 STOP；H1 Rel/MVPE 强但 trajectory-level TKE 保护不稳；Local+Global A1 只有弱 TKE 信号。 | **PARKED** | P1/PARKED |
| **Multi-scale / coarse+fine A2** | matched@3000 三项无正收益。 | **NO_GO** | STOP |
| **Feature Engineering** | runtime-safe temporal/spatial features在简单 probe 有预测价值，但现有 CNO fusion 路线未形成稳定三指标收益；Feature Discovery 已关闭。 | **PARKED** | P1/PARKED |
| **Sim2Real / CFD** | raw temporal transfer 不支持；official frozen CFD representation 不支持；CFD-only conditions 仅保留 OOD/coverage 价值。 | **WEAK_SIGNAL / PARKED** | PARKED |
| **Loss / Objective** | N2 已是当前有效基础；继续做简单 scalar weight scan 信息价值低。 | **KEEP BASE / PARK SCAN** | P1 |
| **SPS / Uncertainty** | learned adaptive uncertainty 已在线上验证有效并提升 Final；当前作为 submission-layer 默认组件。 | **KEEP** | Exploit |
| **Data Split / Distribution Audit** | 50/16/16 `SPLIT_OK`；无明显 Final-only coverage gap。Train `6300_0.h5` 与 Final `7575_0.h5` Past20 exact duplicate，locked-final 审计需同时报告 all16 / unique15。 | **CLOSED** | Support |

---

## 4. Structured Temporal Dynamics 最新收口

该方向最初验证两个假设：

1. `Past20 → Future20` 一次性预测是否缺少未来帧内部时序关系；
2. 除直接 u/v 外，速度增量、涡量等中间变量能否提供更稳定监督。

最终结果：

- ΔUV Temporal Supervision：Rel 有明显信号，但持续牺牲 TKE；
- output Temporal Mixer：无稳定收益；
- Latent Temporal Conv：约 1%～3% 弱信号，far horizon 略强；
- Latent Temporal Attention：基本无收益；
- Vorticity：短训曾明显改善 Rel/MVPE，但 long paired final 未达到稳定 merge gate。

因此：

> **显式 Future20 temporal structure 不是当前 CNO 的主要性能瓶颈；物理辅助监督有一定价值，但目前也没有形成足够稳定、足够大的 SOTA 增量。**

结论：

- `Vorticity Supervision = PARK`
- `Structured Temporal Dynamics exploration = CLOSED`
- 不继续 Temporal Transformer / SSM / autoregressive / Block rollout / ΔUV / Vorticity λ 扫描。

完整过程：`docs/modeling/structured_temporal_dynamics_closeout.md`

---

## 5. 当前 P0 队列

### P0-1 Random Window / Phase Augmentation

核心问题：现有 stride20 是否严重浪费完整 PIV trajectory 的 temporal phase coverage。当前实验正在执行，等待结果后再决定是否进入 merge pool。

### P0-2 Backbone Family Screen

使用统一 PIV protocol 对 CNO / FNO / Transolver 做 bounded 粗筛。目标不是调到各自最优，而是判断是否存在 5% 级 family difference 或互补 metric profile。

### P0-3 POD / Modal Dynamics

先做 POD/PCA reconstruction ceiling，再用轻量 temporal predictor 预测 modal coefficients。目标是判断：

> 当前任务真正难的是 spatial field reconstruction，还是低维 temporal dynamics？

若低维 modal ceiling 很高，再升级为正式建模方向。

---

## 6. 当前明确不做

除非出现新的独立机制证据，否则不继续：

- Temporal Transformer / SSM / autoregressive rollout；
- Vorticity / ΔUV 权重扫描；
- MF 机制精修；
- Local / Point / coarse+fine residual 变体扫描；
- raw CFD transfer / 长 CFD pretraining / 复杂 Sim2Real campaign；
- Feature 21/22/23 式继续堆手工特征；
- 同 LR 的无目的超长训练；
- SPS 固定 bounds 微调。

---

## 7. 关键文档

- 当前线上：`docs/sota迭代/README.md`
- Submission：`docs/submission_log.md`
- Modeling：`docs/modeling/modeling概要.md`
- Structured Temporal closeout：`docs/modeling/structured_temporal_dynamics_closeout.md`
- Feature Engineering：`docs/feature_engineering/feature_engineering概要.md`
- Sim2Real：`docs/sim2real/sim2real概要.md`
- Dataset：`docs/data/DATASET_PROFILE.md`
- Duplicate audit：`docs/data/DUPLICATE_AUDIT.md`
- Experiment registry：`docs/track1_experiment_registry.md`
- Vorticity final：`docs/coordination/CHATGPT_HANDOFF_VORTICITY_LOWMEM_FINAL.md`

后续战略决策优先以本文件的当前状态为入口，具体数值回溯到对应方向文档和 registry。