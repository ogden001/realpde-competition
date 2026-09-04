# Modeling 优化概要

## 1. 整体思路

以当前 CNO 主线为性能锚点；任何新范式都要在冻结 ID protocol 下，与其匹配的对照同时比较 Rel-L2、TKE、MVPE、runtime 与稳定性。

表示学习要区分“某个具体 representation 实现失败”和“representation 方向整体关闭”。Sim2Real Round 2 只否定了 official CFD `sim_pretrain` frozen representation 作为 PIV forecasting 主增量来源，不等价于否定所有频谱、模态、PIV self-supervised 或新 backbone representation。

## 2. 当前结论

| 技术方向 | 内容概要 | 关键实验结果 | 状态 | 详细文档 |
|---|---|---|---|---|
| CNO 主线 | CNO 当前拥有最好已知 Codabench 结果。 | 当前 SOTA：`P0-A + N2 + CNO + full@43260 + explicit SPS bounds`，Final `76.149726`。 | KEEP | [SOTA](../sota迭代/README.md) / [Submission log](../submission_log.md) |
| 纯 Point MLP | 无空间上下文的 Point 模型未通过既定 dev gate。 | Point residual 对 PERSIST 的 Rel-L2/MVPE 未达门槛。 | STOP | [Point-V0](../coordination/CHATGPT_HANDOFF_POINT_V0.md) |
| LOCAL3 Point | LOCAL3 与降 TKE 权重的 bounded 变体均未在 1500-step screen 通过。 | LOCAL3 Rel-L2/MVPE 均退化；`λ_TKE=0.001` 仍未改善 Rel-L2。 | STOP | [LOCAL3](../coordination/CHATGPT_HANDOFF_POINT_V1_LOCAL3.md), [balanced loss](../coordination/CHATGPT_HANDOFF_POINT_LOCAL3_BALANCED_L001.md) |
| CNO + Point H1 | 原始 H1 的 TKE 代价触发早停；train-selected `alpha=0.5` 缩放通过 aggregate gate，但 trajectory-level TKE 保护不稳。 | Rel/MVPE 16/16 trajectory 改善；满足 TKE 保护仅 3/16。 | REVIEW | [scale](../coordination/CHATGPT_HANDOFF_HYBRID_CNO_POINT_H1_SCALE.md), [stability](../coordination/CHATGPT_HANDOFF_HYBRID_CNO_POINT_H1_SCALE_STABILITY.md) |
| Hybrid CNO + Local A1 | P0-A CNO global + raw Past20 u/v lightweight Conv3D local residual；zero-init 后 joint training。 | 有效 rerun 中 TKE 在 matched@2000/2500/3000 均略优；A1@2500 三指标仅约 `+0.68%/+0.20%/+0.67%`；A1@3000 为 Rel `-0.286%`、TKE `+0.498%`、MVPE `-2.277%`，wins `2/16,10/16,1/16`。 | **WEAK_SIGNAL_PARKED** | [Sol review](reviews/hybrid_cno_local_a1_rerun_20260904/SOL_REVIEW.md) |
| Official CFD frozen representation | official `sim_pretrain` CNO 主干冻结，仅训练同预算 tiny probe；与同架构 random frozen CNO 对照。 | CFD rep 的 Rel-L2/TKE/MVPE 为 `0.9026/32.0180/1.1509`，random control 为 `0.7680/13.0809/0.7800`，三项均明显更差。 | **STOP** | [REP-01](../coordination/CHATGPT_HANDOFF_SIM2REAL_REP01.md) |
| MF-01 Mean/Fluctuation | 输出层 temporal mean + zero-mean fluctuation factorization。 | 1500 updates: Rel-L2 `-2.72%`、MVPE `-2.80%`，但 TKE `+1.79%`；该 checkpoint 后续被证明明显 under-converged。 | HISTORICAL_SCREEN | [MF-01 handoff](../coordination/CHATGPT_HANDOFF_MF01.md), [deep analysis](../coordination/CHATGPT_HANDOFF_MF01_DEEP_ANALYSIS.md) |
| MF Energy Campaign 01 | MF-01 fixed-factor probes: TKE weight 2x, scalar conditional gain, spatial gain。 | E2/E3 Rel-L2 `-2.624%/-2.644%` 和 MVPE `-3.314%/-3.488%` vs MF-01，但 TKE wins 仅 `4/16` 和 `3/16`；gain 基本未学出有效校准。 | NO-GO / REVIEW | [Campaign handoff](../coordination/CHATGPT_HANDOFF_MF_ENERGY_CAMPAIGN01.md) |
| MF Energy Campaign 02 | Scorer-aligned TKE、RMS、high-energy weighting、frozen conditional/spatial gain。 | E5 有 `15/16` TKE wins 但损伤 MVPE；E4 弱；E6 损伤 MVPE 与 high-energy cases；E7/E8 gain 近 identity。 | NO-GO / REVIEW | [Campaign02 handoff](../coordination/CHATGPT_HANDOFF_MF_ENERGY_CAMPAIGN02.md) |
| MF Campaign02 Full Evidence Review | Cross-experiment replay of MF@1500→C0@3000 and E4–E8。 | C0 长训收益主要来自 Mean convergence；E5 的 TKE 信号实质存在但不 MVPE-safe；E6 high-energy weighting 失败。 | REVIEW | [Full evidence review](../coordination/CHATGPT_HANDOFF_MF_CAMPAIGN02_FULL_EVIDENCE_REVIEW.md) |
| **MF Direction Closeout** | Direct@1500→Direct@3000 matched continuation versus MF@1500/MF@3000。 | **MF@3000 vs Direct@3000：Rel-L2 `-6.541%`、TKE `-1.971%`、MVPE `-14.019%`；trajectory wins `16/16`、`8/16`、`15/16`。** | **PROMISING_PARKED** | [Closeout](../coordination/CHATGPT_HANDOFF_MF_DIRECTION_CLOSEOUT.md) |

### 2.1 Mean / Fluctuation 第一阶段收口结论

Mean / Fluctuation（MF）已完成 breadth-first 第一阶段验证，当前结论为 **`PROMISING_PARKED`**。

公平 matched comparison：

| Model | Rel-L2 | TKE | MVPE |
|---|---:|---:|---:|
| Direct@1500 | 0.193675 | 0.633786 | 0.165178 |
| MF@1500 | 0.188409 | 0.645156 | 0.160552 |
| Direct@3000 | 0.175829 | 0.594649 | 0.151631 |
| MF@3000 | **0.164327** | **0.582928** | **0.130374** |

当前可保留的稳定判断：

1. **MF 方向本身有明确价值。** 在相同 3000-update continuation 语义下，MF 三项官方 raw error 均优于 Direct；Rel-L2 与 MVPE 的收益尤其明显，且分别为 `16/16`、`15/16` trajectory wins。
2. **MF@1500 不能代表该方向最终能力。** Full Evidence Review 已确认 MF@1500 明显 under-converged；继续训练后的主要收益来自 Mean reconstruction，Fluctuation 也有改善但更不稳定。
3. **TKE 仍是该方向未来最值得处理的薄弱项。** RMS/amplitude objective 曾产生 `15/16` trajectory TKE wins，但会损伤 Mean/MVPE；简单 scorer-aligned TKE、high-energy weighting 和 conditional/spatial gain 均未形成可保留方案。
4. **按“60 分原则”当前停止继续精修。** 不再自动推进 RMS decoupling、spectrum、MF-02、temporal head 或其它机制级实验。待其它一级方向完成粗筛后，再决定是否将 MF 作为第二阶段重点方向回收。

因此，未来若重新打开 MF，首要问题不是重新证明“有没有价值”，而是：**如何在保持 Mean/MVPE 优势的同时继续提高 Fluctuation/TKE。**

### 2.2 CFD Representation 收口结论

Sim2Real Round 2 的 REP-01 结论是 **`CFD_REPRESENTATION_NOT_SUPPORTED`**。

需要保留两个边界：

1. 这只否定 official CFD `sim_pretrain` frozen representation + tiny probe 这条具体低成本路线；
2. latent pooled mean 的 CFD↔PIV gap 虽然比 raw field 更小，但没有转化为 downstream PIV forecasting 增量，因此“latent 更接近”不能单独作为 representation 成功标准。

按照 60 分原则，不继续设计更复杂的 CFD teacher/student、adversarial alignment 或专门 CFD self-supervised Campaign。其它与 CFD 无关的 Representation 方向仍保持开放。

### 2.3 Local + Global A1 第一阶段结论

A1 的第一次执行因 local branch 未进入 optimizer 而判定为 `INVALID_IMPLEMENTATION`；修复后 rerun 已通过 optimizer membership、parameter delta、branch output 与 checkpoint round-trip 等 preflight，最终证据有效。

Sol 复核 matched checkpoint 后得到：

- TKE 在 Direct/A1 matched@2000、2500、3000 三个点均小幅改善，说明 local correction 存在可重复的 energy/TKE 信号；
- 但 Rel-L2 与 MVPE 不稳定，仅 A1@2500 出现三指标同时改善，幅度均小于 1%；
- A1@3000 trajectory wins 为 Rel `2/16`、TKE `10/16`、MVPE `1/16`，说明 TKE 信号较普遍，但 reconstruction 增量不稳；
- by-horizon 显示约 `t+3 ... t+17` 的 Rel/MVPE 主要退化，提示简单 output-level local residual 的时间/相位校准不足；
- final local residual 真实非零，量级约为 global output 的 1% 左右；runtime 增加约 2.65%。

因此当前具体实现定为 **`WEAK_SIGNAL_PARKED`**。不继续扫 local width、kernel、gate 或更多 Local residual 变体；`Local + Global` 大方向并未被整体否定，但当前优先级下降。Architecture breadth-first 下一步应测试一个机制明显不同的方向，优先 **Multi-scale / coarse+fine modeling**。

## 3. TODO

| 技术方向 | 内容概要 | 优先级 |
|---|---|---|
| Architecture A2 | 测试与 Local residual 明显不同的 Multi-scale / coarse+fine 机制；保持 bounded breadth-first，不做大规模 backbone sweep。 | P0 |
| H1 | 等待 ChatGPT/Sol 对 scale stability 的单一、受限 NEXT_ACTION；不自动启动 H2、joint training 或 LOCAL5。 | P1 |
| CFD representation | 当前 `STOP / PARKED`；仅在出现可靠 calibration、新 OOD failure linkage 或新的明确机制时重新打开。 | PARKED |

## 4. 相关文档

- [A1 Sol Review](reviews/hybrid_cno_local_a1_rerun_20260904/SOL_REVIEW.md)
- [Sim2Real / CFD 利用概要](../sim2real/sim2real概要.md)
- [REP-01](../coordination/CHATGPT_HANDOFF_SIM2REAL_REP01.md)
- [Round 2 Decision](../coordination/CHATGPT_HANDOFF_SIM2REAL_ROUND2_DECISION.md)
- [协调状态](../coordination/STATUS.md)
- [Track 1 实验注册表](../track1_experiment_registry.md)
