# Modeling 优化概要

## 1. 整体思路

以当前 CNO 主线为性能锚点；任何新范式都要在冻结 ID protocol 下，与其匹配的对照同时比较 Rel-L2、TKE、MVPE、runtime 与稳定性。

表示学习要区分“某个具体 representation 实现失败”和“representation 方向整体关闭”。Sim2Real Round 2 只否定了 official CFD `sim_pretrain` frozen representation 作为 PIV forecasting 主增量来源，不等价于否定所有频谱、模态、PIV self-supervised 或新 backbone representation。

## 2. 当前结论

| 技术方向 | 内容概要 | 关键实验结果 | 状态 | 详细文档 |
|---|---|---|---|---|
| CNO 主线 | CNO 当前拥有最好已知 Codabench 结果。 | 当前 SOTA：`P0-A + N2 + CNO + full@43260 + learned adaptive uncertainty bounds`，Final `76.694784`。 | KEEP | [SOTA](../sota迭代/README.md) / [Submission log](../submission_log.md) |
| 纯 Point MLP | 无空间上下文的 Point 模型未通过既定 dev gate。 | Point residual 对 PERSIST 的 Rel-L2/MVPE 未达门槛。 | STOP | [Point-V0](../coordination/CHATGPT_HANDOFF_POINT_V0.md) |
| LOCAL3 Point | LOCAL3 与降 TKE 权重的 bounded 变体均未在 1500-step screen 通过。 | LOCAL3 Rel-L2/MVPE 均退化；`λ_TKE=0.001` 仍未改善 Rel-L2。 | STOP | [LOCAL3](../coordination/CHATGPT_HANDOFF_POINT_V1_LOCAL3.md), [balanced loss](../coordination/CHATGPT_HANDOFF_POINT_LOCAL3_BALANCED_L001.md) |
| CNO + Point H1 | 原始 H1 的 TKE 代价触发早停；train-selected `alpha=0.5` 缩放通过 aggregate gate，但 trajectory-level TKE 保护不稳。 | Rel/MVPE 16/16 trajectory 改善；满足 TKE 保护仅 3/16。 | REVIEW | [scale](../coordination/CHATGPT_HANDOFF_HYBRID_CNO_POINT_H1_SCALE.md), [stability](../coordination/CHATGPT_HANDOFF_HYBRID_CNO_POINT_H1_SCALE_STABILITY.md) |
| Hybrid CNO + Local A1 | P0-A CNO global + raw Past20 u/v lightweight Conv3D local residual；zero-init 后 joint training。 | 有效 rerun 中 TKE 在 matched@2000/2500/3000 均略优；A1@2500 三指标仅约 `+0.68%/+0.20%/+0.67%`；A1@3000 为 Rel `-0.286%`、TKE `+0.498%`、MVPE `-2.277%`。 | WEAK_SIGNAL_PARKED | [Sol review](reviews/hybrid_cno_local_a1_rerun_20260904/SOL_REVIEW.md) |
| Multi-scale / coarse+fine A2 | Lightweight coarse+fine residual branch。 | matched@3000 相对 Direct：Rel `-0.690%`、TKE `-0.005%`、MVPE `-2.010%`，没有形成正收益。 | **NO_GO** | [overnight evidence](../sota迭代/reviews/overnight_integrated_20260905/README.md) |
| Official CFD frozen representation | official `sim_pretrain` CNO 主干冻结，仅训练同预算 tiny probe；与同架构 random frozen CNO 对照。 | CFD rep 的 Rel-L2/TKE/MVPE 为 `0.9026/32.0180/1.1509`，random control 为 `0.7680/13.0809/0.7800`，三项均明显更差。 | STOP | [REP-01](../coordination/CHATGPT_HANDOFF_SIM2REAL_REP01.md) |
| Mean / Fluctuation | 输出 temporal mean + zero-mean fluctuation factorization。 | @3000 曾出现 Rel/MVPE 明显收益，但长收敛验证到 15000 后收益 wash out / mixed；最终 @15000 仅 Rel `+1.157%`、TKE `+0.207%`、MVPE `-1.969%`，未通过强 gate。 | **WEAK_SIGNAL_PARKED** | [Closeout](../coordination/CHATGPT_HANDOFF_MF_DIRECTION_CLOSEOUT.md) / long-convergence evidence |
| ΔUV Temporal Supervision | 对 Future20 帧间速度增量增加辅助监督。 | R1 中 Rel-L2 可改善，但持续牺牲 TKE，属于明确 trade-off。 | PARKED | [R1](../coordination/CHATGPT_HANDOFF_STRUCTURED_TEMPORAL_DYNAMICS_R1.md) |
| Latent Temporal Conv | 在 CNO project 前 latent 上加入时间卷积。 | R2@3000 相对 C0：Rel-L2约 `+1.08%`、TKE `+0.21%`、MVPE `+2.70%`；Far horizon 信号略强，但幅度不足。 | WEAK_SIGNAL_PARKED | [R2](../coordination/CHATGPT_HANDOFF_STRUCTURED_TEMPORAL_DYNAMICS_R2.md) |
| Latent Temporal Attention | 在 CNO project 前对 Future20 latent 做 temporal-only self-attention。 | R2@3000 基本无增益：Rel/TKE/MVPE 约 `-0.04%/-0.22%/-0.06%`。 | **NO_GO** | [R2](../coordination/CHATGPT_HANDOFF_STRUCTURED_TEMPORAL_DYNAMICS_R2.md) |
| Vorticity Supervision | N2 + fixed vorticity MSE；不改变推理结构。 | low-memory matched final 的 late `6000/9000/12000` median：Rel-L2 `+3.243%`、TKE `-0.564%`、MVPE `+2.059%`；MVPE 4% gate 与 2-of-3 rule 均失败。 | **PARK** | [Low-memory final](../coordination/CHATGPT_HANDOFF_VORTICITY_LOWMEM_FINAL.md) |

### 2.1 Mean / Fluctuation 最终结论

Mean / Fluctuation 在 matched@3000 曾出现很强的早期信号，但后续 long-convergence 证明该优势并不稳定：继续训练到 6000/9000/12000/15000 后，三项指标反复交叉；@15000 相对 Direct 仅 Rel-L2 `+1.157%`、TKE `+0.207%`，MVPE 反而 `-1.969%`。固定强 gate 从未被满足。

因此旧的 `PROMISING_PARKED` 只代表早期粗筛，不再作为当前战略结论。当前状态统一为 **`WEAK_SIGNAL_PARKED`**。不继续做 RMS decoupling、MF-02、spectrum 或其它 MF 精修，除非未来出现新的独立机制证据。

### 2.2 CFD Representation 收口结论

Sim2Real Round 2 的 REP-01 结论是 **`CFD_REPRESENTATION_NOT_SUPPORTED`**。

需要保留两个边界：

1. 这只否定 official CFD `sim_pretrain` frozen representation + tiny probe 这条具体低成本路线；
2. latent pooled mean 的 CFD↔PIV gap 虽然比 raw field 更小，但没有转化为 downstream PIV forecasting 增量，因此“latent 更接近”不能单独作为 representation 成功标准。

按照 60 分原则，不继续设计更复杂的 CFD teacher/student、adversarial alignment 或专门 CFD self-supervised Campaign。其它与 CFD 无关的 Representation 方向仍保持开放。

### 2.3 Local + Global / Multi-scale 收口

Local+Global A1 只有弱 TKE 信号且 Rel/MVPE 不稳，已 `WEAK_SIGNAL_PARKED`。后续 coarse+fine A2 作为机制不同的 multi-scale probe 也没有产生正收益，matched@3000 三项均未改善，因此 **Multi-scale/coarse+fine A2 = NO_GO**。

不再把 Multi-scale / coarse+fine 作为当前 P0 建模方向，也不继续扫 local width、kernel、gate 或更多轻量 residual branch。

### 2.4 Structured Temporal Dynamics / Vorticity Supervision 终局收口

这个方向最初要验证的是：当前 `Past20 → Future20` 是否因为缺乏 Future20 内部显式时序关系而存在明显结构性瓶颈，以及中间物理变量是否能帮助 u/v 预测。

R1/R2/终局验证得到：

1. **ΔUV Temporal Supervision**：能明显改变 Rel-L2，但伴随持续 TKE trade-off，`PARKED`。
2. **Output Temporal Mixer**：没有形成可靠收益，且 R1 optimizer 口径存在混杂，仅作为历史弱证据。
3. **Latent Temporal Conv**：有约 1%～3% 的小幅信号，far horizon 略强，但不足以成为比赛级增量，`WEAK_SIGNAL_PARKED`。
4. **Latent Temporal Attention**：基本无收益，`NO_GO`。
5. **Vorticity Supervision**：R2 短训曾显示 Rel/MVPE 正收益且无推理代价，因此进行了终局长程验证。共享 GPU 低显存 matched final 使用 micro-batch `4`、accumulation `2`、effective batch `8`、12 GiB cap，C0/V1 都从 R2@3000 恢复并训练到 @12000。late `6000/9000/12000` median 为 Rel-L2 `+3.242654%`、TKE `-0.564134%`、MVPE `+2.059027%`；MVPE `>=4%` threshold 失败，2-of-3 checkpoint rule 也失败，仅 @12000 同时满足三项方向条件。因此机械 gate 为 **`FINAL_GATE = PARK`**。独立 official-v9 replay 两臂 raw-error 最大差异均为 `0.0`。

当前直觉结论：**显式 Future20 时序结构不是当前 CNO 的主要性能瓶颈；物理辅助监督有一定价值，但目前也没有形成足够稳定、足够大的 SOTA 增量。**

因此：

- `Vorticity Supervision = PARK`
- `Structured Temporal Dynamics exploration = CLOSED`
- 不启动 Temporal Transformer / SSM / autoregressive / Block rollout / ΔUV / Vorticity λ 扫描等后续 R4。

## 3. TODO

| 技术方向 | 内容概要 | 优先级 |
|---|---|---|
| Random Window / Phase Augmentation | 利用完整 PIV trajectory 的更多合法时间起点，验证 phase/transition coverage 是否是当前数据利用瓶颈。 | **P0 / RUNNING** |
| Backbone Family | CNO / FNO / Transolver 在统一 PIV protocol 下做 bounded family screen。 | **P0** |
| POD / Modal Dynamics | 用 POD/PCA 检查流场是否具有明显低维模态结构，并做轻量 temporal predictor probe。 | **P0** |
| H1 | 保留历史 strong Rel/MVPE signal，但 trajectory-level TKE 风险高；只在全局方向收敛后考虑回收。 | P1 / PARKED |
| CFD representation | 当前 STOP / PARKED；仅在出现可靠 calibration、新 OOD failure linkage 或新的明确机制时重新打开。 | PARKED |

## 4. 相关文档

- [Structured Temporal Dynamics R1](../coordination/CHATGPT_HANDOFF_STRUCTURED_TEMPORAL_DYNAMICS_R1.md)
- [Structured Temporal Dynamics R2](../coordination/CHATGPT_HANDOFF_STRUCTURED_TEMPORAL_DYNAMICS_R2.md)
- [Vorticity Low-memory Final](../coordination/CHATGPT_HANDOFF_VORTICITY_LOWMEM_FINAL.md)
- [A1 Sol Review](reviews/hybrid_cno_local_a1_rerun_20260904/SOL_REVIEW.md)
- [Sim2Real / CFD 利用概要](../sim2real/sim2real概要.md)
- [REP-01](../coordination/CHATGPT_HANDOFF_SIM2REAL_REP01.md)
- [协调状态](../coordination/STATUS.md)
- [Track 1 实验注册表](../track1_experiment_registry.md)
