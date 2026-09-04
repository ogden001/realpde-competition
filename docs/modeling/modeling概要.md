# Modeling 优化概要

## 1. 整体思路

以官方三通道 CNO 为主性能锚点；任何新范式都要在冻结 ID protocol 下，与其匹配的对照同时比较 Rel-L2、TKE、MVPE、runtime 与稳定性。

## 2. 当前结论

| 技术方向 | 内容概要 | 关键实验结果 | 状态 | 详细文档 |
|---|---|---|---|---|
| CNO 主线 | CNO 当前拥有最好已知 Codabench 结果，优于曾上线的 UNet 后处理候选的隐藏物理子分。 | CNO `75.58455`；UNet `74.48384`。 | KEEP | [Submission log](../submission_log.md) |
| 纯 Point MLP | 无空间上下文的 Point 模型未通过既定 dev gate。 | Point residual 对 PERSIST 的 Rel-L2/MVPE 未达门槛。 | STOP | [Point-V0](../coordination/CHATGPT_HANDOFF_POINT_V0.md) |
| LOCAL3 Point | LOCAL3 与降 TKE 权重的 bounded 变体均未在 1500-step screen 通过。 | LOCAL3 Rel-L2/MVPE 均退化；`λ_TKE=0.001` 仍未改善 Rel-L2。 | STOP | [LOCAL3](../coordination/CHATGPT_HANDOFF_POINT_V1_LOCAL3.md), [balanced loss](../coordination/CHATGPT_HANDOFF_POINT_LOCAL3_BALANCED_L001.md) |
| CNO + Point H1 | 原始 H1 的 TKE 代价触发早停；train-selected `alpha=0.5` 缩放通过 aggregate gate，但 trajectory-level TKE 保护不稳。 | Rel/MVPE 16/16 trajectory 改善；满足 TKE 保护仅 3/16。 | REVIEW | [scale](../coordination/CHATGPT_HANDOFF_HYBRID_CNO_POINT_H1_SCALE.md), [stability](../coordination/CHATGPT_HANDOFF_HYBRID_CNO_POINT_H1_SCALE_STABILITY.md) |
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

## 3. TODO

| 技术方向 | 内容概要 | 优先级 |
|---|---|---|
| H1 | 等待 ChatGPT/Sol 对 scale stability 的单一、受限 NEXT_ACTION；不自动启动 H2、joint training 或 LOCAL5。 | P0 |

## 4. 相关文档

- [协调状态](../coordination/STATUS.md)
- [Track 1 实验注册表](../track1_experiment_registry.md)
