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
| MF-01 Mean/Fluctuation | 输出层 temporal mean + zero-mean fluctuation factorization。 | 1500 updates: Rel-L2 `-2.72%`、MVPE `-2.80%`，但 TKE `+1.79%`；trajectory wins Rel/TKE/MVPE `13/16, 7/16, 10/16`。 Deep replay: MF fluctuation RMS lower in all 16 cases; ΔTKE is descriptor-tail correlated but not explained by OOD or horizon alone. | NO-GO / REVIEW | [MF-01 handoff](../coordination/CHATGPT_HANDOFF_MF01.md), [deep analysis](../coordination/CHATGPT_HANDOFF_MF01_DEEP_ANALYSIS.md) |
| MF Energy Campaign 02 | Scorer-aligned TKE, RMS, high-energy weighting, frozen conditional/spatial gain. | E5 has 15/16 TKE wins vs C0 but costs MVPE; E4 weak; E6 harms MVPE and high-energy cases; E7/E8 gains remain identity. | NO-GO / REVIEW | [Campaign02 handoff](../coordination/CHATGPT_HANDOFF_MF_ENERGY_CAMPAIGN02.md) |
| MF Campaign02 Full Evidence Review | Cross-experiment replay of MF@1500→C0@3000 and E4–E8. | C0 gains are broad and led by Mean convergence; E5 TKE wins are substantive but not MVPE-safe; E6 high-energy weighting fails; gains remain identity. | REVIEW | [Full evidence review](../coordination/CHATGPT_HANDOFF_MF_CAMPAIGN02_FULL_EVIDENCE_REVIEW.md) |
| MF Direction Closeout | Direct@1500→Direct@3000 matched continuation versus MF@1500/MF@3000. | MF@3000 improves Rel-L2 `6.541%`, TKE `1.971%`, MVPE `14.019%` versus Direct@3000; trajectory wins `16/16`, `8/16`, `15/16`. | PROMISING_PARKED | [Closeout](../coordination/CHATGPT_HANDOFF_MF_DIRECTION_CLOSEOUT.md) |
| MF Energy Campaign 01 | MF-01 fixed-factor probes: TKE weight 2x, scalar conditional gain, spatial gain. | E2/E3 Rel-L2 `-2.624%/-2.644%` and MVPE `-3.314%/-3.488%` vs MF-01, but TKE wins only `4/16` and `3/16`; E1 TKE `-0.614%` but Rel-L2 worsens `+1.161%`. | NO-GO / REVIEW | [Campaign handoff](../coordination/CHATGPT_HANDOFF_MF_ENERGY_CAMPAIGN01.md) |

## 3. TODO

| 技术方向 | 内容概要 | 优先级 |
|---|---|---|
| H1 | 等待 ChatGPT/Sol 对 scale stability 的单一、受限 NEXT_ACTION；不自动启动 H2、joint training 或 LOCAL5。 | P0 |

## 4. 相关文档

- [协调状态](../coordination/STATUS.md)
- [Track 1 实验注册表](../track1_experiment_registry.md)
