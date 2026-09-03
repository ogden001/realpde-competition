# Physics 优化概要

## 1. 整体思路

Track 1 的正式预测任务是利用过去 20 帧 PIV 流场预测未来 20 帧 PIV 流场。Sim2Real 主要体现在 CFD 与真实 PIV 数据的联合利用、预训练、迁移学习和分布差异，而不是正式推理阶段直接进行 CFD→PIV 逐帧映射。物理理解在当前阶段的作用，是划清正式推理可用信息与不可用信息，并同时保护平均流和波动结构。

## 2. 当前结论

| 技术方向 | 内容概要 | 关键实验结果 | 状态 | 详细文档 |
|---|---|---|---|---|
| Runtime 物理信息边界 | 正式输入可靠信息是当前窗口 `u/v` 与 tensor shape；`p=0` 是接口兼容占位。 | FF-00 固化了 `[B,20,32,64,3]` 接口与禁止信息清单。 | KEEP | [FF-00 handoff](../coordination/CHATGPT_HANDOFF_FEATURE_FUSION_FF00.md) |
| 波动结构保护 | Rel-L2/MVPE 改善可伴随 TKE 恶化，不能以点误差替代物理判断。 | H1 缩放 Rel-L2/MVPE 分别改善 18.024%/20.137%，TKE error 恶化 3.958%。 | REVIEW | [H1 scale handoff](../coordination/CHATGPT_HANDOFF_HYBRID_CNO_POINT_H1_SCALE.md) |

## 3. TODO

| 技术方向 | 内容概要 | 优先级 |
|---|---|---|
| 物理约束 | 仅在明确授权的候选中检验能同时保护 TKE 的 runtime-safe 物理约束。 | P1 |

## 4. 相关文档

- [特征工程长期总结](../feature_engineering/README.md)
- [Track 1 实验注册表](../track1_experiment_registry.md)
