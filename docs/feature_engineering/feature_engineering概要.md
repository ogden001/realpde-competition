# Feature Engineering 优化概要

## 1. 整体思路

Feature Discovery 回答“哪些 runtime-safe 信息值得保留”，Feature Fusion 回答“模型如何使用这些信息”。当前重点已从继续寻找更多 Feature，转向如何融合已确认有增量信号的 Temporal/Spatial 信息，同时保护 TKE 与波动结构。

## 2. 当前结论

| 技术方向 | 内容概要 | 关键实验结果 | 状态 | 详细文档 |
|---|---|---|---|---|
| Feature Discovery | 当前目录已冻结为 Raw、Temporal、Spatial；不再自动扩展 Feature catalog。 | Discovery 已 `CLOSED`。 | STOP | [长期总结](README.md) |
| Temporal / Spatial 信息价值 | 在 PERSIST residual probe 中，Temporal、Spatial 与联合包都呈现同方向增量；这仅说明信息价值，不保证神经模型收益。 | 联合包在三项 raw error 上优于 Raw-Control。 | KEEP_FOR_MODEL_PROBE | [incremental probe](../coordination/CHATGPT_HANDOFF_FE_INCREMENTAL_PROBE.md) |
| RawSpatial8 直接注入 | 将空间导数直接注入 CNO residual head 未通过受保护的 multi-metric gate。 | Rel-L2/MVPE 未改善，TKE 退化。 | NO-GO | [Registry](../track1_experiment_registry.md) |
| Feature Value 与 Fusion Value | Feature 有预测增量信息，不等于某个 fusion/correction 实现有效。 | 历史 FE-01/FE-02 与 frozen-CNO ridge 均未形成稳定 all-three-metric 收益。 | KEEP | [final review](../coordination/CHATGPT_HANDOFF_FE_FINAL_REVIEW.md) |
| 自动 fusion / correction | FF-00 完成协议与基线冻结；当前不授权自动启动 FF-01/02/03 或 correction 训练。 | 状态为 `REVIEW_REQUIRED`；closure 对自动执行结论为 `STOP`。 | REVIEW / STOP | [FF-00](../coordination/CHATGPT_HANDOFF_FEATURE_FUSION_FF00.md), [final review](../coordination/CHATGPT_HANDOFF_FE_FINAL_REVIEW.md) |

## 3. TODO

| 技术方向 | 内容概要 | 优先级 |
|---|---|---|
| Feature Fusion | 等待 ChatGPT/Sol 给出单一、受限的 review 结论；仅在明确授权后探索如何利用 Temporal/Spatial 残差信号并保护 TKE。 | P1 |

## 4. 相关文档

- [Feature Engineering 长期技术总结](README.md)
- [Feature Engineering final review](../coordination/CHATGPT_HANDOFF_FE_FINAL_REVIEW.md)
- [Track 1 实验注册表](../track1_experiment_registry.md)
