# Loss / 指标 / 离线评估概要

## 1. 整体思路

离线选择必须以 Track 1 starting kit v9 `scoring.py` 的公开子分为准，同时检查 Rel-L2、TKE、MVPE；runtime 与 SPS 也属于最终比赛目标。自定义聚合 proxy 只能辅助诊断。

## 2. 当前结论

| 技术方向 | 内容概要 | 关键实验结果 | 状态 | 详细文档 |
|---|---|---|---|---|
| 官方 scorer | v9 scorer 是本地评价权威，prediction 口径为 `(N,20,32,64,3)`。 | scorer SHA 已冻结并记录于 registry/FF-00。 | KEEP | [Registry](../track1_experiment_registry.md), [FF-00](../coordination/CHATGPT_HANDOFF_FEATURE_FUSION_FF00.md) |
| TKE 权衡 | 降低 LOCAL3 的 `λ_TKE` 至 `0.001` 改善 TKE/MVPE，仍未改善 Rel-L2，不能作为通用 loss 结论。 | Rel-L2 `-7.286%` vs PERSIST，screen early stop。 | NO-GO | [balanced LOCAL3](../coordination/CHATGPT_HANDOFF_POINT_LOCAL3_BALANCED_L001.md) |
| Feature-aware objective | 历史损失重复审计已完成；尚无已验证的新 objective，任何尝试须单独授权。 | FF-00 未训练、未产生性能结论。 | REVIEW | [FF-00 audit](../coordination/CHATGPT_HANDOFF_FEATURE_FUSION_FF00.md) |

## 3. TODO

| 技术方向 | 内容概要 | 优先级 |
|---|---|---|
| 指标门槛 | 后续已授权候选继续使用 all-three-metric 保护、trajectory-level 证据与 matched control。 | P0 |

## 4. 相关文档

- [提交记录的评分教训](../submission_log.md)
- [Track 1 实验注册表](../track1_experiment_registry.md)
