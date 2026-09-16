# Inference / Submission 优化概要

## 1. 整体思路

提交工程应优先保证 starting-kit 接口、冷启动/推理耗时、区间输出与打包可复现；本地 SPS 或综合 proxy 不能替代未公开公式的线上 `final_score`。

## 2. 当前结论

| 技术方向 | 内容概要 | 关键实验结果 | 状态 | 详细文档 |
|---|---|---|---|---|
| 当前线上 SOTA | 当前最好正式结果为 `Dense-All + P0-A + MF-CNO + N2 + vorticity + Stage-B + full@53582 + fresh Adaptive Uncertainty Head@1400`。 | Final `77.314446`；Rel-L2 `93.816645`；TKE `79.164203`；MVPE `93.411176`；Time `86.898836`；SPS `30.319572`。 | **KEEP / NEW SOTA** | [SOTA-V2 review](../sota迭代/reviews/sota_v2_adaptive_20260916/README.md) |
| SOTA-V2 predictive backbone | 50/16 Dev sweet spot `@32500` 映射到 all-82 Dense-All full refit `@53582`；MF-CNO + N2 + vorticity + Stage-B。 | 相对上一线上 SOTA，Rel-L2 `+0.382261`、TKE `+1.575404`、MVPE `+0.891613`；说明新 predictive backbone 成功迁移到 hidden evaluation。 | **KEEP** | [SOTA-V2 review](../sota迭代/reviews/sota_v2_adaptive_20260916/README.md) |
| Adaptive Uncertainty | 用 50 Train canonical windows 训练 fresh head，在 16 Dev 固定 28-row grid 校准；最终 `half_width_uv = 0.0025 + sigma`，pressure half-width `0`。 | Dev SPS `42.124892 → 45.070082`；线上 SPS `29.519724 → 30.319572`；package prediction parity A/B 均为 `0.0`。 | **KEEP** | [SOTA-V2 review](../sota迭代/reviews/sota_v2_adaptive_20260916/README.md) |
| Previous full@43260 SOTA | `P0-A + N2 + CNO + full@43260 + v5 Adaptive Uncertainty Head@1400`。 | Final `76.694784`；现已被 SOTA-V2 的 `77.314446` 超越。 | PREVIOUS SOTA | [Submission log](../submission_log.md) |
| Residual Corrector | V5 canonical validation 中 Rel-L2/MVPE 有明显改善，但 TKE aggregate 恶化 `5.06%`，且 6/16 trajectory 超保护阈值。 | 不进入当前 SOTA package。 | NO-GO / PARKED_SIGNAL | [V5 review](../sota迭代/reviews/overnight_integrated_20260905_v5/SOL_REVIEW.md) |
| 模型选择风险 | UNet 后处理的本地 proxy 看似较好，线上隐藏物理分弱于 CNO。 | UNet final `74.48384`。 | NO-GO | [Submission log](../submission_log.md) |
| 评分边界 | starting kit 可复现五个子分，但不公开 leaderboard final-score 组合。 | 不应承诺或优化自定义 `mean5_proxy`。 | KEEP | [Submission log](../submission_log.md) |

## 3. 当前提交原则

1. 新的 SOTA merge 先在固定 50 Train / 16 Dev 上形成可比证据，再决定是否值得 full-data refit。
2. 当前 predictive anchor 已升级为 SOTA-V2 full@53582；下一轮新增变量必须相对这个更强 anchor 评估，不能继续拿 full@43260 当默认基线。
3. Adaptive Uncertainty 继续作为默认 SPS 组件；若 backbone 发生明显变化，仍按 `50 Train head training → 16 Dev calibration → full backbone package` 的非泄漏流程重新训练/校准。
4. package 前继续坚持 clean rebuild + A/B real-fixture parity；uncertainty 绝不能改变 prediction。

## 4. 相关文档

- [SOTA 迭代](../sota迭代/README.md)
- [Codabench 提交记录](../submission_log.md)
- [SOTA-V2 adaptive online review](../sota迭代/reviews/sota_v2_adaptive_20260916/README.md)
- [SOTA-V2 handoff](../coordination/CHATGPT_HANDOFF_SOTA_V2_ADAPTIVE_20260916.md)
- [项目 README](../../README.md)
