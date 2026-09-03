# Inference / Submission 优化概要

## 1. 整体思路

提交工程应优先保证 starting-kit 接口、冷启动/推理耗时、区间输出与打包可复现；本地 SPS 或综合 proxy 不能替代未公开公式的线上 `final_score`。

## 2. 当前结论

| 技术方向 | 内容概要 | 关键实验结果 | 状态 | 详细文档 |
|---|---|---|---|---|
| 线上性能锚点 | 简单 CNO 目前是最好已知正式结果。 | `submission_cno_tke1200_bounds_rel00.zip`：final `75.58455`。 | KEEP | [Submission log](../submission_log.md) |
| 模型选择风险 | UNet 后处理的本地 proxy 看似较好，线上隐藏物理分弱于 CNO。 | UNet final `74.48384`。 | NO-GO | [Submission log](../submission_log.md) |
| 评分边界 | starting kit 可复现五个子分，但不公开 leaderboard final-score 组合。 | 不应承诺或优化自定义 `mean5_proxy`。 | KEEP | [Submission log](../submission_log.md) |

## 3. TODO

| 技术方向 | 内容概要 | 优先级 |
|---|---|---|
| 新候选提交 | 只在用户明确授权后，以 official kit smoke test、运行时间记录与 submission log 为前置条件。 | P1 |

## 4. 相关文档

- [Codabench 提交记录](../submission_log.md)
- [项目 README](../../README.md)
