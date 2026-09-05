# Inference / Submission 优化概要

## 1. 整体思路

提交工程应优先保证 starting-kit 接口、冷启动/推理耗时、区间输出与打包可复现；本地 SPS 或综合 proxy 不能替代未公开公式的线上 `final_score`。

## 2. 当前结论

| 技术方向 | 内容概要 | 关键实验结果 | 状态 | 详细文档 |
|---|---|---|---|---|
| 当前线上 SOTA | 当前最好正式结果为 `P0-A + N2 + CNO + full@43260 + v5 Adaptive Uncertainty Head@1400`。 | Final `76.694784`；Rel-L2 `93.434384`；TKE `77.588799`；MVPE `92.519563`；Time `87.066646`；SPS `29.519724`。 | **KEEP / SOTA** | [SOTA 迭代](../sota迭代/README.md) |
| Adaptive Uncertainty | 在 full@43260 backbone prediction 完全不变的前提下，用 learned sigma 替换 static bounds：`half_width_uv = 0.0025 + sigma`，pressure half-width `0`。 | Prediction parity `max_abs_diff=0.0`；线上 SPS `27.545059 → 29.519724`，Final `76.149726 → 76.694784`。 | **KEEP** | [Adaptive package review](../sota迭代/reviews/full43260_adaptive_package_20260905/README.md) |
| P0-A + N2 full@43260 | full-data late continuation 已形成当前 backbone；相比早期 full@15300，Rel/MVPE/SPS 综合明显改善并保持强 TKE。 | 2026-09-04 static-bounds 版本 Final `76.149726`，随后仅替换 uncertainty bounds 后提升到 `76.694784`。 | KEEP | [Submission log](../submission_log.md) |
| Residual Corrector | V5 canonical validation 中 Rel-L2/MVPE 有明显改善，但 TKE aggregate 恶化 `5.06%`，且 6/16 trajectory 超保护阈值。 | 不进入当前 SOTA package。 | NO-GO / PARKED_SIGNAL | [V5 review](../sota迭代/reviews/overnight_integrated_20260905_v5/SOL_REVIEW.md) |
| 模型选择风险 | UNet 后处理的本地 proxy 看似较好，线上隐藏物理分弱于 CNO。 | UNet final `74.48384`。 | NO-GO | [Submission log](../submission_log.md) |
| 评分边界 | starting kit 可复现五个子分，但不公开 leaderboard final-score 组合。 | 不应承诺或优化自定义 `mean5_proxy`。 | KEEP | [Submission log](../submission_log.md) |

## 3. 当前提交原则

1. 新的 SOTA merge 先在固定 50 Train / 16 Dev 上用同一 recipe 训练约 2 小时，与历史 SOTA 50/16 结果直接比较。
2. 50/16 整体 OK 后立即启动同 recipe 的全量训练，不把 merge 重新变成长期研究。
3. 当前默认保留 Adaptive Uncertainty Head 作为 SPS 组件；除非新 backbone 的误差分布明显变化并有证据需要重训/重校准。
4. package 前只做必要 correctness smoke，线上失败或回退可以接受并记录。

## 4. 相关文档

- [SOTA 迭代](../sota迭代/README.md)
- [Codabench 提交记录](../submission_log.md)
- [Adaptive package review](../sota迭代/reviews/full43260_adaptive_package_20260905/README.md)
- [项目 README](../../README.md)
