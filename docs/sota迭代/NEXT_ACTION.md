# NEXT_ACTION

## Goal

SOTA-V2 已完成正式 Codabench 提交并刷新线上 SOTA。下一步先做**线上结果复盘与下一轮 Merge Worthiness Review**，不要立即启动新的 full-data 训练。

状态：`ONLINE_SOTA / REVIEW_REQUIRED`

## Current online anchor

- Final：`77.314446`
- Rel-L2：`93.816645`
- TKE：`79.164203`
- MVPE：`93.411176`
- Time：`86.898836`
- SPS：`30.319572`

相对上一线上 SOTA `76.694784`：

- Final：`+0.619662`
- Rel-L2：`+0.382261`
- TKE：`+1.575404`
- MVPE：`+0.891613`
- Time：`-0.167810`
- SPS：`+0.799848`

## Frozen current SOTA recipe

- Predictive backbone：`Dense-All + P0-A + MF-CNO + N2 + vorticity + Stage-B`
- Full-data checkpoint：`@53582`
- Full checkpoint SHA256：`f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce`
- Adaptive Uncertainty Head：fresh head `@1400`
- Bounds：`half_width_uv = 0.0025 + sigma`，pressure half-width `0`
- Final clean ZIP SHA256：`9cfc055c6232d2b0aef9f88f1cb3de2aae883b7cff7f02659ed8b9cf85b3ed55`

## Next review questions

1. 把这次线上增益拆成 predictive gain 与 SPS gain，确认下一轮主要瓶颈。
2. 以 `77.314446` 作为新的线上 anchor，重新审视已有 PARKED / PROMISING 方向是否仍值得 merge。
3. 只有存在明显预期线上收益时才启动下一次 50/16 → full-data → package → Codabench 周期。
4. 不再以 full@43260 或 `76.694784` 作为默认 SOTA 基线。

## Constraints

- 当前 SOTA package 已完成，不需要重新打包或重复提交。
- 不基于 Codabench 做高频参数搜索。
- 下一次 full-data 训练必须先通过 Merge Worthiness Review。
- 保持 locked-final/private test 边界。

## References

- `docs/submission_log.md`
- `docs/sota迭代/reviews/sota_v2_adaptive_20260916/README.md`
- `docs/coordination/CHATGPT_HANDOFF_SOTA_V2_ADAPTIVE_20260916.md`
- `docs/inference/inference概要.md`
