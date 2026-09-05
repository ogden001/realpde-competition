# NEXT_ACTION

## Goal

只评估已经训练完成的 **v5 base Adaptive Uncertainty Head** 是否能提升当前 canonical SPS。**禁止重训、corrector full refit、package、locked-final/private-test、Codabench。**

先读：
- `docs/sota迭代/reviews/overnight_integrated_20260905_v5/SOL_REVIEW.md`
- `docs/sota迭代/reviews/overnight_integrated_20260905_v5/README.md`

状态：`IMPLEMENT_AND_EXECUTE_AUTHORIZED / CALIBRATION_ONLY`

## Tasks

1. 使用 v5 已训练完成的 **base uncertainty head@1400**，不得重新训练。
2. 冻结 canonical validation backbone@30900、50/16 manifest、16 Dev / 659 windows。
3. 只扫描固定 28 组 calibration：
   - floor：`0 / 0.0025 / 0.005 / 0.0075`
   - mult：`0.5 / 1 / 1.5 / 2 / 2.5 / 3 / 4`
4. 使用 official v9 SPS scorer，输出完整 grid：SPS、coverage、mean width。
5. 明确比较 canonical static reference：
   - `half_width = 0.0075 + 0.02 * abs(prediction)`
   - Dev SPS `39.112385`
6. 提交 evidence 到：
   `docs/sota迭代/reviews/overnight_integrated_20260905_v5/`
   至少包括 calibration CSV/JSON、best row、head/probe SHA256、scorer/manifest/checkpoint provenance、review log。
7. 最终状态只写 `REVIEW_REQUIRED`，等待 Sol 决定是否值得进入 package/submission review。

## Constraints

- 不使用 v4 calibration 结果或 v4 权重。
- 不修改模型、head、Loss、grid 或 backbone。
- 不训练 corrected head。
- 不启动 full corrector refit。
- 不构建 package，不提交 Codabench。

## Stop

calibration evidence commit + push `main` 后停止，返回 commit SHA。
