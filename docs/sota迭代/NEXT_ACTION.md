# NEXT_ACTION

## Goal

只完成 `Residual Corrector + Adaptive Uncertainty` validation-family 的证据收口，供 Sol 最终复核。**禁止重训、full refit、package、Codabench。**

当前执行方报告：corrector Gate `PASS`，corrected uncertainty head 已完成 1400/1400；这些结论在 Git evidence 完整前仍视为 `REVIEW_REQUIRED`。

## Tasks

1. 完成固定 calibration grid：
   - floor：`0 / 0.0025 / 0.005 / 0.0075`
   - mult：`0.5 / 1 / 1.5 / 2 / 2.5 / 3 / 4`
   - 分别对 **base head + frozen backbone** 与 **corrected head + validation corrector** 评估；
   - 使用 official v9 SPS scorer；
   - 提交完整 grid，不只提交 best row；记录 best floor/mult、SPS 与必要 coverage/width 统计。

2. 将本次 v4 validation evidence 归档到：
   `docs/sota迭代/reviews/overnight_integrated_20260905/`

   至少包括：
   - execution commit / tests 结果 / provenance；
   - `gate_result.json`；
   - 16 trajectory Gate table；
   - baseline/corrected official Rel-L2、TKE、MVPE；
   - corrector、base head、corrected head training review logs；
   - calibration grid CSV/JSON；
   - runtime / artifact hashes；
   - raw log path + SHA256；
   - 简洁 README/handoff。

3. 最终状态只写：`REVIEW_REQUIRED`。

## Constraints

- 不改变任何模型、Loss、Feature、Gate、训练预算或 calibration grid。
- 不重新训练任何 head/corrector。
- 不访问 locked-final/private-test。
- 不启动 full refit/package，不提交 Codabench。
- 不只给 Luna 总结，必须提交 Sol 可直接复核的原始 JSON/CSV/log 证据。

## Stop

完成 evidence commit + push `main` 后停止，返回 commit SHA，等待 Sol review。
