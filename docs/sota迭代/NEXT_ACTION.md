# NEXT_ACTION

## Goal

只做 `Residual Corrector + Adaptive Uncertainty` 的 **baseline parity audit**。**禁止重训、full refit、package、Codabench。**

先读：
- `docs/sota迭代/README.md`
- `docs/sota迭代/reviews/overnight_integrated_20260905/SOL_REVIEW.md`
- `docs/sota迭代/reviews/overnight_integrated_20260905/gate_result.json`

状态：`IMPLEMENT_AND_EXECUTE_AUTHORIZED / AUDIT_ONLY`

## Audit

1. 核验 v4 使用的 validation checkpoint：
   - SHA256 必须记录；
   - checkpoint iteration / feature_set / feature_config / manifest SHA；
   - 确认它是否就是历史 SOTA SPS screen 使用的 update `30900` checkpoint。

2. 在同一 frozen 16 Dev / 659 windows 上，对**同一个 base checkpoint**分别走：
   - 历史/标准 P0-A validation evaluation path；
   - 当前 `evaluate_adaptive_gate.py` 的 baseline path。

   两边都输出 Rel-L2 / TKE / MVPE。

3. 必须做 prediction-level parity：
   - 同一 batch 或完整 Dev 的 prediction max_abs_diff / mean_abs_diff；
   - 若能直接复用已保存 prediction artifact，优先复用，不要重新训练。

4. 解释为何当前 Gate baseline：
   - `0.12998666 / 0.66506779 / 0.16063145`

   与 SOTA README 注册的 validation@30900：
   - `0.11284460 / 0.50010282 / 0.08728255`

   不一致。

5. 若 baseline parity 成立，并确认当前 v4 base 就是 canonical 30900：
   - 补充 base vs corrected 的 16-trajectory Rel-L2 / TKE / MVPE 表；
   - 修正 review README 中 stale 的 execution commit / v3 OUT_ROOT；
   - 提交 audit evidence，状态 `REVIEW_REQUIRED`。

6. 若发现 v4 corrector 实际训练在错误 checkpoint / config / manifest 上：
   - 标记 `V4_INVALID_BASELINE`；
   - 不自动重跑 validation；
   - 提交证据后停止等待 Sol。

## Constraints

- 不修改模型、Loss、Feature、Gate 或 calibration grid。
- 不重新训练 corrector/head。
- 不访问 locked-final/private-test。
- 不启动 full refit/package，不提交 Codabench。

## Stop

完成 audit evidence commit + push `main` 后停止，等待 Sol review。
