# NEXT_ACTION

## Goal

修复 `Residual Corrector + Adaptive Uncertainty` 与冻结 reference 的实现漂移，并只重跑 50/16 validation。**本轮禁止 full refit、package、Codabench。**

先读：
- `docs/sota迭代/TEAMMATE_ADAPTIVE_PROBE_REFERENCE_20260905.md`
- `docs/sota迭代/reviews/overnight_integrated_20260905/SOL_REVIEW.md`
- `docs/CHATGPT_CODEX_WORK_PROTOCOL.md`

状态：`IMPLEMENT_AND_EXECUTE_AUTHORIZED / REVALIDATION_ONLY`

## Tasks

1. 修复并 TDD 以下冻结语义：
   - `residual_mse = MSE(delta_uv, target_uv - frozen_backbone_prediction_uv)`；不得使用 corrected prediction 构造 residual target。
   - **base uncertainty head** 只对 `base_pred` 的误差训练。
   - **corrected uncertainty head** 只允许在 corrector Gate PASS 后训练，并对 `base_pred + delta` 的误差训练。
   - uncertainty head 初始化必须满足：对任意输入，step-0 `sigma == 0.02`（允许浮点容差）；建议 final log-std projection weight=0、bias=`log(0.02)`。
   - `full=True` 只能训练 corrector，固定 3960 updates；不得重新拟合 uncertainty head。
   - package builder 必须显式区分 PRIMARY/BACKUP；BACKUP 不应用 corrector。
   - 修复 generated `submission.py` 中的 literal `+` 前缀问题，并加入 `py_compile`/import smoke test。

2. 新增/加强 tests，至少证明：
   - residual target 数学等价；
   - base head loss 与 corrector 输出无关；
   - corrected head 仅在 Gate PASS 路径存在；
   - fresh uncertainty head step-0 sigma≈0.02；
   - full mode 不训练/修改 uncertainty head；
   - BACKUP prediction 数值等价于 frozen backbone prediction；
   - generated PRIMARY/BACKUP `submission.py` 可编译、导入并通过 shape/finite/pressure-zero smoke。

3. tests 全过后，从头重跑 validation family：
   - frozen backbone@30900；
   - corrector 2400 updates，batch 8；
   - base uncertainty head 1400 updates，batch 8；
   - 固定 50/16 manifest、seed/optimizer/LR/feature/loss 语义不变。
   - 不复用当前 invalid corrector/head 权重。

4. 用**仓库内已提交的 evaluator/script**执行 validation@2400 Gate，不再用临时 heredoc 生成 evaluator。

5. Gate 固定标准：
   - Rel-L2 raw improvement >= 2%；
   - MVPE raw improvement >= 2%；
   - aggregate TKE degradation <= 2%；
   - dev trajectory 中 TKE degradation >15% 的数量 <=2。

6. 无论 Gate PASS/FAIL，本轮都停止于 Sol review：
   - PASS：只训练并评估 corrected uncertainty head 1400 updates，并完成固定 calibration grid evidence；**不启动 full refit/package**。
   - FAIL：不训练 corrected head；完成 base-head adaptive SPS evidence 后停止。

## Constraints

- 不访问 locked-final/private-test。
- 不提交 Codabench。
- 不改变 backbone、P0-A、N2、manifest、Gate、训练预算、calibration grid。
- 不扩大实验矩阵。
- 任何 reference 语义仍有歧义时停止并报告。
- 训练日志保留 raw log；Git review evidence 按当前项目日志协议提交，metric/warning/error 不得丢失。

## Deliverables

提交到 `docs/sota迭代/reviews/overnight_integrated_20260905/`：
- repaired validation run provenance / execution commit；
- tests 结果；
- corrector/base-head/corrected-head（若 Gate PASS）training review logs；
- official baseline/candidate Rel-L2/TKE/MVPE；
- 16 trajectory Gate table；
- `gate_result.json`；
- uncertainty calibration grid / SPS evidence；
- runtime / artifact hashes；
- 简洁 handoff，最终状态只写 `REVIEW_REQUIRED`。

完成后 commit + push `main`，停止等待 Sol 复核。
