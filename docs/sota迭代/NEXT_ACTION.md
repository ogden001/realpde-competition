# NEXT_ACTION

## Goal

把当前线上 SOTA backbone `P0-A + N2 full@43260` 与已经验证通过的 **v5 base Adaptive Uncertainty Head@1400** 组合成一个 submission candidate，并完成 clean-room smoke。**禁止重训、corrector、recalibration、locked-final/private-test、Codabench。**

先读：
- `docs/sota迭代/reviews/overnight_integrated_20260905_v5/SOL_REVIEW.md`
- `docs/sota迭代/reviews/overnight_integrated_20260905_v5/README.md`
- `docs/sota迭代/reviews/overnight_integrated_20260905_v5/BASE_UNCERTAINTY_CALIBRATION.md`
- `docs/CHATGPT_CODEX_WORK_PROTOCOL.md`

状态：`IMPLEMENT_AND_EXECUTE_AUTHORIZED / PACKAGE_ONLY`

## Tasks

1. 同步 `main`，确认工作区无未知改动，`HEAD == origin/main`，并确认本任务 required base commit 是当前 HEAD 的祖先。
2. 从已有 artifact/manifest 精确定位并记录 SHA256：
   - full@`43260` checkpoint，必须是当前线上 `76.149726` 对应 backbone；预期 checkpoint SHA256：`50b692e236d5df9285a5cee976a51e3457a7eeed0f87d55b6568745077645d71`；
   - v5 validation probe 中的 `base_head_state_dict@1400`，不得使用 v4 或 corrector/corrected-head 权重。
3. 做最小 bounded packaging 实现。最终推理语义固定为：
   - backbone 使用 full@43260 自身 checkpoint `feature_config`，不得重新推导 `dx/dy`；
   - `prediction = full@43260 backbone prediction`，pressure channel 强制为 0；
   - uncertainty input 固定为 `concat([Past20 x(3ch), flow_features(prediction_uv)(12ch)])`，共 15 channels；
   - `sigma = AdaptiveUncertaintyHead(hidden=32, blocks=2)` 的 v5 `base_head_state_dict` 输出，保持 `exp(raw).clamp(1e-4, 1.0)` 语义；
   - `uv_half_width = 0.0025 + 1.0 * sigma`；
   - pressure half-width = `0`；
   - 返回 `{"prediction", "lower", "upper"}`；
   - **不得包含 ResidualCorrector3D。**
4. 加/改最小必要 tests，至少证明：
   - adaptive package 的 `prediction` 与 full@43260 直接 backbone path 数值等价，`max_abs_diff <= 1e-6`；
   - shape `(1,20,32,64,3)`、float32、finite、pressure-zero、deterministic；
   - sigma finite 且正；
   - bounds 与 `floor=0.0025, mult=1` 公式逐元素匹配；pressure bounds 为 0；
   - package 不携带 corrector 权重/执行路径。
5. 使用新的 `OUT_ROOT` 构建唯一 candidate ZIP，不覆盖之前 static package。记录 checkpoint/head/ZIP SHA256、size、inventory 和 execution commit。
6. 在独立临时目录解压 ZIP 后做 clean-room smoke；同时做一次简单 warmed runtime 对照，报告 adaptive package 与现有 full@43260 static package 的推理耗时比例，只记录事实，不据此自行调参。
7. 将 package/smoke 证据写入：
   `docs/sota迭代/reviews/full43260_adaptive_package_20260905/`
   至少提交：`README.md`、build/smoke 摘要、完整 `*.review.log`、SHA256 provenance。ZIP/checkpoint/head 权重不得提交 Git。
8. 完成后 commit + push `main`，确认工作树干净。

## Constraints

- 不训练任何模型或 uncertainty head。
- calibration 固定 `floor=0.0025, mult=1`，不得重新扫描 grid。
- 不使用 Residual Corrector，不做 full corrector refit。
- 不改变 full@43260 prediction；若 prediction parity 失败立即停止。
- 不访问 locked-final/private-test，不提交 Codabench。
- 不修改已有 full@43260 checkpoint、v5 probe 或既有 submission ZIP。
- 若 exact checkpoint/head 无法唯一定位、SHA/config 不符、tests/smoke 失败，停止并报告，不用相邻资产替代。

## Deliverables

- 一个新的 adaptive submission ZIP；
- build report + clean-room smoke report；
- prediction parity / bounds-match / runtime evidence；
- Git 中完整 review evidence；
- 最终状态：`READY_FOR_SUBMISSION_REVIEW` 或 `FAILED`。

## Stop

package + clean-room smoke + evidence commit/push 完成后停止。**不自动提交 Codabench。**
