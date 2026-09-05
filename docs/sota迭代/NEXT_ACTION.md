# NEXT_ACTION

## Goal

重新执行 `Residual Corrector + Adaptive Uncertainty`，但**所有 P0-A feature config 必须直接继承对应 backbone checkpoint**。先在 canonical validation@30900 上重跑；只有 baseline parity 与 Gate 都通过，才自动进入 full@43260 corrector refit + package。

状态：`IMPLEMENT_AND_EXECUTE_AUTHORIZED`

## Tasks

1. 同步 `main`，阅读：
   - `docs/CHATGPT_CODEX_WORK_PROTOCOL.md`
   - `docs/sota迭代/reviews/overnight_integrated_20260905/BASELINE_PARITY_AUDIT.md`
   - 本文件。

2. 最小修复 auxiliary pipeline：
   - `P0FeatureConfig` 的 `dx/dy` 及相关语义必须从当前 backbone checkpoint 的 `feature_config` 读取；
   - 禁止从 H5 grid、subsample 后 grid 或其他当前数据重新推导 spacing；
   - validation@30900 应为约 `dx=+0.001710832, dy=-0.001710832`；
   - full@43260 使用其自身 checkpoint 内记录的 `feature_config`。

3. 增加 invariant / regression test：
   - auxiliary path 的 feature config 与 backbone checkpoint 一致；
   - canonical validation path 与 adaptive baseline path 在同一 batch prediction `max_abs_diff <= 1e-6`；
   - 不允许 doubled-spacing 配置静默进入训练。

4. 刷新 runtime snapshot。使用 exact validation update `30900` checkpoint、冻结 50/16 manifest、官方 v9 scorer。正式训练前必须先通过 baseline parity preflight：
   - 16 Dev / 659 windows；
   - base raw errors 应复现约 `0.11284460 / 0.50010282 / 0.08728255`；
   - prediction parity `max_abs_diff <= 1e-6`。
   任一不满足立即 `FAILED`，不得训练。

5. parity 通过后，从头训练，不复用 v4 corrector/head 权重：
   - validation corrector：固定 `2400 updates`；
   - base uncertainty head：固定 `1400 updates`；
   - corrected uncertainty head：固定 `1400 updates`；
   - 其他结构、Loss、LR、seed、56-row calibration grid 均保持上一版冻结定义，不扩展矩阵、不挑中间 checkpoint。

6. 用 official-v9 Gate 评估 canonical base vs corrected：
   - Rel-L2 improvement >= `2%`；
   - MVPE improvement >= `2%`；
   - aggregate TKE degradation <= `2%`；
   - 16 Dev 中 TKE degradation > `15%` 的 trajectory 数 <= `2`。
   同时保存 16-trajectory Rel-L2/TKE/MVPE 表与 calibration 结果。

7. 若 Gate FAIL：提交完整证据并停止，状态 `REVIEW_REQUIRED / NO_FULL_REFIT`。

8. 仅当 parity + Gate 都 PASS：
   - exact full backbone 固定为 current SOTA full@`43260`；
   - corrector 在 82 trajectories / 3383 windows 上从头 refit，固定 `3960 updates`；
   - full corrector 必须继承 full@43260 checkpoint feature_config，不重新推导 spacing；
   - uncertainty 不在 all-82 上重新拟合，使用 validation 阶段对应的 base/corrected uncertainty head 与固定 56-row calibration 最优 floor/mult；
   - 构建并 clean-room smoke：
     - PRIMARY：full@43260 + full corrector + corrected adaptive uncertainty；
     - BACKUP：full@43260 + base adaptive uncertainty；
   - 不提交 Codabench。

9. 长任务证据必须完整落盘并提交 Git：
   - raw stdout/stderr 保留在远程 artifact；
   - 复制为 `docs/sota迭代/reviews/overnight_integrated_20260905_v5/*.review.log`；
   - review log 可脱敏机器绝对路径，但不得删除 metric / warning / error 行；
   - 因 `*.log` 被 ignore，使用 `git add -f` 明确提交；
   - 同目录提交 Gate、trajectory、calibration、runtime/provenance、package smoke 摘要和 README；
   - checkpoint / `.pth` / `.pt` / ZIP / NPZ / H5 不进入 Git。

10. 长任务结束后生成 `artifact_manifest.json`，handoff 记录实际 `EXECUTION_COMMIT`、checkpoint SHA256、resumable/inference-only 资产、命令、日志、PID、artifact 路径。最终 evidence commit + push `main`，工作区干净。

## Constraints

- v4 数值结论标记为 `V4_INVALID_BASELINE`，不得复用其模型权重或 promoted metrics。
- 不改变 corrector / uncertainty 的已冻结模型结构、Loss、预算、Gate 或 calibration grid。
- 不访问 locked-final/private-test。
- 不提交 Codabench。
- 不因中途指标自行调参、挑 checkpoint、扩实验矩阵。

## Deliverables

Git evidence：`docs/sota迭代/reviews/overnight_integrated_20260905_v5/`

远程 artifact 至少包含：
- `runtime_snapshot.json`
- validation parity / corrector / uncertainty / Gate / calibration artifacts
- 若 Gate PASS：full corrector refit、`artifact_manifest.json`、PRIMARY/BACKUP package + smoke artifacts

## Stop

最终状态只能是：
- `REVIEW_REQUIRED / NO_FULL_REFIT`，或
- `REVIEW_REQUIRED / READY_FOR_SUBMISSION_REVIEW`。

完成 evidence commit + push 后停止，等待 ChatGPT/Sol review。