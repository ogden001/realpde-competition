# NEXT_ACTION

## Goal

完成当前 adaptive candidate 的**提交前准备**：确认最终 ZIP、复制到本地 Mac、核对 SHA256。**不再训练、不重打包已有正确 ZIP、不提交 Codabench。**

状态：`EXECUTE_AUTHORIZED / SUBMISSION_PREP_ONLY`

## Tasks

1. 同步 `main`，确认工作区干净、`HEAD == origin/main`。
2. 使用已经 clean-room smoke PASS 的唯一 ZIP：
   - SHA256：`3285ad3a424988ab35061337ca836c23b5f7db04773246167da3a9f8eaa2178a`
   - 若该 exact ZIP 仍存在，**不要重新 build**。
3. 确认该 ZIP 的 SHA256 和大小与 review evidence 一致。
4. 将 ZIP 复制到本地 Mac：
   - `/Users/oukairi/project/RealPDE Competion/artifacts/full43260_adaptive_package_20260905/submission.zip`
5. 在 Mac 上再次计算 SHA256，必须仍为：
   - `3285ad3a424988ab35061337ca836c23b5f7db04773246167da3a9f8eaa2178a`
6. 返回最终本地路径、SHA256、文件大小和 `READY_TO_UPLOAD`。

## Constraints

- 不训练、不 recalibration、不使用 corrector。
- 不重新跑完整测试，不做 benchmark，不扩展 package 流程。
- 不访问 locked-final/private-test。
- 不提交 Codabench。
- 只有 exact ZIP 缺失或 SHA 不匹配时才停止并报告，不自行重建替代包。

## Deliverables

- Mac 上最终 `submission.zip`。
- 本地 SHA256 核验结果。
- 状态：`READY_TO_UPLOAD` 或 `FAILED`。

## Stop

本地文件与 SHA256 核对完成后立即停止。
