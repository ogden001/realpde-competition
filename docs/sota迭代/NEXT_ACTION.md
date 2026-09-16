# NEXT_ACTION

## Goal

复刻同事已在线验证有效的 SPS 基础配方到当前 SOTA-V2：先在固定 50/16 上验证 `stride=1 uncertainty training`，若通过 Gate，再给 full@53582 训练匹配的 full-specific uncertainty head 并生成候选包。

状态：`READY_FOR_EXECUTION / REVIEW_REQUIRED`

## Tasks

1. 按 `AGENTS.md` 做 preflight：工作区干净、`git pull --rebase origin main`、`git push --dry-run origin HEAD:main` 成功。
2. 先运行测试：
   - `pytest -q tests/test_sps_stride1_replica.py tests/test_dense_all_windows.py tests/test_adaptive_probe.py`
   - 再运行仓库现有相关测试；若失败，只允许做不改变实验语义的 bounded 修复并记录。
3. 执行 `tools/realpde_sps_stride1_replica.py`：
   - Phase 1：固定 50 Train / 16 Dev、SOTA-V2 validation backbone@32500、h32/b2、Gaussian NLL、1400 updates、固定 28-row calibration grid；唯一实验变量是 uncertainty head 训练窗口由 canonical stride20 改为所有合法 stride1 (`dense_all`)。
   - Phase-1 Gate：Dev SPS 相对当前 `45.0700816004` 至少 `+1.5`，且 mean UV width 不超过当前 `0.0235833097` 的 `1.15x`。
   - Phase 1 `NO_GO`：立即停止，不做 Phase 2、不打包。
   - Phase 1 `GO`：自动进入 Phase 2，用冻结 full SOTA-V2@53582 自己的 residual、all-82 released PIV、stride1 训练 fresh h32/b2 head@1400；禁止在 full 数据重新搜索 floor/mult，必须复用 Phase-1 clean 16-Dev 选出的 bounds。
4. 若 Phase 2 完成，使用同一 runner 生成 candidate package；随后用 `tools/verify_sota_v2_adaptive_package.py` 做 clean-room smoke / prediction parity。prediction 必须与当前 full SOTA 完全保持，目标 parity `max_abs_diff <= 1e-6`。
5. 将轻量 evidence（Phase-1 calibration/grid、by-horizon/by-channel、gate、Phase-2 head provenance、package/smoke summary）写入 `docs/sota迭代/reviews/sps_stride1_replica_20260916/`，更新本方向 README 的实验事实，commit + push `origin/main`。

## Constraints

- 不训练或修改 predictive backbone。
- 不改变 h32/b2、Gaussian NLL、1400 updates、固定 28-row calibration grid。
- 不引入 OOF、pinball、非对称区间、direct SPS loss 或新模型结构。
- 不访问 locked-final/private Future20；不提交 Codabench。
- 不因 Phase-1 `NO_GO` 自行扩大实验范围。
- 大 checkpoint / package / raw log 留在远程 artifact 路径，不写入 Git。
- 实现语义若需要变化，立即停止并报告，不自行决定。

## Deliverables

- `phase1_50_16_stride1/calibration_summary.json`
- `phase1_50_16_stride1/calibration_grid.csv`
- `phase1_50_16_stride1/by_horizon.csv`
- `phase1_50_16_stride1/by_channel.csv`
- 若 GO：`phase2_full_specific_stride1/full_head_summary.json`
- 若 GO：candidate package build summary + clean-room smoke report
- Git review evidence + commit SHA

## Stop

返回 `REVIEW_REQUIRED`。不要提交 Codabench；由 ChatGPT/Sol 复核后再决定是否消耗唯一一次线上 SPS A/B 提交。
