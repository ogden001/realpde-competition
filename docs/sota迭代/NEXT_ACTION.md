# NEXT_ACTION

## Goal

在冻结的 50/16 validation dev 上，用官方 v9 scorer 对 P0-A + N2 validation update `30900` 做小范围 SPS bounds screen，为 full@`43260` primary / `40560` backup 提供 bounds 证据。

## Tasks

1. 确认工作区无未知未提交改动，执行：
   - `git fetch origin`
   - `git pull --rebase origin main`
   - 确认 `HEAD == origin/main`
   - 确认 commit `668e9d17cef79c1ff020c19e902b0836c13b2836` 已包含在当前 HEAD。
2. 运行：
   - `pytest -q tests/test_p0a_submission_package.py tests/test_sota_sps_screen.py tests/test_runtime_context.py`
   - `bash -n tools/run_sota_sps_screen_20260904.sh`
3. 复用已核验的 `DATA_ROOT`、`MANIFEST`、`KIT_ROOT`、validation update `30900` checkpoint；重新确认 scorer SHA256 为 `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`。
4. 保留失败 artifact `/home/chyfuture/realpde_runs/sota_sps_screen_20260904_run3` 不动，使用新的 `OUT_ROOT`。
5. 执行：
   `bash tools/run_sota_sps_screen_20260904.sh`
6. 汇报 `sps_screen.json`：raw Rel-L2/TKE/MVPE、fallback SPS/coverage、全部显式 bounds SPS/coverage、best candidate 及相对 fallback delta。

## Constraints

- 本阶段只做 SPS screen，不训练模型。
- candidate 固定：fallback、`abs=0.005/0.0075/0.010/0.0125 rel=0`、`abs=0.0075 rel=0.01/0.02`；不得扩展矩阵。
- SPS 必须直接使用 `KIT_ROOT/scoring.py` 的 `aggregate_sps()` / `score_sps()`。
- validation checkpoint 标记为 `P0-A`；feature config 必须复现 P0-A-only 语义，显式禁用 P0-B，不重新推导 spacing。
- 不访问 locked-final/private-test，不提交 Codabench。
- 不修改或覆盖 full@`43260/40560` checkpoints。
- 不自动打包、不自动选择线上提交方案。

## Deliverables

新 `OUT_ROOT` 包含：
- `runtime_snapshot.json`
- `sps_screen.json`
- `status.json`

最终状态只能是 `REVIEW_REQUIRED` 或 `FAILED`。

## Stop

`sps_screen.json` 生成并汇报后立即停止，由 ChatGPT/Sol review 后再决定 package / submission。
