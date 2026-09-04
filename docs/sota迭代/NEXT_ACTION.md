# NEXT_ACTION

## Goal

在冻结的 50/16 validation dev 上，用官方 v9 scorer 对 P0-A + N2 validation update `30900` 做一个小范围 SPS bounds screen，为 full@`43260` primary / `40560` backup 的正式提交 bounds 提供离线证据。

## Tasks

1. 确认工作区无未知未提交改动，执行：
   - `git fetch origin`
   - `git pull --rebase origin main`
   - 确认 `HEAD == origin/main`
   - 确认 `e4f63a5a7e4d5a98449c249937729464a4b8c668` 已包含在当前 HEAD。
2. 阅读：
   - `docs/CHATGPT_CODEX_WORK_PROTOCOL.md`
   - 本文件
   - `docs/sota迭代/README.md`
3. 运行：
   - `pytest -q tests/test_p0a_submission_package.py tests/test_sota_sps_screen.py tests/test_runtime_context.py`
   - `bash -n tools/run_sota_sps_screen_20260904.sh`
4. 只解析真实环境路径并设置：`DATA_ROOT`、`MANIFEST`、`KIT_ROOT`、`VALIDATION_CHECKPOINT`、`OUT_ROOT`。
   - `MANIFEST` 必须是冻结 50/16 manifest；
   - `KIT_ROOT/scoring.py` SHA256 必须为 `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`；
   - `VALIDATION_CHECKPOINT` 必须是 P0-A validation update `30900`，优先从既有 `p0a_n2_simreal_validation_20260903/continuation_10300_to30900/run/` artifact 定位；不得换成其他 update；
   - `OUT_ROOT` 必须是新目录。
5. 执行：
   `bash tools/run_sota_sps_screen_20260904.sh`
6. 完成后汇报 `sps_screen.json` 中：
   - validation raw Rel-L2 / TKE / MVPE；
   - fallback SPS score / coverage；
   - 每个显式 bounds candidate 的 SPS score / coverage；
   - best local SPS candidate 及相对 fallback delta。

## Constraints

- 本阶段只做 SPS screen，不训练模型。
- candidate 固定为脚本中的小范围集合：fallback、`abs=0.005/0.0075/0.010/0.0125 rel=0`，以及 `abs=0.0075 rel=0.01/0.02`；不得扩展矩阵。
- SPS 必须直接使用 `KIT_ROOT/scoring.py` 的 `aggregate_sps()` / `score_sps()`；不得改用旧自定义 proxy。
- P0-A feature config 必须从 validation checkpoint 读取，不重新推导 spacing。
- 不访问 locked-final/private-test，不提交 Codabench。
- 不修改或覆盖昨晚 full@`43260/40560` checkpoints。
- 不自动打包、不自动选择线上提交方案；结果必须交给 ChatGPT/Sol review。

## Deliverables

`OUT_ROOT` 最终应包含：

- `runtime_snapshot.json`
- `sps_screen.json`
- `status.json`

最终状态只能是 `REVIEW_REQUIRED` 或 `FAILED`。

## Stop

`sps_screen.json` 生成并汇报后立即停止。由 ChatGPT/Sol 选择 bounds，并决定 full@`43260` primary / `40560` backup 的 package + clean smoke + Codabench submission。
