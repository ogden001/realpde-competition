# NEXT_ACTION

## Goal

无人值守继续 P0-A + N2 全量 82 trajectory 模型，从 update `15300` 尽量训练到 `43260`，产出明早可 review 的多 checkpoint submission candidates。

## Tasks

1. 确认工作区无未知未提交改动，执行 `git pull --rebase origin main`。
2. 阅读：
   - `docs/CHATGPT_CODEX_WORK_PROTOCOL.md`
   - 本文件
   - `docs/coordination/CHATGPT_HANDOFF_T1_P0A_N2_FULL15300_CODABENCH.md`
3. 运行：
   `pytest -q tests/test_p0a_full_training.py tests/test_sota_full_night.py tests/test_runtime_context.py`
4. 只解析真实环境路径并设置：`DATA_ROOT`、`KIT_ROOT`、`RESUME_CHECKPOINT`、`OUT_ROOT`。
   - `RESUME_CHECKPOINT` 必须是 full-data update `15300` 的 `model_last.pth`；
   - SHA256 必须为 `3ea4e4def03ae2f1d970975e4217358e1d762b88a69bdddfdf844d551baaa3e4`；
   - 必须包含 optimizer state；
   - 上一次失败 artifact 保留不动；本次必须使用新的 `OUT_ROOT`，例如 `.../sota_full_night_20260904_retry1`。
5. 在已验证的 CUDA/container 环境中 detached 启动：
   `bash tools/run_sota_full_night_20260904.sh`
6. 启动后只检查一次：launcher 已进入 `RUNNING`。记录 PID、命令、launcher log、`OUT_ROOT`，并把 `runtime_snapshot.json` 中的 GPU、82/3383、scorer SHA、15300 checkpoint resume 信息一并反馈，然后停止轮询。

## Constraints

- 固定 P0-A、四项 N2、CNO、seed `20260901`、LR `1e-5`、micro-batch `4`、accumulate `2`。
- continuation 必须复现 full@15300 历史 P0-A grid-spacing 数值语义；不得改 checkpoint feature_config 或绕过 spacing 校验。
- 正式训练必须严格为 `82 trajectories / 3383 windows`；preflight 或 runner 任一校验失败立即停止，不绕过。
- 目标 update `43260`；硬训练时间上限 `21300s`（5h55m）。时间到时允许安全停止并保存最后真实完成的 update。
- 必须保留 milestones：`31100 / 36500 / 37850 / 40560 / 43260`；另保留 `model_last.pth`。
- 不根据 full-data training loss 自行选择 checkpoint。
- 今晚不做 LR screen、H1、FF-01、新 Loss、SPS、locked-final/private-test 或 Codabench。
- 只允许路径、CUDA、conda/container 等环境适配；不得改变实验定义。

## Deliverables

`OUT_ROOT` 最终应包含：

- `runtime_snapshot.json`
- `preflight.json`
- `smoke_full_15300/`
- `full_15300_to_43260/`
  - `artifact_manifest.json`
- `full_night_summary.json`
- `launcher_status.json`

最终 launcher 状态只能是 `REVIEW_REQUIRED` 或 `FAILED`，不自动选择 submission checkpoint。

## Stop

完成一次启动检查并记录 `RUNNING` 后，Codex 停止主动工作。明早由 ChatGPT/Sol review `full_night_summary.json`、`artifact_manifest.json` 和已落盘 checkpoints，再进入 SPS / package / submission。
