# NEXT_ACTION

## Goal

无人值守完成 P0-A + N2 validation late-stage LR A/B，为明天 full-data SOTA continuation 决定是否在平台期降学习率。

## Tasks

1. 确认工作区无未知未提交改动，执行 `git pull --rebase origin main`。
2. 阅读 `docs/CHATGPT_CODEX_WORK_PROTOCOL.md` 和本文件；运行：
   `pytest -q tests/test_sota_lr_screen.py tests/test_p0a_full_training.py`
3. 只解析真实环境路径并设置：`DATA_ROOT`、`MANIFEST`、`KIT_ROOT`、`RESUME_CHECKPOINT`、`OUT_ROOT`。`RESUME_CHECKPOINT` 必须是 P0-A + N2 validation update `18860` 且包含 optimizer state。
4. 在当前已验证的 CUDA / container 环境中启动：
   `bash tools/run_sota_lr_screen_20260904.sh`
   建议用 `nohup` / detached 方式，并把 launcher stdout/stderr 写到 `OUT_ROOT` 之外的独立日志文件。
5. 启动后只做一次检查：进程已启动，`launcher_status.json` 为 `RUNNING`。记录 PID、命令、launcher log、`OUT_ROOT`。
6. 在 `docs/sota迭代/README.md` 追加一条简短 `RUNNING` 记录并 commit/push `main`，然后停止轮询。

## Constraints

- A/B 固定为同一 `18860` resume：Control `LR=1e-5`，Decay `LR=5e-6`，都训练到 `22960`。
- 固定 P0-A、N2、CNO、50/16 split、seed `20260901`、micro-batch `4`、accumulate `2`。
- launcher 会校验 frozen manifest SHA、v9 scorer SHA、resume iteration 和 optimizer state；任一失败立即停止，不绕过检查。
- 只允许路径、CUDA、conda/container 等环境适配；不得改变实验定义。
- 今晚不做 full-data 长训、SPS、H1、FF-01、新 Loss、locked-final/private-test 或 Codabench。

## Deliverables

无人值守脚本最终应在 `OUT_ROOT` 产生：

- `preflight.json`
- `smoke_decay_lr5e6/`
- `control_lr1e5/`
- `decay_lr5e6/`
- `lr_ab_summary.json`
- `launcher_status.json`

其中最终状态只能是 `REVIEW_REQUIRED` 或 `FAILED`，脚本不自动选择赢家。

## Stop

完成一次启动检查并记录 `RUNNING` 后，Codex 停止主动工作，不持续轮询，不启动下一轮实验。明早由 ChatGPT/Sol review `lr_ab_summary.json` 后决定 full-data continuation 和后续 SPS / submission。
