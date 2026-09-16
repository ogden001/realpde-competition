# NEXT_ACTION

## Goal

执行 SOTA-V2 **all-82 released PIV full-data refit**。完整 recipe 已在 50/16 Dev 上验证，full-data 仅按 Dense epoch 对齐训练强度，不再选模或改科学配方。

状态：`READY_FOR_EXECUTION / REVIEW_REQUIRED`

## Tasks

1. 同步 `main`，确认工作区干净、`HEAD == origin/main`，并验证 GitHub 写权限。
2. 运行 `tests/test_sota_v2_full.py`、SOTA-V2/real-fixture/Dense-All/MF focused tests 与 `py_compile`。
3. 使用 `tools/realpde_sota_v2_full.py --preflight-only` 在 RTX 3090/CUDA 环境核验：
   - released PIV trajectories = `82`；
   - canonical windows = `3383`；
   - Dense-All windows = `66755`；
   - P0-A features = `20`；
   - official `sim_real_ft` SHA、scorer SHA、Direct→MF parity、pressure=0、finite loss/gradient 全部通过。
4. preflight PASS 后 detached 训练到 update `53582`：
   - Stage A：`1..49461`, LR `1e-5`；
   - Stage B：`49462..53582`, LR `3e-6`，额外 Rel-L2 `0.027514`；
   - effective batch 固定 `8`；
   - milestones：`12365,24730,32974,41217,49461,51109,53582`。
5. 训练完成后记录 final checkpoint path/SHA、runtime、peak GPU、training milestone evidence，commit + push `main`。

## Constraints

- 科学 recipe 固定：`Dense-All + P0-A + MF-CNO + N2 + vorticity + Stage-B extra Rel`。
- 初始化固定 official `sim_real_ft` Direct CNO：SHA256 `82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61`。
- scorer SHA256 固定 `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`。
- `53582` 是在 full refit 前由 Dev `32.5k` sweet spot 按 Dense epoch 映射后预先冻结的 primary checkpoint；**不要继续到 57704，也不要用 training loss 重新选 checkpoint**。
- 不访问 Codabench/private test，不做 SPS/Adaptive Uncertainty/package，不做消融或超参搜索。
- 不提交 checkpoint、dataset、大日志到 Git。

## Deliverables

- `docs/sota迭代/reviews/sota_v2_full_20260916/`：轻量 preflight/run metadata/runtime/status/training milestones/README。
- `docs/coordination/CHATGPT_HANDOFF_SOTA_V2_FULL_20260916.md`。
- full checkpoint 保留在远程 artifact 目录并记录 SHA256。
- 状态：`REVIEW_REQUIRED`。

## Stop

update `53582` 完成、轻量 evidence 和 handoff push 到 `main` 后立即停止，等待 ChatGPT/Sol 复核；不要自动进入 SPS 或提交。
