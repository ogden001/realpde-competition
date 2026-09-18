# NEXT_ACTION — SOTA-V3 FAST JOINT 50/16

Status: `READY_FOR_EXECUTION / REVIEW_REQUIRED`

`REQUIRED_COMMIT = 933a5a8c8f74910465ae707aa83c6878b910e68c`

## Goal

停止旧的 SOTA-V3 长程重训方案。只在项目固定 **50 Train / 16 Dev** 上快速验证：

`SOTA-V2 @32500 + mature residual @30000 + AoA -> 5k joint fine-tune -> Dev16`

本轮 **禁止 full-data train、禁止 package、禁止 Codabench**。只有用户审阅 50/16 结果并明确同意后，才另开 full train 任务。

## Frozen starting assets

必须复用已经验证过的资产，不重新训练它们：

- Dev backbone：SOTA-V2 `@32500`
  - SHA256: `6926722895611c38dee79cf16ce43b87c7de3d4a1305e9db431dadd4c1a7bf47`
  - 已知历史路径通常位于 SOTA-V2 integrated 50/16 run 下。
- Mature residual corrector：`@30000`
  - SHA256: `1f72b329d9a160b206a6e9f4fce5e780022b3bac7a9ce4bdf5b8c9efa30b4506`
  - backbone SHA 必须是上述 `@32500`。
- frozen 50/16 manifest SHA256:
  - `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`

如果 remote 上找不到正确 corrector checkpoint，只允许按 SHA 搜索现有 artifact；**禁止重训 30k corrector**。找不到则 STOP。

## Historical comparator

新方案必须超过我们已经得到的历史组合，而不是只超过裸 backbone：

`@32500 backbone + @30000 residual + alpha=1 + spatial_tke_map`

Dev16 raw error：

- Rel-L2: `0.0889103040`
- TKE: `0.4692927301`
- MVPE: `0.0711286217`

runner 在 update=0 会先 replay 这组指标；任一指标偏差 > `5e-6` 立即 STOP，避免用错资产。

## Joint fine-tune semantics

代码：`tools/realpde_sota_v3_fast_joint.py`

训练：

- Train：固定 50 trajectories / Dense-All。
- Dev：固定 16 trajectories，只在 update `0/1000/2000/3000/4000/5000` 做评估。
- 总预算：`5000 updates`。
- effective batch：`8`。
- 默认：micro-batch `2` × accumulation `4`。
- Backbone LR：`1e-6`。
- Corrector LR：`1e-5`。
- optimizer：AdamW。
- scheduler：CosineAnnealingLR over 5k。
- AoA：沿用冻结 V3 语义，`±2° / p=0.5`，Past20/Future20 同角度，仅训练启用。
- forward：
  `base -> residual(alpha=1) -> spatial_tke_map -> final`
- **Backbone 与 Residual 同时反向传播更新**。
- Loss：现有 SOTA Stage-B physical loss 直接作用于最终 `final` prediction：
  `N2 + vorticity + Stage-B extra Rel`。
- 不添加新的 auxiliary loss，不扫 LR，不扫 AoA，不扫 projection。

checkpoint selection：

`Rel/hist_Rel + TKE/hist_TKE + MVPE/hist_MVPE` 最小；同分取更早 milestone。

## GO_FULL Gate

本任务只是计算 Gate，**不得自动 full train**。

selected checkpoint 只有同时满足以下条件才输出 `FAST_JOINT_GO_FULL`：

1. 三指标 normalized average error 相对历史 comparator 至少下降 `1%`；
2. 任一单指标恶化不超过 `2%`；
3. 三项里至少 `2` 项严格改善。

否则：`FAST_JOINT_NO_GO`。

## Optional Dev SPS after point Gate

如果且仅如果 joint runner 输出 `FAST_JOINT_GO_FULL`，继续在 **同一 50/16** 上运行现有：

`tools/realpde_sota_v3_sps.py --mode dev`

使用 joint runner 输出的 selected backbone/corrector pair。

目的：顺手验证 teammate35 的 `head(base) + interval centered on final` 路径。SPS 完成后仍然 STOP，等待用户审阅；**不得进入 full mode**。

如果 point Gate = NO_GO，不跑 SPS，直接整理 evidence。

## Preflight

开始前：

```bash
git status --short
git fetch origin
git pull --rebase origin main
git rev-parse HEAD
git rev-parse origin/main
git merge-base --is-ancestor 933a5a8c8f74910465ae707aa83c6878b910e68c HEAD
git push --dry-run origin HEAD:main
```

要求：

- 工作区无未知修改；
- `HEAD == origin/main`；
- required commit 是 HEAD 祖先；
- dry-run push 成功；
- 确认旧的 `sota_v3_merge_20260918` GPU 训练进程已经停止，不允许两套训练抢同一张卡。

## Tests

至少：

```bash
PYTHONPATH=tools pytest -q \
  tests/test_sota_v3_fast_joint.py \
  tests/test_sota_v3_pipeline.py \
  tests/test_teammate_residual_transfer.py \
  tests/test_residual_corrector_projection.py \
  tests/test_sps_teammate_uncertainty.py
```

若 test 需要改变算法语义才能通过，STOP。

## Run

建议：

```bash
RUN_ROOT=/home/chyfuture/realpde_runs/sota_v3_fast_joint_20260918

PYTHONPATH=tools python tools/realpde_sota_v3_fast_joint.py \
  --manifest "$MANIFEST" \
  --data-root "$DATA_ROOT" \
  --kit-root "$KIT_ROOT" \
  --backbone-checkpoint "$BACKBONE_32500" \
  --corrector-checkpoint "$CORRECTOR_30000" \
  --out-dir "$RUN_ROOT/joint" \
  --micro-batch 2 \
  --accumulation-steps 4 \
  --workers 2 \
  --require-cuda \
  --resume
```

如果仅因 CUDA OOM，允许唯一 fallback：

- micro-batch `1`
- accumulation `8`
- effective batch 仍为 `8`

不得改其他科学参数。

若 `summary.json.gate.status == FAST_JOINT_GO_FULL`：

```bash
PYTHONPATH=tools python tools/realpde_sota_v3_sps.py \
  --mode dev \
  --manifest "$MANIFEST" \
  --data-root "$DATA_ROOT" \
  --kit-root "$KIT_ROOT" \
  --backbone-checkpoint "$(python -c 'import json; print(json.load(open("'"$RUN_ROOT"'/joint/summary.json"))["selected_backbone"])')" \
  --corrector-checkpoint "$(python -c 'import json; print(json.load(open("'"$RUN_ROOT"'/joint/summary.json"))["selected_corrector"])')" \
  --out-dir "$RUN_ROOT/dev_sps" \
  --workers 2 \
  --require-cuda
```

若 shell quoting 不方便，可以用等价的两行 Python/JSON 读取路径；不得修改训练语义。

## Deliverables

新建：

`docs/sota迭代/reviews/sota_v3_fast_joint_20260918/`

至少提交：

- `README.md`
- joint `summary.json`
- joint `aggregate_metrics.csv`
- selected milestone 的 `metrics.json`
- selected milestone 的 `horizon_error_summary.json`
- selected milestone 的 `trajectory_metrics.csv`
- 若 point Gate=GO：Dev SPS `summary.json`、`head_training_summary.json`、`checkpoint_evals.json`、selected calibration grid
- tests / runtime / GPU peak / bounded fix 说明
- backbone/corrector 输入 SHA 与 selected 输出 SHA

README 必须明确：

- 本实验只访问 frozen Train50/Dev16；
- `full-data NOT accessed`；
- `locked-final/private NOT accessed`；
- `Codabench NOT accessed`；
- 本任务不会自动启动 full training。

大 checkpoint/raw log 留远程，不进 Git。

完成 evidence 后 commit + pull/rebase + push `main`，最终只返回：

`REVIEW_REQUIRED`

等待用户与 ChatGPT/Sol 审阅，再决定是否做 full train。
