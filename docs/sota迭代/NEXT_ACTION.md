# NEXT_ACTION — SOTA-V3 MERGE LONG RUN

Status: `READY_FOR_EXECUTION / REVIEW_REQUIRED`

`REQUIRED_COMMIT = 95d76862944d23ae6d73d9efb49deb0e745f4fcb`

## Goal

执行已经由 ChatGPT/Sol 实现并冻结的 SOTA-V3 一体化长程 pipeline：

`SOTA-V2 + AoA augmentation -> residual corrector + spatial-TKE projection -> teammate35 SPS -> all-82 refit -> package -> smoke`

本任务**不做消融、不做临时扫参、不重新设计算法**。Codex 仅负责 preflight、测试、GPU 执行、日志与 evidence 回收、commit + push。

## Frozen scientific semantics

### 1. Backbone / AoA

- 基础 recipe 保持 SOTA-V2：Dense-All + P0-A + MF-CNO + N2 + vorticity + Stage-B extra Rel。
- Dev 协议：固定 50 Train / 16 Dev manifest。
- AoA augmentation 仅训练启用：
  - 每个 sample 概率 `0.5`；
  - `delta_alpha ~ Uniform[-2deg,+2deg]`；
  - Past20 与 Future20 使用同一个 `delta_alpha`；
  - 只旋转 `u/v` 向量，pressure 不变；
  - Dev / inference 不做 augmentation。
- 这是本轮预注册的保守 augmentation；不是声称复刻了同事未提供的具体 AoA 实现。
- Dev 只在既有 frozen milestones 评估；只允许 `update >= 30000` 参与 checkpoint selection。
- selection objective 固定为：
  `Rel/base_Rel + TKE/base_TKE + MVPE/base_MVPE`，越低越好；同分取更早 update。
- Full 82 refit 的训练终点由 Dev-selected reference update 按 Dense-All epoch exposure 映射，禁止根据 full-data loss 重新选模。

### 2. Residual corrector

- Backbone 完全冻结。
- `ResidualCorrector3D`：`42ch / h64 / blocks2 / max_delta=0.04`。
- 复用已验证 teammate residual objective 与固定 loss weights。
- Dev corrector：`30000` updates，batch `8`，AdamW `lr=1e-4 / wd=1e-5`，CosineAnnealingLR。
- Full corrector：按 Dense-All exposure 将 30k 映射到 `49461` updates；不做 full-data checkpoint selection。
- `alpha=1.0` 固定。
- inference projection 固定为 `spatial_tke_map`；禁止重新扫 alpha / window_energy / projection。

### 3. SPS / uncertainty

- Head feature/input：同事 35-channel exact recipe，`h32/b2/drop0/include_pressure`。
- Head **看 base backbone prediction**。
- Final point prediction：`base -> residual corrector -> spatial_tke_map projection`。
- Interval **以 final point prediction 为中心**。
- Head training：masked Gaussian NLL，误差中心使用 final corrected prediction。这是本轮冻结的 reconstruction choice；同事 package 没有包含训练脚本，不得表述为 package-confirmed。
- Dev：seed `41`，AdamW `lr=1e-3 / wd=1e-5`，最多 `2000` updates，每 `200` eval；固定 28-row floor×mult grid；按真实 Dev SPS 最大选 checkpoint，同 SPS 取更窄 interval。
- Full：fresh head 在 all-82 canonical windows 上训练，updates 使用 Dev-selected iteration；floor/mult 原样复用 Dev-selected 值；禁止 full-data recalibration。

### 4. Package

- 白名单 staging。
- package inference 必须严格是：
  `base -> residual -> spatial_tke_map -> final`；
  `sigma = head(Past20, base)`；
  `bounds = final +/- (floor + mult*sigma)`。
- pressure prediction 与 pressure interval width 均为 0。
- ZIP `<256 MiB`。
- 必须对 released Train/Dev fixture 做 clean-room smoke 与 direct-stack prediction parity。

## Preflight

开始前必须：

```bash
git status --short
git fetch origin
git pull --rebase origin main
git rev-parse HEAD
git rev-parse origin/main
git merge-base --is-ancestor "$REQUIRED_COMMIT" HEAD
git push --dry-run origin HEAD:main
```

条件：

- 工作区无未知未提交改动；
- `HEAD == origin/main`；
- required commit 是 HEAD 祖先；
- dry-run push 成功；
- 不满足则 STOP，不启动 GPU。

## Tests

至少：

```bash
PYTHONPATH=tools pytest -q \
  tests/test_sota_v3_pipeline.py \
  tests/test_sota_v2_integrated.py \
  tests/test_sota_v2_full.py \
  tests/test_teammate_residual_transfer.py \
  tests/test_residual_corrector_projection.py \
  tests/test_sps_teammate_uncertainty.py \
  tests/test_sps_teammate_exact_submit.py
```

若成本正常，再跑全仓 `pytest -q`。若出现与本任务无关的已知平台问题，只记录，不改变实验语义。

## Run

资产使用现有已验证路径，启动前逐项核对：

- frozen 50/16 manifest；
- released PIV data root，必须正好 82 trajectories；
- official starting kit v9；
- official `sim_real_ft` warm checkpoint，SHA256 必须匹配代码冻结值；
- smoke fixture 必须来自 released Train/Dev，禁止 locked-final/private。

推荐单命令：

```bash
RUN_ROOT=/home/chyfuture/realpde_runs/sota_v3_merge_20260918
EXECUTION_COMMIT=$(git rev-parse HEAD)

PYTHONPATH=tools python tools/run_sota_v3_pipeline.py \
  --manifest "$MANIFEST" \
  --data-root "$DATA_ROOT" \
  --kit-root "$KIT_ROOT" \
  --warm-checkpoint "$WARM_CKPT" \
  --fixture "$FIXTURE" \
  --run-root "$RUN_ROOT" \
  --execution-commit "$EXECUTION_COMMIT" \
  --backbone-micro-batch 4 \
  --backbone-accumulation 2 \
  --workers 2
```

以 detached 方式启动。总控 runner 与 backbone/residual stage 支持恢复；若会话中断，重新执行同一命令，不另造 launcher。

## Codex permissions

**Execution-only. Do not implement or redesign experimental logic.**

允许 bounded runtime fix：

- host/container 路径映射；
- shell quoting；
- PYTHONPATH；
- 文件权限；
- CUDA / Torch / NumPy ABI；
- worker 数量在不改变数据顺序/科学语义前提下的运行适配；
- 已冻结的 micro-batch/accumulation 组合保持 effective batch=8。

任何需要修改以下内容的情况必须 STOP 并返回 `REVIEW_REQUIRED`：

- AoA max angle / probability / rotation semantics；
- split / data scope；
- backbone loss / Stage-B / selection objective；
- residual architecture / loss / 30k budget / alpha / projection；
- SPS features / architecture / seed / optimizer / loss / 2000 budget / eval cadence / grid；
- base-vs-final SPS wiring；
- full-data mapping 与 calibration 规则；
- package inference semantics。

禁止访问 locked-final/private Future20，禁止自动 Codabench 提交，禁止新增 ablation。

## Evidence / Deliverables

执行完成后新建：

`docs/sota迭代/reviews/sota_v3_merge_20260918/`

至少提交：

- `README.md`
- `summary.json`（复制 pipeline summary 的轻量版）
- Dev backbone `selection.json` / `aggregate_metrics.csv`
- Dev residual `evaluation_summary.json` / `physical_metrics.csv`
- Dev SPS `summary.json` / `head_training_summary.json` / `checkpoint_evals.json` / selected grid
- full backbone / residual / SPS 的 summary 与关键 runtime metadata
- `package_build.json`
- `smoke_report.json`

README 必须记录 execution commit、各 checkpoint/head/ZIP SHA256、Dev 三物理指标、Dev SPS 与 calibration、full mapped updates、运行时间、测试、bounded fixes、远程 artifact 路径，并明确：

`locked-final/private/Codabench NOT accessed`。

大 checkpoint、raw log、submission ZIP 留远程，不写 Git。

最后 commit + pull/rebase + push `main`，确认远端包含 evidence commit，再返回：

`REVIEW_REQUIRED`

并报告 submission ZIP 绝对路径、bytes、SHA256。等待用户手工决定是否提交。
