# NEXT_ACTION — MATCHED RESIDUAL TRAJECTORY-SAMPLING SCREEN

Status: `READY_FOR_EXECUTION / REVIEW_REQUIRED_AFTER_RUN`

## 1. Scientific question

验证：

> 在完全相同的 released PIV trajectory、frozen CNO、fresh residual 初始化、模型、loss、optimizer 和训练 sample budget 下，只改变 residual 的训练采样策略，能否通过更充分地利用每条真实 trajectory 的 temporal starts 提高学习效果？

本实验不是：

- 数据增强；
- AoA 实验；
- CFD 实验；
- current80 continuation；
- stride=1 flattened dataset 重跑；
- 长训。

本实验只比较 **sampling policy（采样策略）**。

---

## 2. Why this experiment exists

已知事实：

1. 原始 80pt Stage2 residual：
   - all81 released PIV
   - fixed stride=20
   - **3341 windows**
   - h96/b2
   - 38400 updates
   - best@34000

2. `dense stride=1 from scratch` 在同事历史记录中曾失败，但当时没有严格的 trajectory-stratified sampler 对照。

3. 2026-09-23 的 current80 dense continuation 也失败，但它回答的是：
   - 一个已经被 stride20 长训到成熟状态的 residual，
   - 突然切换 sampling distribution 后短程 continuation 是否有效。
   
   它**不能回答** fresh residual 从训练开始就使用更合理 random-start sampler 是否更好。

4. 因此这次必须从相同的 fresh residual 开始做 matched A/B，避免 continuation history bias。

---

## 3. Frozen data / model baseline

### Released PIV data

严格使用 historical colleague protocol：

- train: all 81 usable released PIV trajectories
- dev: frozen Dev16，按历史 colleague protocol 与 all81 overlap
- locked-final/private: 禁止访问

Frozen split manifest SHA256：

`d127e851f3f5eefb011313b1b0b1d79e65db6d99c0754d264119ff12046a83a2`

Frozen data manifest SHA256：

`3612185c939f6c4cd4890afd3968d160a44e88d3e6f7733b8cb9ce71501d532c`

不得重新划分，不得换 trajectory。

### Frozen Stage1 CNO

使用 colleague all81 CNO：

SHA256：

`ff28aaf0114d57e320e5ea9cf9945d874a7482f22e3f0aeaed5ebbe286573f8a`

两个 arm 使用完全相同 checkpoint。

### Residual initialization

两个 arm 都：

- fresh `ResidualCorrector3D`
- hidden=96
- blocks=2
- max_delta=0.04
- output layer zero-init
- seed=41
- **禁止 resume 任何 residual checkpoint**

runner 必须在训练完成后比较两个 `model_init.pth` 的 `model_state_dict`：

- key 完全相同
- tensor 完全 bitwise equal
- max absolute difference = 0

否则：`BLOCKED`。

---

## 4. Exact original Stage2 sampling baseline

原始 handoff 的真实 Stage2 evidence：

`docs/colleague_80pt_handoff/evidence/run_config.json`

明确记录：

`train_windows = 3341`

因此本实验 Control 必须复刻：

- stride=20
- starts = 0,20,40,...
- all 81 trajectories
- **3341 candidate windows**
- batch=8
- DataLoader drop-last

3341 / 8：

- 每完整 epoch = 417 batches
- 实际 consumed samples = 3336
- 5 windows 被 drop-last

为了消除最后 partial epoch 的 trajectory exposure 偏差，本实验固定：

`5004 updates = 417 × 12`

即：

> **恰好 12 个完整 matched epochs**

这仍然是约 5k update screen，只比 5000 多 4 steps。

### Important compatibility rule

当前重构版 runner 曾加入 phase-count equalization。

本任务必须：

`--disable-phase-count-equalization`

并强制验证：

- `reference_fixed_stride_windows == 3341`
- `train_samples_per_epoch == 3336`
- `phase_counts_equalized == false`

任何一项不满足：`BLOCKED`。

---

## 5. Arm A — Baseline Control

名称：

`A_fixed_stride20`

Sampling：

```text
all81 trajectories
        ↓
fixed start = 0,20,40,...
        ↓
3341 windows
        ↓
global shuffle
        ↓
batch=8, drop-last
```

保持原始 residual Stage2 的 fixed-stride sampling semantics。

---

## 6. Arm B — Trajectory-Stratified Random-Start

名称：

`B_stratified_random_start`

数据集 membership 与 A **完全相同**。

每个 epoch 先读取 Arm A 在该 epoch、drop-last 后的实际 trajectory sample quota。

例如某 trajectory 在 A 这一 epoch 实际出现 41 次，那么 B 也必须出现 **恰好 41 次**。

因此 B 不得：

- 均匀重权重 short trajectory；
- 增加或减少任何 trajectory 的 epoch exposure；
- 改变总 sample 数。

B 只改变如何选择 temporal start 与 batch composition：

```text
Arm A 的每-trajectory quota
        ↓
按相同 quota 选择 trajectory
        ↓
一个 batch 内 8 条 trajectory 必须互不重复
        ↓
每条 trajectory 从全部合法 start 中取一个 start
        ↓
per-trajectory shuffled-start bag
        ↓
该 trajectory 的合法 start 未遍历完之前尽量不重复
```

### Hard sampler requirements

必须满足：

1. 每个完整 epoch：
   - A/B total consumed samples 相同；
   - A/B 每条 trajectory draw count 完全相同。

2. B：
   - batch 内 duplicate trajectory 数 = 0；
   - candidate 使用所有合法 Past20→Future20 starts 作为 start pool；
   - start selection deterministic under seed=41；
   - shuffled bag 在合法 starts 耗尽前不重复。

3. 5004 updates 完成后：
   - A/B total consumed samples 完全相同；
   - A/B per-trajectory draw counts 完全相同；
   - B 的 unique windows / unique starts 必须明显高于 A。
   
若 B 没有实际提升 unique-start exposure，说明 sampler 实现错误，返回 `BLOCKED`。

---

## 7. Common training semantics — HARD CONTRACT

两个 arm 完全相同：

- frozen CNO SHA above
- fresh residual
- hidden=96
- blocks=2
- dropout=0
- include pressure feature=true
- max_delta=0.04
- train alpha=1.0
- updates=`5004`
- eval steps:
  - 0
  - 1000
  - 2000
  - 3000
  - 4000
  - 5000
  - 5004
- batch=8
- test batch=32
- lr=`2e-4`
- weight decay=`1e-5`
- AdamW
- cosine scheduler T_max=5004
- seed=41
- fp32
- grad clip=1.0
- eval stride=20
- same Dev16
- same alpha scan
- same fixed-time proxy

Loss 必须保持 colleague Stage2：

```text
point          = 1.0
mse            = 0.05
tke            = 0.06
temporal       = 0.03
grad           = 0.015
p_zero         = 0.01
residual_mse   = 0.25
delta_penalty  = 0.02
```

禁止：

- 改 LR
- 改 batch
- 改 loss
- 改 TKE weight
- 改 residual capacity
- AMP / bf16
- augmentation
- AoA transform
- random rotation
- CFD
- EMA
- checkpoint averaging

本轮**唯一科研变量 = training sampling policy**。

---

## 8. Required implementation

Sampler：

`TrajectoryStratifiedRandomStartBatchSampler`

位于：

`tools/colleague_80pt/realpde_h5_feature_adapter_train.py`

Residual runner 已支持：

`--train-window-mode trajectory_stratified_random_start`

主 campaign runner：

`tools/colleague_80pt/run_trajectory_sampling_screen.py`

Archive：

`tools/colleague_80pt/archive_trajectory_sampling_screen.py`

---

## 9. Required evidence

### A. Provenance

必须归档：

- execution commit
- data manifest + SHA
- split manifest + SHA
- base CNO SHA
- GPU / VRAM
- Python / PyTorch / CUDA
- environment variables explicitly changed
- commands

### B. Initialization parity

`initialization_parity.json`

必须：

`exact = true`

否则 BLOCKED。

### C. Sampling audit

两个 arm 都必须输出：

`sampling_audit.json`

至少包含：

- samples_consumed
- unique_windows_consumed
- candidate_legal_windows
- unique_window_fraction
- duplicate_trajectory_batches
- 每条 trajectory:
  - draws
  - unique_starts
  - legal_starts
  - repeated_draws

Campaign 必须输出：

`sampling_comparison.json`

强制验证：

- A/B samples consumed equal
- A/B per-trajectory draws exact equal
- B batch duplicate trajectory = 0
- B unique window count > A

### D. Training curve

`matched_progress.csv`

必须包含：

- step0
- 1k
- 2k
- 3k
- 4k
- 5k
- 5004 final

每一步 A/B：

- Rel-L2
- TKE
- MVPE
- percentage delta B vs A
- fixed-time final estimate

### E. Standard diagnostics

对：

- init
- control_best
- control_final
- candidate_best
- candidate_final

统一输出：

- aggregate Rel/TKE/MVPE
- Future1..20
- by trajectory
- trajectory × horizon
- mean field
- fluctuation
- TKE energy ratio
- correction help/hurt
- delta RMS
- spatial maps

---

## 10. Scientific review rule

Primary comparison：

```text
candidate_final@5004
vs
control_final@5004
```

Best checkpoint 只作为 secondary evidence。

原因：

> 这轮首先研究同预算 sampler learning behavior，不允许用不同 early-stop iteration 掩盖 sampler 差异。

### What counts as positive signal

不预注册一个机械单数字阈值。

Sol review 时重点看：

1. 至少 2/3 primary metrics 是否改善；
2. 是否存在明显 TKE↔Rel/MVPE tradeoff；
3. trajectory-level 改善是否广泛；
4. horizon-level 改善是否广泛；
5. B 是否真的获得大幅更多 unique-start exposure；
6. 是否出现系统性 energy-ratio / late-horizon 退化；
7. 学习曲线到 5004 时是在拉开、持平还是反转。

只有出现**清晰、机制一致的正信号**，才讨论 20k matched follow-up。

不得由 Codex 自动判 GO，也不得自动长训。

---

## 11. Environment autonomy

这是新配置 GPU 环境。

科研语义是硬约束，运行环境是软约束。

Codex 可以自主处理：

- CUDA / driver / PyTorch compatibility
- Python / venv
- missing dependencies
- `CUDA_VISIBLE_DEVICES`
- `PYTORCH_CUDA_ALLOC_CONF`
- OMP / MKL / OPENBLAS / NUMEXPR thread vars
- `TMPDIR`
- `TORCH_HOME`
- `HF_HOME`
- `XDG_CACHE_HOME`
- `PYTHONPATH`
- `HDF5_USE_FILE_LOCKING`
- DataLoader workers
- cache/temp paths
- Git tracking / detached checkout
- path discovery
- file permissions
- launcher / nohup / tmux / PID / logs
- output-root v2/v3 naming
- pure compatibility fixes
- tests for compatibility fixes
- pure infrastructure failure retry

无需人工确认。

但不得改变：

- dataset membership
- split
- base checkpoint
- 5004 update budget
- batch
- lr / wd
- seed
- model
- loss
- sampler scientific semantics
- fp32
- eval protocol
- locked-final/private/Codabench boundary

如果环境修复必须触碰这些科研语义，返回 BLOCKED。

---

## 12. Preflight tests

拉取最新 main，记录实际 execution HEAD。

至少运行：

```bash
python -m pytest -q \
  tests/test_trajectory_sampling_screen.py \
  tests/test_colleague_incremental_screen.py \
  tests/test_post_train_diagnostics.py
```

另外建议：

```bash
python -m py_compile \
  tools/colleague_80pt/realpde_h5_feature_adapter_train.py \
  tools/colleague_80pt/residual_multi.py \
  tools/colleague_80pt/run_trajectory_sampling_screen.py \
  tools/colleague_80pt/archive_trajectory_sampling_screen.py
```

测试失败：

- 纯环境 / integration issue：Codex 最小修复 + 补测试 + 继续；
- 科研语义问题：BLOCK。

---

## 13. Execute

参考：

```bash
python -u -B tools/colleague_80pt/run_trajectory_sampling_screen.py \
  --real-root /hy-tmp/realpde_data/train_real \
  --base-checkpoint /hy-tmp/realpde_assets/colleague-80pt/20260921/extracted/handoff_colleague_80pt_archive_20260921/checkpoints/cno_final_all81.pt \
  --model-root tools/colleague_80pt/submission \
  --data-manifest /hy-tmp/realpde_data/data_manifest.tsv \
  --split-manifest /hy-tmp/realpde_runs/colleague_incremental_screen_20260921/colleague_dev16_manifest.json \
  --out-root /hy-tmp/realpde_runs/trajectory_sampling_screen_20260923_v1 \
  --workers 4
```

workers 可根据新 GPU 环境自行调整。

如果 output root 已存在：

- 用 v2/v3
- 不覆盖、不删除旧 run

---

## 14. Archive

训练完成且所有 audit PASS：

```bash
python -u -B tools/colleague_80pt/archive_trajectory_sampling_screen.py \
  --run-root <ACTUAL_RUN_ROOT> \
  --dest docs/colleague_screening/results/20260923_trajectory_sampling_screen
```

若 destination 已存在，使用 v2/v3。

然后：

```bash
git diff --check
git add docs/colleague_screening/results/20260923_trajectory_sampling_screen*
git commit -m "Archive matched trajectory sampling screen"
git pull --rebase origin main
git push origin main
```

禁止提交：

- checkpoint
- H5
- raw full training log
- prediction cache

---

## 15. Stop

完成归档并 push 后停止。

禁止自动：

- 20k / 38.4k continuation
- sampler sweep
- LR sweep
- batch sweep
- loss tweak
- EMA
- ensemble
- full refit
- package
- Codabench
- locked-final/private access

返回：

```text
REALPDE MATCHED TRAJECTORY SAMPLING SCREEN

Status:
REVIEW_REQUIRED / BLOCKED

Execution commit:
...

Results commit:
...

Base CNO SHA:
PASS / FAIL

Split/data manifest:
PASS / FAIL

Original 3341-window baseline:
PASS / FAIL

A/B initialization parity:
PASS / FAIL

Arm A fixed stride20:
COMPLETE / BLOCKED

Arm B stratified random-start:
COMPLETE / BLOCKED

A/B samples consumed:
MATCHED / FAIL

A/B per-trajectory draws:
MATCHED / FAIL

Candidate duplicate-trajectory batches:
0 / ...

Candidate unique-start gain:
...

Training review logs:
PASS / FAIL

Matched progress:
PASS / FAIL

Standard diagnostics:
PASS / FAIL

Control best iteration:
...

Candidate best iteration:
...

Long follow-up started:
NO

Codabench accessed:
NO

Locked-final accessed:
NO

Missing items:
NONE / ...
```

Codex 不做最终科研判断。Sol 在 Git Evidence Acceptance 后 review。
