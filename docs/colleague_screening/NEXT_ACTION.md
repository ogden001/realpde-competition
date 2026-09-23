# NEXT_ACTION — RESIDUAL TRAIN65→DEV16 DISJOINT SAMPLING SCREEN

Status: `READY_FOR_EXECUTION / REVIEW_REQUIRED_AFTER_RUN`

## 1. Scientific question

验证：

> 在 Residual 层面，把 frozen all81 CNO 固定住后，只用 Train65 训练 fresh residual，并在 residual 从未训练过的 Dev16 trajectories 上评估时，trajectory-stratified random-start sampling 是否比 fixed stride20 sampling 更有泛化价值？

本实验只研究：

**Residual sampling policy（残差模型训练采样策略）**

不是：

- 数据增强；
- AoA；
- CFD；
- current80 continuation；
- dense flattened stride1；
- 模型结构实验；
- loss 实验；
- 长训。

---

## 2. Scope / interpretation boundary

必须明确：

- Stage1 CNO checkpoint 历史上已经用 all81 训练；
- 因此 Dev16 对 frozen CNO **不是端到端未见 trajectory**；
- 本实验只保证：
  - Residual Arm A 训练只看 Train65；
  - Residual Arm B 训练只看同一个 Train65；
  - Dev16 不进入任一 residual 的训练 sampler。

所以这是：

> **Residual-stage differential generalization test**

不是：

> end-to-end clean holdout generalization。

任何结果描述都不得越过这个边界。

---

## 3. Frozen source split

Source manifest 必须是 frozen colleague all81 / Dev16 manifest。

SHA256：

`d127e851f3f5eefb011313b1b0b1d79e65db6d99c0754d264119ff12046a83a2`

Data manifest SHA256：

`3612185c939f6c4cd4890afd3968d160a44e88d3e6f7733b8cb9ce71501d532c`

Runner 自动派生：

```text
Train65 = source all81 - frozen Dev16
Dev16   = frozen Dev16
```

必须验证：

- Train65 = 65 trajectories
- Dev16 = 16 trajectories
- Train65 ∩ Dev16 = ∅
- residual `allow_train_dev_overlap = false`
- 不使用 `--train-on-all`

派生 manifest：

`residual_train65_dev16_manifest.json`

必须归档进 Git evidence。

---

## 4. Frozen backbone

Stage1 all81 CNO：

SHA256：

`ff28aaf0114d57e320e5ea9cf9945d874a7482f22e3f0aeaed5ebbe286573f8a`

两个 arm 使用完全相同 frozen checkpoint。

禁止重新训练 CNO。

---

## 5. Fresh residual initialization

两个 arm 都：

- `ResidualCorrector3D`
- hidden = 96
- blocks = 2
- dropout = 0
- include_pressure = true
- max_delta = 0.04
- final Conv3d zero-init
- seed = 41
- no resume checkpoint

必须生成并比较两个：

`model_init.pth`

要求：

- state_dict keys 完全相同
- every tensor bitwise equal
- max_abs_difference = 0

否则：

`BLOCKED`

---

## 6. Exact Train65 window count

Frozen source all81 fixed stride20 一共：

`3341 windows`

Frozen Dev16 fixed stride20：

`640 windows`

因此 residual Train65：

```text
3341 - 640 = 2701 fixed-stride20 windows
```

Control 必须验证：

`reference_fixed_stride_windows == 2701`

batch=8，drop-last：

```text
2701 windows
→ 337 full batches
→ 2696 consumed samples / epoch
```

本 screen 使用：

```text
5055 updates
= 337 batches × 15 complete epochs
```

目的：

> 保证 A/B 都在完整 matched epoch 边界结束，不引入 partial-epoch trajectory exposure 偏差。

---

## 7. Arm A — Train65 fixed stride20

名称：

`A_train65_fixed_stride20`

Sampling：

```text
Train65 only
      ↓
fixed starts = 0,20,40,...
      ↓
2701 windows
      ↓
global shuffle
      ↓
batch=8 / drop-last
```

每 epoch：

- 337 batches
- 2696 consumed samples

---

## 8. Arm B — Train65 trajectory-stratified random-start

名称：

`B_train65_stratified_random_start`

Dataset membership：

**与 Arm A 完全相同 Train65。**

每个 epoch：

1. 先重建 Arm A 当 epoch 在 drop-last 后的实际 per-trajectory draw quota；
2. Arm B 对每条 trajectory 使用完全相同的 draw count；
3. batch 内 8 条 trajectory 必须互不重复；
4. 每条 trajectory 从该 trajectory 所有合法 Past20→Future20 start 中取样；
5. 使用 deterministic shuffled-start bag；
6. 合法 starts 没耗尽前不得重复。

因此 B 只改变：

> **temporal-start coverage + batch trajectory composition**

不得改变：

- trajectory membership
- trajectory weighting
- total sample count

---

## 9. Hard matched contract

两个 arm 完全相同：

- Train65 membership
- Dev16
- frozen CNO
- fresh residual architecture
- initialization seed
- updates = `5055`
- batch = 8
- test batch = 32
- lr = `2e-4`
- weight decay = `1e-5`
- AdamW
- cosine scheduler T_max=5055
- fp32
- grad clip = 1.0
- train alpha = 1.0
- eval stride = 20
- same alpha scan
- same fixed-time proxy
- no phase-count equalization

Loss 固定为 colleague Stage2：

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
- 改 seed
- 改 loss
- 改 TKE weight
- 改 residual architecture
- AMP / bf16
- augmentation
- AoA transform
- CFD
- EMA
- checkpoint averaging
- `--train-on-all`
- `--allow-train-dev-overlap`

本轮唯一科研变量：

> **Residual training sampling policy**

---

## 10. Required code

Main runner：

`tools/colleague_80pt/run_disjoint_trajectory_sampling_screen.py`

Sampler：

`TrajectoryStratifiedRandomStartBatchSampler`

Archive：

`tools/colleague_80pt/archive_disjoint_trajectory_sampling_screen.py`

Tests：

`tests/test_disjoint_trajectory_sampling_screen.py`

---

## 11. Eval milestones

两个 arm 都必须：

- step 0
- 1000
- 2000
- 3000
- 4000
- 5000
- 5055 final

全部使用完整 Dev16：

- 16 trajectories
- 640 fixed stride20 windows

Primary comparison：

```text
candidate_final@5055
vs
control_final@5055
```

Best checkpoint only secondary evidence。

---

## 12. Required sampling audit

每个 arm：

`sampling_audit.json`

至少包含：

- samples_consumed
- unique_windows_consumed
- candidate_legal_windows
- unique_window_fraction
- duplicate_trajectory_batches
- per trajectory:
  - draws
  - unique_starts
  - legal_starts
  - repeated_draws

Campaign：

`sampling_comparison.json`

必须强制：

- A/B total consumed samples exact equal
- A/B per-trajectory draws exact equal
- B duplicate-trajectory batches = 0
- B unique-window count > A

否则：

`BLOCKED`

---

## 13. Required diagnostics

对：

- init
- control_best
- control_final
- candidate_best
- candidate_final

输出：

- Rel-L2
- TKE
- MVPE
- Future1..20
- by trajectory
- trajectory × horizon
- mean-field
- fluctuation
- TKE energy ratio
- correction help/hurt fraction
- delta RMS
- spatial maps

同时生成：

`matched_progress.csv`

记录 0 / 1k / 2k / 3k / 4k / 5k / 5055 的 A/B matched curve。

---

## 14. Scientific review logic

Codex 不做 GO/NO-GO。

Sol review 时重点看：

1. candidate 在 Residual-unseen Dev16 上是否至少 2/3 primary metrics 改善；
2. 是否存在 TKE 与 Rel/MVPE 的系统 tradeoff；
3. trajectory-level win/loss 是否广泛；
4. horizon-level 是否广泛；
5. candidate unique-start exposure 实际提高多少；
6. TKE energy ratio 是否改善或恶化；
7. candidate 学习曲线到 5055 是继续拉开、平台还是反转。

只有出现明确正信号，才讨论后续更长 matched training。

本 runner 禁止自动长训。

---

## 15. Environment autonomy

当前为新配置 GPU 环境。

**科研语义是硬约束，运行环境是软约束。**

Codex 可自主处理：

- CUDA / driver / PyTorch compatibility
- Python / venv
- missing packages
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
- cache/temp path
- Git tracking / detached checkout
- file/path discovery
- permissions
- launcher / nohup / tmux / PID / logs
- output-root v2/v3
- pure compatibility patch
- infrastructure failure retry

不需要人工确认。

但不得改变第 9 节任何科研语义。

若环境修复必须改变科研语义：

`BLOCKED`

---

## 16. Preflight

先：

```bash
git pull --rebase origin main
git status --short
```

记录实际 execution HEAD。

至少执行：

```bash
python -m py_compile \
  tools/colleague_80pt/realpde_h5_feature_adapter_train.py \
  tools/colleague_80pt/residual_multi.py \
  tools/colleague_80pt/run_disjoint_trajectory_sampling_screen.py \
  tools/colleague_80pt/archive_disjoint_trajectory_sampling_screen.py

python -m pytest -q \
  tests/test_disjoint_trajectory_sampling_screen.py \
  tests/test_trajectory_sampling_screen.py \
  tests/test_colleague_incremental_screen.py \
  tests/test_post_train_diagnostics.py
```

工程/environment bug：

- 最小修复
- 补测试
- commit
- 继续

科研语义问题：

- STOP / BLOCKED

---

## 17. Execute

参考：

```bash
python -u -B tools/colleague_80pt/run_disjoint_trajectory_sampling_screen.py \
  --real-root /hy-tmp/realpde_data/train_real \
  --base-checkpoint /hy-tmp/realpde_assets/colleague-80pt/20260921/extracted/handoff_colleague_80pt_archive_20260921/checkpoints/cno_final_all81.pt \
  --model-root tools/colleague_80pt/submission \
  --data-manifest /hy-tmp/realpde_data/data_manifest.tsv \
  --source-split-manifest /hy-tmp/realpde_runs/colleague_incremental_screen_20260921/colleague_dev16_manifest.json \
  --out-root /hy-tmp/realpde_runs/disjoint_trajectory_sampling_screen_20260923_v1 \
  --workers 4
```

workers 可根据环境调整。

若 output root 已存在：

- 使用 v2/v3
- 不覆盖旧结果

---

## 18. Archive

完成且 audit 全 PASS：

```bash
python -u -B tools/colleague_80pt/archive_disjoint_trajectory_sampling_screen.py \
  --run-root <ACTUAL_RUN_ROOT> \
  --dest docs/colleague_screening/results/20260923_disjoint_trajectory_sampling_screen
```

如果 destination 已存在：

- 使用 v2/v3

然后：

```bash
git diff --check
git add docs/colleague_screening/results/20260923_disjoint_trajectory_sampling_screen*
git commit -m "Archive residual-disjoint trajectory sampling screen"
git pull --rebase origin main
git push origin main
```

禁止提交：

- checkpoints
- H5
- raw full logs
- prediction caches

---

## 19. Stop conditions

完成 archive + push 后停止。

禁止自动：

- 20k / 38.4k
- sampler sweep
- LR sweep
- batch sweep
- loss tweak
- EMA
- ensemble
- full-data refit
- package
- Codabench
- locked-final/private access

---

## 20. Final handoff

返回：

```text
REALPDE RESIDUAL-DISJOINT TRAJECTORY SAMPLING SCREEN

Status:
REVIEW_REQUIRED / BLOCKED

Execution commit:
...

Results commit:
...

Base CNO SHA:
PASS / FAIL

Source split/data manifest:
PASS / FAIL

Derived Train65 / Dev16:
PASS / FAIL

Residual Train65∩Dev16:
EMPTY / FAIL

Train65 fixed stride20 windows:
2701 / ...

Samples per epoch:
2696 / ...

A/B initialization parity:
PASS / FAIL

Arm A Train65 fixed stride20:
COMPLETE / BLOCKED

Arm B Train65 stratified random-start:
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

Codex 不做最终科研结论。Sol 在 Git Evidence Acceptance 后 review。
