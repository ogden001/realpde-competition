# NEXT_ACTION — 80PT BASELINE: ADJACENT-AOA MEAN-FIELD LONG SCREEN

Status: `READY_FOR_EXECUTION / REVIEW_REQUIRED_AFTER_RUN`

## Goal

从冻结的同事线上 80.0788 分 residual baseline 出发，验证一种更物理保守的训练期攻角增强：

**相邻攻角均值场插值（Adjacent-AoA Mean-Field Interpolation）**

中文含义：

> 不把两个不同攻角、不同涡脱落相位的瞬时流场直接做 Mixup。  
> 对 anchor trajectory，只在相同 Reynolds 数下寻找真实的相邻攻角 trajectory；  
> 只用两者 Past20 的时间平均空间速度场差异，估计攻角改变造成的均值流场变化；  
> 再把这一空间均值场变化一致地加到 anchor 的 Past20 和 Future20。  
> anchor 自己的动态波动、涡结构和时间相位保持不被另一条 trajectory 混掉。

本任务只做一次 20k-update 长训，不做参数网格。

---

## Frozen baseline

线上事实基线：

- online final: `80.078849`
- residual checkpoint SHA256:
  `909fdc7f8a6a42335e4ea4ce7471fc9a507135e9a29a4927bc7b7318be9c85b2`

统一离线 baseline：

- Rel-L2: `0.0804204196`
- TKE: `0.4491582513`
- MVPE: `0.0711169168`

历史 matched 5k no-augmentation control：

- Rel-L2: `0.0802410692`
- TKE: `0.4503658414`
- MVPE: `0.0710276663`

5k 时必须与这组 historical no-augmentation control 做同预算比较。

---

## Experiment semantics

代码：

- `tools/colleague_80pt/aoa_meanfield_augmentation.py`
- `tools/colleague_80pt/residual_multi.py`
- `tools/colleague_80pt/run_aoa_meanfield_long_screen.py`

固定语义：

1. 只使用训练期 HDF5 metadata 的 `re / aoa` 寻找邻居；
2. model input 中不加入 Re 或 AoA；
3. inference 不需要 Re / AoA；
4. neighbor 必须相同 Re；
5. 只允许最近相邻 AoA，最大 gap `5.1°`；
6. 至少 80% train trajectories 必须有合法相邻 AoA neighbor，否则实验无效并停止；
7. augmentation probability = `0.5`；
8. lambda uniform in `[0.2, 0.5]`；
9. 只用 Past20 mean field 计算空间 shift；
10. Future20 不参与构造 input shift；
11. 同一 spatial shift 同时应用到 anchor Past20 和 Future20；
12. 不做旧的 u/v global angle rotation；
13. Dev 不做 augmentation。

数学定义：

```text
delta_mean_uv = mean_t(Past20_neighbor) - mean_t(Past20_anchor)

Past20_aug   = Past20_anchor   + lambda * delta_mean_uv
Future20_aug = Future20_anchor + lambda * delta_mean_uv
```

这不是声称合成样本是真实 CFD 解，而是一个保守的“攻角相关均值场插值”训练增强。

---

## Training budget

固定：

- start checkpoint = frozen current 80pt residual；
- updates = `20,000`；
- eval every `2,500` updates；
- batch size = `8`；
- lr = `2e-4`；
- weight decay = `1e-5`；
- hidden = `96`；
- blocks = `2`；
- max_delta = `0.04`；
- TKE loss weight = `0.06`；
- fixed temporal windows；
- seed = `41`；
- optimizer / cosine schedule 与 previous residual screens 一致。

不得缩短为 5k，也不得自动超过 20k。

---

## Why 20k without rerunning a 20k control

本任务不额外消耗 GPU 重跑 long no-augmentation control。

原因：

- 已有同起点、同 optimizer、同 loss、同 seed 的 historical no-augmentation 5k control；
- candidate 在 5k evaluation point 做严格同预算 comparison；
- 然后继续到 20k，用于判断 data augmentation 是否只是收敛更慢；
- 20k 最终 checkpoint 同时与 frozen current80 baseline 比较；
- 若只有极小增益，不因为“训练更久”自动判 GO。

---

## Environment autonomy

科研语义是硬约束，运行环境是软约束。

Codex 可自行修复并继续：

- Git remote / tracking ref / detached checkout；
- venv / Python executable / CUDA device；
- 文件路径、launcher、PID/log；
- output root 版本号；
- whitelist archive；
- HDF5 I/O 兼容；
- DataLoader `num_workers`，建议 2–8，根据机器吞吐自行选；
- 其他不影响实验数值语义的工程问题。

Codex 不得自行修改：

- baseline / checkpoint SHA；
- train/dev split；
- 20k budget；
- batch size / lr / seed；
- augmentation probability；
- lambda range；
- same-Re / adjacent-AoA pairing；
- mean-field augmentation formula；
- model / feature / loss / loss weight；
- Gate / scorer；
- locked-final/private/Codabench 边界。

---

## Preflight

同步最新 main，并记录 actual HEAD：

```bash
git fetch origin
git pull --rebase origin main
git status --short
git rev-parse HEAD
git push --dry-run origin HEAD:main
```

运行：

```bash
/hy-tmp/realpde_venv_v2/bin/python -m pytest -q \
  tests/test_colleague_next_two_experiments.py \
  tests/test_colleague_incremental_screen.py \
  tests/test_post_train_diagnostics.py
```

若测试失败：

- 可以自行修复纯工程/兼容性错误并补测试；
- 若修复会改变上述科研语义，则停止。

---

## Execute

默认：

```bash
/hy-tmp/realpde_venv_v2/bin/python -u -B \
  tools/colleague_80pt/run_aoa_meanfield_long_screen.py \
  --real-root /hy-tmp/realpde_data/train_real \
  --base-checkpoint /hy-tmp/realpde_assets/colleague-80pt/20260921/extracted/handoff_colleague_80pt_archive_20260921/checkpoints/cno_final_all81.pt \
  --start-checkpoint /hy-tmp/realpde_assets/colleague-80pt/20260921/extracted/handoff_colleague_80pt_archive_20260921/checkpoints/residual_model_best.pth \
  --model-root tools/colleague_80pt/submission \
  --data-manifest /hy-tmp/realpde_data/data_manifest.tsv \
  --split-manifest /hy-tmp/realpde_runs/colleague_incremental_screen_20260921/colleague_dev16_manifest.json \
  --out-root /hy-tmp/realpde_runs/aoa_meanfield_long_20260922_v1 \
  --workers 4
```

若默认 out-root 已存在，使用新的 `v2/v3...`，不得删除旧 run。

---

## Required evidence

必须有：

### Provenance / audit

- campaign manifest；
- exact execution commit；
- checkpoint/data/split SHA；
- `aoa_augmentation_audit.json`；
- eligible trajectory fraction；
- 每条 trajectory 的 Re / AoA / selected-neighbor 列表；
- run_config；
- commands。

### Training

- step 0 / 2500 / 5000 / 7500 / 10000 / 12500 / 15000 / 17500 / 20000；
- Rel-L2 / TKE / MVPE；
- learning rate；
- augmentation applied fraction；
- effective AoA shift magnitude；
- deterministic training review log。

### Matched 5k comparison

必须生成：

`comparison_at_5k.csv`

同时列：

- frozen current80；
- historical no-aug 5k；
- AoA mean-field aug 5k。

### Post-train diagnostics

对以下三个 checkpoint 统一 replay：

1. `current80`
2. `aoa_best`
3. `aoa_final`

每个必须包含：

- overall；
- Future1..20；
- by-trajectory；
- trajectory × horizon；
- mean / fluctuation / energy；
- spatial maps；
- residual before/after。

最终 comparison 目录必须有：

- `comparison_summary.csv`
- `comparison_by_horizon.csv`
- `manifest.json`

---

## Archive

完成后：

```bash
/hy-tmp/realpde_venv_v2/bin/python -u -B \
  tools/colleague_80pt/archive_aoa_meanfield_long_screen.py \
  --run-root <actual_run_root> \
  --dest docs/colleague_screening/results/20260922_aoa_meanfield_long
```

若 Git destination 已存在，使用带序号的新目录，不覆盖。

然后：

```bash
git diff --check
git status --short
git add docs/colleague_screening/results/20260922_aoa_meanfield_long*
git commit -m "Archive adjacent AoA mean-field long screen"
git pull --rebase origin main
git push origin main
```

禁止提交：

- model checkpoints；
- H5；
- raw prediction cache；
- raw full training log。

---

## Stop

完成 Git push 后停止。

不要：

- 自动 full train；
- 自动提交 Codabench；
- 自动做 submission package；
- 访问 locked-final/private；
- 自动调 augmentation 参数；
- 自动启动下一轮实验。

最终只返回：

```text
REALPDE AOA MEAN-FIELD LONG SCREEN

Status:
REVIEW_REQUIRED / BLOCKED

Execution commit:
...

Results commit:
...

Git evidence:
...

AoA pairing audit:
PASS / FAIL
Eligible trajectories:
...

Training 20k:
COMPLETE / BLOCKED

5k matched comparison:
PASS / FAIL

Standard diagnostics:
PASS / FAIL

Training review log:
PASS / FAIL

Best checkpoint iteration:
...

New full training started:
NO

Codabench accessed:
NO

Locked-final accessed:
NO

Missing items:
NONE / ...
```

如果只是环境工程问题，先自行修复并继续。只有会改变科研语义时才停止。
