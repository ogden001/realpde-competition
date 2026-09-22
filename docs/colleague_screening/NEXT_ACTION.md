# NEXT_ACTION — 80PT BASELINE: TWO DIRECTION SCREEN

Status: `READY_FOR_EXECUTION / REVIEW_REQUIRED_AFTER_RUN`

## Goal

从冻结的同事线上 80.0788 分方案出发，一次性执行两个独立增量实验：

1. **波动幅度校准**：保持 Future20 时间平均流场不变，只增强动态波动幅度；
2. **攻角近似数据增强**：训练时对 Past20/Future20 的 u/v 速度方向施加同一个小角度旋转，测试是否提升 80 分 residual 的泛化能力。

两个实验都必须使用同一冻结 Dev16 和标准诊断协议。Codex 只负责执行、产出结构化证据和提交 Git，不做最终科研判断。

---

## Frozen baseline

线上事实基线：

- online final: `80.078849`
- residual checkpoint SHA256:
  `909fdc7f8a6a42335e4ea4ce7471fc9a507135e9a29a4927bc7b7318be9c85b2`

统一离线 decision baseline：

- Rel-L2: `0.0804204196`
- TKE: `0.4491582513`
- MVPE: `0.0711169168`

不得更换 checkpoint，不寻找 h96x16，不吸收同事新的 checkpoint。

---

## Experiment A — 波动幅度校准

中文含义：

> 保留模型预测的时间平均流场和主要时空结构，只对“随时间变化的动态波动部分”做幅度恢复。

实现：

`tools/colleague_80pt/fluctuation_calibration.py`

固定 coarse scan：

- Future1 scale = 1.0
- Future20 end scale = `1.0 / 1.2 / 1.4 / 1.6`
- 中间 horizon 线性增长；
- 校正后重新去均值，保证 Future20 temporal mean 与原预测一致；
- 不训练模型。

不允许 Codex 增加 scale 网格。

---

## Experiment B — 攻角近似数据增强

中文含义：

> 不使用 AoA 标签，不改变正式推理输入。训练时对每个 window 随机采样一个 `[-2°, +2°]` 小角度，把 Past20 和 Future20 的 u/v 速度向量同时旋转相同角度。

注意：

- 空间网格不旋转；
- p 保持不变；
- 这是小角度“流向扰动/攻角近似增强”，不是声称生成了真实新 AoA CFD 解；
- Dev 不做任何 augmentation；
- 只改变训练数据增强，其他训练配置完全匹配昨晚的无增强 R0 control。

固定训练：

- start checkpoint = 当前 80 分 residual；
- 5000 updates；
- TKE loss weight = 0.06；
- fixed temporal windows；
- batch 8；
- lr 2e-4；
- hidden 96；
- blocks 2；
- max_delta 0.04；
- seed 41；
- angle max = ±2°。

不允许扫角度、不允许自动延长训练。

---

## Execution

先进入仓库并同步：

```bash
git fetch origin
git pull --rebase origin main
git status --short
git rev-parse HEAD
git rev-parse origin/main
git push --dry-run origin HEAD:main
```

要求：

- working tree clean；
- `HEAD == origin/main`；
- dry-run push 成功。

然后运行测试：

```bash
/hy-tmp/realpde_venv_v2/bin/python -m pytest -q \
  tests/test_colleague_next_two_experiments.py \
  tests/test_post_train_diagnostics.py \
  tests/test_colleague_incremental_screen.py
```

再执行：

```bash
/hy-tmp/realpde_venv_v2/bin/python -u -B \
  tools/colleague_80pt/run_next_two_experiments.py \
  --real-root /hy-tmp/realpde_data/train_real \
  --base-checkpoint /hy-tmp/realpde_assets/colleague-80pt/20260921/extracted/handoff_colleague_80pt_archive_20260921/checkpoints/cno_final_all81.pt \
  --start-checkpoint /hy-tmp/realpde_assets/colleague-80pt/20260921/extracted/handoff_colleague_80pt_archive_20260921/checkpoints/residual_model_best.pth \
  --model-root tools/colleague_80pt/submission \
  --data-manifest /hy-tmp/realpde_data/data_manifest.tsv \
  --split-manifest /hy-tmp/realpde_runs/colleague_incremental_screen_20260921/colleague_dev16_manifest.json \
  --out-root /hy-tmp/realpde_runs/next_two_20260922_v1
```

若该 out-root 已存在，不删除、不覆盖，停止并汇报。

---

## Required diagnostics

严格执行：

- `docs/POST_TRAIN_DIAGNOSTICS.md`
- `docs/EXPERIMENT_BY_HORIZON_PROTOCOL.md`
- `docs/TRAINING_LOG_REVIEW_PROTOCOL.md`

Experiment A 每个 scale 必须有：

- final primary metrics；
- Future1..20 by-horizon；
- by-trajectory；
- trajectory × horizon；
- mean / fluctuation / energy；
- spatial maps；
- comparison vs current 80pt baseline。

Experiment B 必须有：

- final primary metrics；
- step 0/1000/2000/3000/4000/5000 training progression；
- Future1..20；
- by-trajectory；
- trajectory × horizon；
- mean / fluctuation / energy；
- spatial maps / spatial summary；
- deterministic training review log；
- comparison vs current 80pt baseline；
- comparison vs historical 5000-step no-augmentation control。

---

## Archive to Git

运行完成后执行 whitelist archive：

```bash
/hy-tmp/realpde_venv_v2/bin/python -u -B \
  tools/colleague_80pt/archive_next_two_experiments.py \
  --run-root /hy-tmp/realpde_runs/next_two_20260922_v1 \
  --dest docs/colleague_screening/results/20260922_next_two
```

然后：

```bash
git diff --check
git status --short
git add docs/colleague_screening/results/20260922_next_two
git commit -m "Archive 80pt next two experiment evidence"
git pull --rebase origin main
git push origin main
```

禁止提交：

- model checkpoint；
- H5；
- raw prediction cache；
- 原始大训练日志。

---

## Stop

完成 Git push 后停止。

不要：

- 自动 full train；
- 自动选 winner；
- 自动继续参数扫描；
- 访问 locked-final/private；
- 访问 Codabench；
- 打包 submission；
- 给出下一轮算法方案。

最终只返回：

```text
REALPDE NEXT TWO EXPERIMENTS

Status:
REVIEW_REQUIRED / BLOCKED

Execution commit:
...

Results commit:
...

Git evidence:
docs/colleague_screening/results/20260922_next_two

Experiment A:
COMPLETE / BLOCKED

Experiment B:
COMPLETE / BLOCKED

Standard diagnostics:
PASS / FAIL

Training review log:
PASS / FAIL

New full training started:
NO

Codabench accessed:
NO

Locked-final accessed:
NO

Missing items:
NONE / ...
```

如果任何 required artifact 缺失，返回 `BLOCKED`，不要写结论。
