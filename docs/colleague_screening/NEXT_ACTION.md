# NEXT_ACTION — CLEAN PIV HELD-OUT AoA=10° BENCHMARK

Status: `READY_FOR_EXECUTION / REVIEW_REQUIRED_AFTER_RUN`

## 1. Scientific question

验证：

> 当真实 PIV 训练数据完全没有见过 10° 攻角时，攻角增强是否能提高对真实 10° PIV 的泛化能力？

这是一个 **Control vs Augmentation** 的 clean benchmark。

注意其严格含义：

- **真实 PIV 10° 完全 held out（留出）**；
- 两条路线允许共享相同的官方 sim-pretrain CNO；
- 因此测试的是 **Sim→Real 场景下的未见 PIV 攻角泛化**；
- 不声称模型在 CFD/sim 阶段也从未见过 10°。

---

## 2. Data split — HARD CONTRACT

数据源只允许使用 frozen 50 Train / 16 Dev。

冻结 manifest：

`artifacts/loss_optimization_v9_20260901_run1/evidence/manifests/id_seed20260901.json`

预期 SHA256：

`42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`

若默认路径不存在，Codex 可以按 SHA / 内容寻找同一 manifest，但**不得重新随机划分**。

构造：

### Train

Frozen Train50 中删除全部 `AoA=10°`。

预期：

- 40 trajectories
- 不含任何真实 PIV 10°

### Inner Dev

Frozen Dev16 中删除全部 `AoA=10°`。

预期：

- 12 trajectories
- 不含任何真实 PIV 10°
- 仅用于 residual checkpoint selection 和训练诊断

### Held-out Test

Frozen Train50 + Dev16 中的全部真实 PIV `AoA=10°`。

预期：

- 14 trajectories
- 训练时不可见
- checkpoint selection 不可见
- 两条路线全部训练完成之后才允许统一 evaluation

### Locked-final

**禁止访问。**

本任务不读取 frozen locked-final 16 的文件名、metadata、Past20、Future20 或 prediction。

Split builder：

`tools/colleague_80pt/build_heldout_aoa_split.py`

必须生成并归档：

- `heldout_aoa_split_audit.json`
- `train_innerdev_manifest.json`
- `heldout_eval_manifest.json`

任何以下情况均 BLOCKED：

- Train != 40
- Inner Dev != 12
- Heldout != 14
- Train / Inner Dev 中出现 AoA=10°
- heldout 与 train/dev 有文件 overlap
- frozen manifest membership / SHA 不匹配

---

## 3. Common initialization

Control 和 Aug 必须从**完全相同**的官方 sim-pretrain CNO 开始。

checkpoint SHA256：

`82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61`

Codex 可以自行定位 GPU 上该 checkpoint 的实际路径，但 SHA 必须严格匹配。

禁止使用：

- 当前 80 分 residual checkpoint
- colleague all81 CNO
- 任何已经 fine-tune 过真实 PIV 10° 的 checkpoint

---

## 4. Two arms

### Arm A — Control（无攻角增强）

训练数据：

- Train40 only
- 不做任何 AoA augmentation

流程：

```text
official sim-pretrain CNO
→ CNO PIV fine-tune on Train40
→ residual corrector on Train40
→ checkpoint selection on InnerDev12
→ after all training: evaluate Heldout10°14
```

### Arm B — Aug（攻角增强）

与 Control 完全相同，唯一科研变量是训练期 AoA augmentation。

因为 10° 被完全拿掉，训练可见角度在 10° 周围形成：

```text
5°          15°
●───────────●
      ↑
     10° held out
```

Aug 只做：

**5° ↔ 15° same-Re bridge（相同 Reynolds 数下的 5°/15° 跨缺口均值场插值）**

对同 Re 的真实 5° 与 15°：

```text
delta_mean_uv =
    mean_t(Past20_neighbor)
  - mean_t(Past20_anchor)

Past20_aug =
    Past20_anchor + lambda * delta_mean_uv

Future20_aug =
    Future20_anchor + lambda * delta_mean_uv
```

固定：

- bridge pair = `5° ↔ 15°`
- same Re required
- x/y coordinate grid compatible
- augmentation probability = `0.5`
- lambda = uniform `[0.35, 0.65]`
- max gap = `10.1°`
- only Past20 used to construct mean-field shift
- Future20 never used to construct input transformation
- AoA/Re never become model inputs
- inference 不需要 AoA/Re

真实 10° trajectory **绝对不能作为 augmentation neighbor**。

Augmentation code：

`tools/colleague_80pt/aoa_meanfield_augmentation.py`

必须有至少：

- 4 eligible 5°/15° source trajectories
- 2 unique Re groups

否则返回 BLOCKED。

---

## 5. Training budget

两条 arm 完全 matched。

### Stage 1 — CNO Real-PIV fine-tune

代码：

`tools/colleague_80pt/train_cno_heldout_aoa.py`

固定：

- updates = `8723`
- stride = `1`
- batch = `8`
- lr = `1e-4`
- seed = `41`
- colleague Stage1 wake/ramp mean+fluctuation loss
- `0.05*TKE + 0.01*pzero`
- cosine LR schedule

Control：无增强。

Aug：启用 5°↔15° bridge augmentation。

### Stage 2 — Residual corrector

代码：

`tools/colleague_80pt/residual_multi.py`

固定：

- 从各自 Stage1 CNO 开始
- **不使用任何旧 residual checkpoint**
- updates = `20,000`
- eval interval = `2,500`
- batch = `8`
- lr = `2e-4`
- weight decay = `1e-5`
- hidden = `96`
- blocks = `2`
- max_delta = `0.04`
- TKE weight = `0.06`
- fixed temporal windows
- seed = `41`

Control：无增强。

Aug：继续使用完全相同的 5°↔15° bridge augmentation。

Stage2 checkpoint selection **只能使用 InnerDev12**。

---

## 6. Heldout evaluation timing

这是硬约束。

执行顺序必须是：

```text
1. build split
2. train Control CNO
3. train Aug CNO
4. train Control residual
5. train Aug residual
6. finish all checkpoint selection using InnerDev12
7. only now load heldout_eval_manifest
8. evaluate real PIV 10° heldout
```

Heldout10° 不得用于：

- early stop
- best checkpoint selection
- learning-rate decisions
- augmentation parameter selection
- retry / rerun selection

本轮不允许看到 heldout 分数后自动调参数重跑。

---

## 7. Required evaluation

对 InnerDev12 和 Heldout10°14 各统一 replay：

1. `control_cno`
   - Control residual 的 model_init
   - 表示 Control Stage1 CNO

2. `control_best`
   - Control residual best selected by InnerDev12

3. `control_final`
   - Control residual @20k

4. `aug_cno`
   - Aug residual 的 model_init
   - 表示 Aug Stage1 CNO

5. `aug_best`
   - Aug residual best selected by InnerDev12

6. `aug_final`
   - Aug residual @20k

必须输出：

- Rel-L2 / TKE / MVPE
- Future1..20
- by-trajectory
- trajectory × horizon
- mean field / fluctuation
- TKE energy ratio
- spatial maps
- residual correction help/hurt
- comparison summary

分析重点最终是：

```text
Aug vs Control on Heldout10°
```

不是与当前 80 分 baseline 比。

---

## 8. Required training evidence

Control 和 Aug 都必须有：

### CNO

- training config
- deterministic training review log
- loss progression

Aug 额外：

- CNO AoA augmentation audit
- eligible source trajectory count
- eligible Re group count
- actual augmentation use

### Residual

- run_config
- step 0 / 2500 / 5000 / 7500 / 10000 / 12500 / 15000 / 17500 / 20000 eval
- training review log
- final diagnostics
- model_init / best / final SHA recorded in run evidence

---

## 9. Environment autonomy

科研语义硬约束，运行环境软约束。

Codex 可自行修改：

- Git remote / tracking ref
- path discovery
- venv / Python executable
- CUDA_VISIBLE_DEVICES
- DataLoader workers
- launcher / PID / logs
- output-root version number
- archive / permissions / line endings
- 纯工程兼容代码

不得修改：

- frozen 50/16 membership
- heldout AoA=10°
- Train40 / InnerDev12 / Heldout14 逻辑
- sim-pretrain SHA
- Control/Aug 唯一变量原则
- 5°↔15° bridge semantics
- augmentation probability / lambda
- CNO / residual training budgets
- model / loss / loss weights
- seed
- heldout evaluation timing
- locked-final/Codabench 边界

---

## 10. Tests

先拉最新 main，然后运行：

```bash
python -m pytest -q \
  tests/test_colleague_next_two_experiments.py \
  tests/test_colleague_incremental_screen.py \
  tests/test_post_train_diagnostics.py
```

纯工程错误允许 Codex 最小修复、补测试并继续。

---

## 11. Execute

主 runner：

`tools/colleague_80pt/run_heldout_aoa_benchmark.py`

参考命令：

```bash
python -u -B tools/colleague_80pt/run_heldout_aoa_benchmark.py \
  --real-root /hy-tmp/realpde_data/train_real \
  --frozen-50-16-manifest artifacts/loss_optimization_v9_20260901_run1/evidence/manifests/id_seed20260901.json \
  --data-manifest /hy-tmp/realpde_data/data_manifest.tsv \
  --sim-pretrain-checkpoint <LOCATE_BY_SHA_82e842...> \
  --model-root tools/colleague_80pt/submission \
  --out-root /hy-tmp/realpde_runs/heldout_aoa10_20260922_v1 \
  --workers 4
```

若 manifest 默认路径不存在：

- 按 SHA256 查找冻结 50/16 manifest；
- 不得创建新 split 替代。

若 output root 已存在：

- 使用 v2/v3...
- 不删除旧 run。

---

## 12. Archive

训练完成后：

```bash
python -u -B tools/colleague_80pt/archive_heldout_aoa_benchmark.py \
  --run-root <ACTUAL_RUN_ROOT> \
  --dest docs/colleague_screening/results/20260922_heldout_aoa10
```

若 destination 已存在，加序号，不覆盖。

然后：

```bash
git diff --check
git add docs/colleague_screening/results/20260922_heldout_aoa10*
git commit -m "Archive clean heldout AoA10 benchmark"
git pull --rebase origin main
git push origin main
```

禁止提交：

- model checkpoint
- H5
- raw prediction cache
- raw full training logs

---

## 13. Stop

不要：

- 根据 Heldout10° 结果自动调参
- 自动 rerun
- 自动 full train
- 自动 package
- 自动 Codabench
- 访问 locked-final

完成后只返回：

```text
REALPDE CLEAN HELDOUT AOA10 BENCHMARK

Status:
REVIEW_REQUIRED / BLOCKED

Execution commit:
...

Results commit:
...

Frozen split:
PASS / FAIL

Derived split:
Train:
Inner Dev:
Heldout10:

PIV 10° leakage:
PASS / FAIL

Locked-final accessed:
NO

Control CNO:
COMPLETE / BLOCKED

Aug CNO:
COMPLETE / BLOCKED

Control residual 20k:
COMPLETE / BLOCKED

Aug residual 20k:
COMPLETE / BLOCKED

Bridge audit:
PASS / FAIL
Eligible trajectories:
...
Unique Re groups:
...

Inner-dev diagnostics:
PASS / FAIL

Heldout10 diagnostics:
PASS / FAIL

Training review logs:
PASS / FAIL

New full training started:
NO

Codabench accessed:
NO

Missing items:
NONE / ...
```

Codex 不做最终科研判断。Sol 在 Git Evidence Acceptance 通过后再 review。
