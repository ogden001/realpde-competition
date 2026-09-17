# NEXT_ACTION

## Goal

执行 `SPS-FULL50-01`：固定最终 full point predictor `@53582`，只改变 uncertainty head 的训练范围，比较：

- Baseline：已存在的 `full@53582 + teammate35 head trained on all 82`；
- Candidate：`full@53582 + 同一 teammate35 head trained on frozen 50 Train only`。

两者都在同一 frozen 16 Dev 上评估，同为 `1600` updates、同一初始化/optimizer/head/features、同一固定 bounds `floor=0.0025, mult=1.0`。唯一变量是 uncertainty-head training scope。

状态：`READY_FOR_EXECUTION / REVIEW_REQUIRED`

`REQUIRED_COMMIT = ee62a3e9bf9278581768033bcc56271f3a8b0736`

Sol implementation:
- `tools/realpde_sps_full50_head_ab.py`
- `tests/test_sps_full50_head_ab.py`

## Interpretation boundary

重要：full `@53582` point predictor 本身使用过全部 released trajectories，因此这 16 Dev **只对 uncertainty head 是未参与训练的**，并不是 point predictor 的 pristine unseen / OOF residual。

本实验只回答：

> 在最终 full predictor 不变时，余量模型用 50 条训练，是否比用全部 82 条训练更好？

不得把结果解释成真正 OOF 泛化证据。

## Execution role

本任务由 Codex / Luna **只执行**。

**Do not implement or redesign experimental logic. All algorithmic/source changes are owned by ChatGPT Sol. Codex is execution-only except for bounded environment/runtime fixes.**

允许的 bounded fix 仅限路径、shell quoting、CUDA / Torch / NumPy ABI、权限等不改变实验语义的运行环境问题。若 runner/test 需要算法、数据、Loss、split、update、bounds、评估逻辑等修改，立即停止并返回 `REVIEW_REQUIRED`。

## Preflight

1. 阅读 `AGENTS.md`、`docs/CHATGPT_CODEX_WORK_PROTOCOL.md`、本文件和 `docs/sota迭代/reviews/sps_teammate_package_audit_20260917/README.md`。
2. `MEMORY.md` 若不存在，只记录 `NOT_FOUND`，不要创建。
3. 必须确认：
   - 工作区无未知未提交改动；
   - `git fetch origin && git pull --rebase origin main`；
   - `HEAD == origin/main`；
   - `git merge-base --is-ancestor ee62a3e9bf9278581768033bcc56271f3a8b0736 HEAD` 成功；
   - `git push --dry-run origin HEAD:main` 成功。
4. 复用冻结资产，不重新训练 point predictor：
   - 50/16 manifest SHA `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`；
   - full checkpoint `@53582` SHA `f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce`；
   - official scorer SHA `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`；
   - **已有** Phase-B full82 teammate35 head：来自 `sps_teammate_final_20260917`，应为 `1600` updates，metadata `head_scope=full_specific_teammate35`。
5. 若已有 full82 head artifact 无法找到或 provenance/SHA/metadata 不匹配，**STOP**。不得重训一个 baseline 代替。

## Tests first

```bash
pytest -q \
  tests/test_sps_full50_head_ab.py \
  tests/test_sps_teammate_uncertainty.py \
  tests/test_sps_teammate_package.py
```

随后运行全仓 `pytest -q`，若成本正常。

若测试暴露实验语义问题，停止，不自行改算法代码。

## Frozen experiment

- point predictor：full SOTA-V2 `@53582`，完全冻结；
- uncertainty architecture/features：现有 teammate35，完全冻结；
- candidate training trajectories：frozen 50 Train；
- candidate canonical windows：`2052`；
- baseline head training trajectories：已有 all82；
- candidate 与 baseline updates：均 `1600`；
- seed：沿用现有 teammate runner；
- Loss：现有 Gaussian NLL reconstruction；
- AdamW：LR `1e-3`，weight decay `1e-5`；
- batch `8`；
- bounds：固定 `0.0025 + 1.0 * sigma`，**不重新调参**；
- evaluation：同一 frozen 16 Dev；
- point prediction 必须完全相同，parity `<=1e-7`。

Primary comparison：

`Candidate(full53582 + head_train50@1600)` vs `Baseline(full53582 + existing_head_train82@1600)`。

必须输出 aggregate SPS / coverage / mean width，以及 16 条 trajectory 的 paired SPS delta。

## Run

先解析以前已验证的资产路径：

```bash
DATA_ROOT=...
KIT_ROOT=...
MANIFEST=...
FULL_CKPT=...
FULL82_HEAD=...   # existing phase-B teammate35_full_head.pth, do not retrain
RUN_ROOT=/home/chyfuture/realpde_runs/sps_full50_head_ab_20260917
```

执行：

```bash
python tools/realpde_sps_full50_head_ab.py \
  --data-root "$DATA_ROOT" \
  --kit-root "$KIT_ROOT" \
  --manifest "$MANIFEST" \
  --full-checkpoint "$FULL_CKPT" \
  --full82-head-checkpoint "$FULL82_HEAD" \
  --out-dir "$RUN_ROOT/run" \
  --batch-size 8 \
  --workers 2 \
  --require-cuda
```

## Evidence

整理轻量 evidence 到：

`docs/sota迭代/reviews/sps_full50_head_ab_20260917/`

至少提交：
- `README.md`
- `summary.json`
- `head_training_summary.json`
- `per_trajectory_paired.csv`
- `per_trajectory_paired.json`

README 只记录事实，至少包括：
- baseline / candidate SPS / delta；
- coverage 与 mean width；
- 16 trajectory wins / ties / losses；
- median / mean / min / max paired SPS delta；
- point prediction parity；
- frozen asset SHA / execution commit；
- tests；
- bounded runtime fixes（如有）；
- 明确写 `locked-final/private/Codabench NOT accessed`；
- 明确写本实验不是 point-model OOF validation。

大 checkpoint、raw log、candidate head 留远程 artifact，不写 Git，只记录路径与 SHA。

## Constraints

- 不训练/修改 point predictor。
- 不训练新的 all82 baseline head。
- 不改 50/16 split。
- 不改 feature/head/loss/optimizer/1600 updates/bounds。
- 不做 200..2000 checkpoint search。
- 不做 calibration grid search。
- 不做 OOF / cross-fitting。
- 不生成 submission package。
- 不访问 locked-final/private Future20。
- **不提交 Codabench。**
- 不扩展第二个实验。

## Stop

实验完成后，将轻量 evidence commit + push 到 `main`，确认远端 `main` 已包含结果 commit，然后返回：

`REVIEW_REQUIRED`

最终科研结论由 ChatGPT Sol 复核。