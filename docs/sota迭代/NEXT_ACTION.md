# NEXT_ACTION — SPS-TEAMMATE-EXACT-SUBMIT-01

Status: `READY_FOR_EXECUTION / REVIEW_REQUIRED`

`REQUIRED_COMMIT = 6c7db859ab6cd083ec1f7fca7399fb8efac89739`

## Goal

把同事已确认的 SPS uncertainty recipe 复制到**当前 frozen full SOTA-V2 @53582 point predictor** 上，训练 head、按 frozen 16 Dev 直接选择 checkpoint + floor/mult、构建 submission ZIP 并做 clean-room smoke。

本任务只改 SPS 区间，不训练/修改 point predictor，不自动提交 Codabench。

## Tasks

1. Preflight：
   - 检查未知未提交改动；有则 STOP。
   - `git fetch origin && git pull --rebase origin main`
   - `HEAD == origin/main`
   - `git merge-base --is-ancestor 6c7db859ab6cd083ec1f7fca7399fb8efac89739 HEAD`
   - `git push --dry-run origin HEAD:main` 必须成功。
2. 运行定向 tests；通过后再启动 GPU。
3. 用 `tools/realpde_sps_teammate_exact_submit.py`：
   - point predictor：full SOTA-V2 `@53582`，SHA256 `f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce`；
   - uncertainty Train/Dev：项目 frozen 50/16 manifest；
   - exact teammate SPS recipe：35ch、h32/b2/drop0、masked Gaussian NLL、seed41、AdamW lr1e-3 wd1e-5、2000 updates、每200步评估；
   - 每个 eval checkpoint 在 frozen 16 Dev 上扫描固定 28-row `floor × mult`，按真实 SPS 选择全局 best；
   - point prediction parity 必须 <=1e-7。
4. 用 `tools/build_sota_v2_teammate_package.py` 直接构建 ZIP。
5. 用 `tools/verify_sota_v2_adaptive_package.py` 对 released Train/Dev real fixture 做 clean-room smoke。
6. 整理轻量 evidence，commit + push `main`，返回 `REVIEW_REQUIRED`。

## Frozen semantics

- 主模型：full SOTA-V2 @53582，完全冻结；**不重新训练，不改 forward，不改 checkpoint**。
- full point model 是 all-released 82-trajectory refit；本任务的 16 Dev 只对 uncertainty head held-out，不是 point-model OOF。
- uncertainty train split：现有 frozen 50 Train，2052 canonical windows。
- calibration/eval split：现有 frozen 16 Dev，659 canonical windows。
- uncertainty features/head：`sps_teammate_uncertainty_runtime.py` 的 teammate exact 35-channel + h32/b2/drop0/include_pressure。
- loss：`masked_gaussian_nll_from_log_std`，仅 target u/v 非零元素。
- seed：41。
- optimizer：AdamW，lr=1e-3，weight_decay=1e-5。
- batch：8。
- updates：2000；eval interval：200。
- calibration grid：
  - floor = [0, 0.0025, 0.005, 0.0075]
  - mult = [0.5, 1, 1.5, 2, 2.5, 3, 4]
- checkpoint selection：200..2000 所有 eval checkpoint 中，按 Dev **真实 SPS 最大**选择；SPS 相同则取更窄 mean width。
- 最终 package 直接使用 selected head + selected floor/mult。
- 不设科研增益 Gate：这次目标是“完整复制同事 SPS 策略并提交测试”，不是继续做消融。
- package/smoke 安全门失败则 STOP，不得上传。

## Tests

至少：

```bash
PYTHONPATH=tools pytest -q \
  tests/test_sps_teammate_uncertainty.py \
  tests/test_sps_teammate_exact_submit.py \
  tests/test_sps_teammate_package.py \
  tests/test_sota_v2_adaptive_package.py
```

若成本正常，再运行全仓 `pytest -q`。与本任务无关的已知平台测试失败要记录，不能擅自改实验语义。

## Run

复用前序 SOTA-V2/SPS evidence 中已经验证的资产，不重新生成模型。必须按 SHA 核对 full checkpoint。

建议：

```bash
DATA_ROOT=...
KIT_ROOT=...
MANIFEST=artifacts/loss_optimization_v9_20260901_run1/evidence/manifests/id_seed20260901.json
FULL_CKPT=...
RUN_ROOT=/home/chyfuture/realpde_runs/sps_teammate_exact_submit_20260918
EXECUTION_COMMIT=$(git rev-parse HEAD)

python tools/realpde_sps_teammate_exact_submit.py \
  --data-root "$DATA_ROOT" \
  --kit-root "$KIT_ROOT" \
  --manifest "$MANIFEST" \
  --full-checkpoint "$FULL_CKPT" \
  --out-dir "$RUN_ROOT/run" \
  --batch-size 8 \
  --workers 2 \
  --require-cuda

python tools/build_sota_v2_teammate_package.py \
  --full-checkpoint "$FULL_CKPT" \
  --head-checkpoint "$RUN_ROOT/run/teammate_exact_full53582_train50_head.pth" \
  --calibration-summary "$RUN_ROOT/run/summary.json" \
  --kit-root "$KIT_ROOT" \
  --out-root "$RUN_ROOT/package" \
  --execution-commit "$EXECUTION_COMMIT"

# FIXTURE 必须来自 released Train/Dev；禁止 locked-final/private。
python tools/verify_sota_v2_adaptive_package.py \
  --zip "$RUN_ROOT/package/submission.zip" \
  --full-checkpoint "$FULL_CKPT" \
  --kit-root "$KIT_ROOT" \
  --fixture "$FIXTURE" \
  --out "$RUN_ROOT/smoke_report.json" \
  --tolerance 1e-6
```

## Deliverables

Evidence 目录：

`docs/sota迭代/reviews/sps_teammate_exact_submit_20260918/`

至少 commit：

- `README.md`
- `summary.json`
- `head_training_summary.json`
- `checkpoint_evals.json`
- `selected_calibration_grid.json`
- `package_build.json`
- `smoke_report.json`

README 记录：

- execution commit；
- full checkpoint SHA；
- selected update / floor / mult；
- Dev SPS / coverage / mean width；
- point prediction parity；
- head SHA；
- ZIP remote path / bytes / SHA；
- smoke 与 tests；
- bounded runtime fixes（若有）；
- `locked-final/private/Codabench NOT accessed`；
- 明确说明 Dev16 只对 uncertainty head held-out。

大 checkpoint、raw log、submission ZIP 留远程，不写 Git。

## Constraints

**Do not implement or redesign experimental logic. All algorithmic/source changes are owned by ChatGPT Sol. Codex is execution-only except for bounded environment/runtime fixes.**

允许的 bounded fix 仅限路径、shell quoting、权限、CUDA/Torch/NumPy ABI、worker/runtime plumbing，且不得改变 feature、loss、split、seed、optimizer、updates、checkpoint selection、calibration grid、point predictor 或 package inference 语义。

若必须修改上述任何 source/实验语义：STOP，返回 `REVIEW_REQUIRED`，不要自行修。

禁止：

- 训练/修改 point predictor；
- 改 50/16 split；
- 改 teammate 35ch feature/head；
- 改 masked Gaussian NLL；
- 改 seed/optimizer/2000 updates/eval200；
- 增删 calibration grid；
- 额外跑 ablation / OOF / cross-fitting；
- 访问 locked-final/private Future20；
- 自动提交 Codabench；
- 自动决定下一轮实验。

## Stop

仅当 tests、训练、package、smoke 全部完成并且轻量 evidence 已 commit + push 到远端 `main` 后，返回：

`REVIEW_REQUIRED`

同时把最终 submission ZIP 的远程绝对路径与 SHA256 发给用户，等待用户 review/手工提交。
