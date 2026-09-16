# NEXT_ACTION

## Goal

执行一次最终 SPS 候选流水线：先在冻结 50/16 Dev 上验证同事 Codabench 包中真实的 `35-channel future-aligned uncertainty` 配方；只有通过预注册 Gate，才自动训练 full@53582 专属 head、生成候选包并做 clean-room smoke。**本任务绝不提交 Codabench。**

状态：`READY_FOR_EXECUTION / REVIEW_REQUIRED`

`REQUIRED_COMMIT = 64a9e8d48db8b60d7ed930e1f51387e07376e850`

源审计：`docs/sota迭代/reviews/sps_teammate_package_audit_20260917/README.md`

## Execution role

本任务由 Codex/Luna **只执行**。算法语义、runner、测试、package builder 已由 ChatGPT Sol 写入 Git。

允许的 bounded fix 仅限路径、shell quoting、Docker/CUDA、Torch/NumPy ABI、权限或不改变实验语义的环境适配。若需要修改 feature、head、loss、训练步数/选择逻辑、split、Gate、calibration、backbone 或 package inference 逻辑，立即停止并返回 `REVIEW_REQUIRED`，不得自行设计或实现。

## Preflight

1. 按 `AGENTS.md` / `MEMORY.md`（如存在）/ `docs/CHATGPT_CODEX_WORK_PROTOCOL.md` 做 preflight。
2. 必须满足：
   - 工作区干净；
   - `git pull --rebase origin main` 成功；
   - `HEAD == origin/main`；
   - `REQUIRED_COMMIT` 是 `HEAD` 或 `HEAD` 的祖先；
   - `git push --dry-run origin HEAD:main` 成功。
3. 复用此前 SOTA-V2 / SPS audit 已验证的远程资产路径，不重新训练 point backbone：
   - 固定 50/16 manifest，SHA `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`；
   - validation checkpoint `@32500`，SHA `6926722895611c38dee79cf16ce43b87c7de3d4a1305e9db431dadd4c1a7bf47`；
   - full checkpoint `@53582`，SHA `f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce`；
   - official scorer SHA `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`。
4. 若上述任何 frozen asset 无法唯一解析，停止并报告，不自行替换。

## Tests first

先运行：

```bash
pytest -q \
  tests/test_sps_teammate_uncertainty.py \
  tests/test_sps_teammate_package.py \
  tests/test_sps_stride1_replica.py \
  tests/test_sota_v2_adaptive.py \
  tests/test_sota_v2_adaptive_package.py
```

随后运行与本任务相关的现有 package / dense-window tests；成本合理时运行全仓 `pytest -q`。

测试不通过时，只允许 bounded execution fix；任何实验语义修改都必须停止。

## Frozen recipe

### Phase A: fixed 50 Train / 16 Dev

执行 `tools/realpde_sps_teammate_final.py`，冻结：

- point predictor：SOTA-V2 validation `@32500`，不训练、不修改；
- uncertainty training windows：canonical fixed stride-20，50 Train，共 `2052` windows；
- uncertainty features：`35-channel teammate35`；
- architecture：LayerNorm(35) + Conv3d(35→32) + GroupNorm + SiLU + 2 residual blocks + Conv3d(32→2)；
- hidden `32`，blocks `2`，dropout `0.0`，include pressure feature `true`；
- `sigma0=0.02`，sigma clamp `[1e-4, 1.0]`；
- Gaussian NLL on log-std；
- AdamW，LR `1e-3`，weight decay `1e-5`，batch `8`；
- maximum `2000` updates；
- every `200` updates 在同一 16 Dev 上扫 frozen 28-row `floor × mult` grid；
- 从 `200..2000` 的 checkpoint 中按 Dev SPS 选择 best；
- 不使用 stride1 / dense_all；
- 不改变 point prediction。

当前 frozen baseline：

- Dev SPS `45.07008160038756`
- mean UV width `0.02358330972492695`

Phase-A GO 必须同时满足：

1. `candidate SPS >= 46.57008160038756`，即至少 `+1.5`；
2. mean UV width `<= 1.15 × 0.02358330972492695`；
3. point prediction parity `max_abs_diff <= 1e-7`。

若 `SPS_TEAMMATE_NO_GO`：**立即停止**。不得训练 Phase B，不得打包，不得 smoke，不得开启任何新 SPS 实验。

### Phase B: conditional full-specific head

仅 Phase A 返回 `SPS_TEAMMATE_GO` 后执行：

- frozen full point predictor `@53582`，SHA 必须匹配；
- all 82 released PIV trajectories；
- canonical fixed stride-20 windows，共 `3383`；
- 同一 `teammate35` feature/head/loss/optimizer recipe；
- Phase-B updates **固定为 Phase-A 选中的 best checkpoint iteration**，不得在 full data 重新选 step；
- 不在 full data 上重新搜索 calibration；
- bounds 直接复用 Phase-A clean 16-Dev 选出的 `floor/mult`；
- 保存完整 full-head provenance。

## Runner invocation

先解析并记录以下环境变量对应的**既有 frozen 资产**：

```bash
DATA_ROOT=...              # all 82 released PIV h5 root used by current SOTA-V2
KIT_ROOT=...               # official RealPDE kit root containing scoring.py
MANIFEST=...               # frozen 50/16 manifest
VALIDATION_CKPT=...        # exact model_update_32500.pth
FULL_CKPT=...              # exact full model_update_53582.pth
RUN_ROOT=/home/chyfuture/realpde_runs/sps_teammate_final_20260917
```

然后执行一次 runner：

```bash
python tools/realpde_sps_teammate_final.py \
  --data-root "$DATA_ROOT" \
  --kit-root "$KIT_ROOT" \
  --manifest "$MANIFEST" \
  --validation-checkpoint "$VALIDATION_CKPT" \
  --full-checkpoint "$FULL_CKPT" \
  --out-dir "$RUN_ROOT/run" \
  --package-out-root "$RUN_ROOT/package_clean" \
  --execution-commit "$(git rev-parse HEAD)" \
  --batch-size 8 \
  --workers 2 \
  --require-cuda
```

Runner 自己负责 Phase-A Gate。NO_GO 时必须自动停止；GO 才可进入 Phase B/package。

## Package + clean-room smoke

仅当 Phase A GO 且 Phase B/package 成功后：

1. 使用 runner 生成的 `package_clean/submission.zip`；
2. 从 frozen 50-train manifest 中选择一条 released Train trajectory 作为 smoke fixture，**不得使用 locked-final/private data**；
3. 执行：

```bash
python tools/verify_sota_v2_adaptive_package.py \
  --zip "$RUN_ROOT/package_clean/submission.zip" \
  --full-checkpoint "$FULL_CKPT" \
  --kit-root "$KIT_ROOT" \
  --fixture "$SMOKE_FIXTURE" \
  --out "$RUN_ROOT/package_clean/smoke_report.json" \
  --tolerance 1e-7
```

Smoke 必须满足：

- status `PASS`；
- point prediction parity `max_abs_prediction_diff <= 1e-7`，目标 `0.0`；
- deterministic max diff `0.0`；
- prediction/lower/upper shape、dtype、finite 正常；
- pressure prediction为0且 pressure interval width为0；
- prediction 位于 `[lower, upper]`；
- 无 fallback；
- ZIP `<256 MiB`。

若 parity/smoke 任一失败：标记 `PACKAGE_INVALID / REVIEW_REQUIRED`，不得自行修改算法后重跑。

## Evidence

将轻量 evidence 复制/整理到：

`docs/sota迭代/reviews/sps_teammate_final_20260917/`

至少记录：

- `README.md`
- Phase-A `calibration_summary.json`
- Phase-A `calibration_grid.csv`
- Phase-A `checkpoint_evals.json`
- Phase-A `head_training_summary.json`
- 若 GO：Phase-B `full_head_summary.json`
- 若 GO：`package_build.json`
- 若 GO：`smoke_report.json`
- package ZIP 路径、bytes、SHA256（ZIP 不写 Git）
- frozen asset paths + SHA256
- tests 结果
- bounded fixes（如有）

README 必须明确：

- baseline SPS / candidate SPS / delta；
- selected uncertainty iteration；
- best floor/mult / coverage / mean width；
- point prediction parity；
- Phase-A GO/NO_GO；
- Phase-B 是否执行；
- package/smoke 状态；
- `locked-final/private/Codabench NOT accessed`。

## Hard constraints

- 不训练或修改 point backbone。
- 不改 35-channel feature recipe。
- 不改 teammate head architecture。
- 不改 Gaussian NLL、sigma0、2000 budget、200-step selection cadence。
- 不改固定 50/16 split、28-row grid 或 Gate。
- 不引入 stride1、OOF、pinball、asymmetric bounds、direct SPS loss、new architecture。
- 不访问 locked-final/private Future20。
- **不提交 Codabench。**
- NO_GO 后不扩大范围。
- 大 checkpoint / raw log / ZIP 只留远程 artifact 路径，不写 Git。

## Stop

完成后 evidence commit + push `main`，确认 `HEAD == origin/main`，然后返回：

`REVIEW_REQUIRED`

若 GO，明天由用户与 ChatGPT Sol 审阅 verified package 后再决定是否提交 Codabench。
