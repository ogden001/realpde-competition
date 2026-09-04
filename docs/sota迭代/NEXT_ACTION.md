# NEXT_ACTION

## Goal

基于当前线上 SOTA `P0-A + N2 CNO full@43260`，无人值守完成 `Residual Corrector + Adaptive Uncertainty` 的验证、全量 refit、打包和 clean smoke；明早产出可人工提交的 PRIMARY / BACKUP package。全程不访问 locked-final/private-test，不自动提交 Codabench。

## Tasks

1. 同步 `main`，阅读：
   - `docs/CHATGPT_CODEX_WORK_PROTOCOL.md`
   - `AGENTS.md`
   - `docs/sota迭代/README.md`
   - `docs/sota迭代/TEAMMATE_ADAPTIVE_PROBE_REFERENCE_20260905.md`
   刷新 runtime snapshot，确认 frozen 50/16 manifest、official scorer SHA、validation P0-A update `30900`、full P0-A update `43260`、82 trajectories / 3383 windows。

2. 在现有 repo 中做最小 bounded 实现并 TDD / smoke：
   - 冻结 backbone，只训练 `ResidualCorrector3D`；结构、future features、loss、LR schedule 严格按 reference 文档；validation 固定 `2400` updates，batch `8`，eval `600/1200/1800/2400`，最终只以固定 `2400` 做 Gate，不按中途指标挑 checkpoint。
   - 同时训练 `base Adaptive Uncertainty Head`：hidden `32`、2 blocks、Gaussian NLL、batch `8`、lr `1e-3`、weight decay `1e-5`、固定 `1400` updates。
   - 若 corrector Gate 通过，再训练 `corrected Adaptive Uncertainty Head`，同样固定 `1400` updates。
   - uncertainty calibration 只扫描 reference 中固定 `floor × mult` 网格；直接使用 official v9 SPS scorer。

3. Corrector Gate 固定为 validation@2400 相对 frozen backbone@30900：
   - Rel-L2 raw 至少改善 `2%`；
   - MVPE raw 至少改善 `2%`；
   - aggregate TKE raw degradation 不超过 `2%`；
   - 16 条 dev trajectory 中，TKE degradation `>15%` 的 trajectory 不超过 `2` 条。
   Gate 不通过则标记 `CORRECTOR_NO_GO`，跳过 full corrector refit，继续完成 base adaptive SPS BACKUP。

4. Gate 通过后，冻结 full backbone@`43260`，在全部 82 released trajectories 上 refit corrector：
   - 不做 checkpoint selection；
   - batch `8`；
   - validation 2400@2052 windows 的 traversal depth 映射到 full 3383 windows，固定 `3960` updates；
   - 其余 corrector 结构 / loss / optimizer / cosine schedule 不变。
   Adaptive uncertainty 不在 all-82 in-sample residual 上重新拟合，使用 validation-family 训练的 head 和 dev 上选出的 floor/mult。

5. 生成并 clean-room smoke：
   - `PRIMARY`：full@43260 + full residual corrector + corrected adaptive head（仅 Gate 通过时）；
   - `BACKUP`：full@43260 + base adaptive head；
   - 当前已线上验证的 full@43260 static bounds (`abs=0.0075, rel=0.02`) 保持现状作为既有 fallback，不重训。
   Smoke 必须检查 direct-vs-package numerical equivalence、shape `(N,20,32,64,3)`、float32、finite、pressure-zero、deterministic、lower/upper shape/order/finite，以及 package size。

6. 完成后更新 `docs/sota迭代/README.md`，并在 `docs/sota迭代/reviews/overnight_integrated_20260905/` 提交完整 review evidence。最后 commit + push `main`，工作树干净，状态 `REVIEW_REQUIRED`。

## Constraints

- 当前 backbone、P0-A、N2、30900 validation checkpoint、43260 full checkpoint均冻结；不得继续 backbone 训练。
- 不复用 teammate checkpoint/head 权重，只复用已核验结构和训练 recipe。
- runtime 输入只允许 Past20 `[u,v,p]` 和 tensor shape；不得使用 Re/AoA/HDF5 metadata/physical grid/mask。reference 中 x/y/t 仅为由 tensor index 生成的 normalized positional features。
- 不扩大模型、loss、feature、floor/mult 网格或训练预算；实现中出现语义歧义立即停止。
- 不使用 full-data training loss 选择 checkpoint；full corrector 固定 3960 updates。
- 不访问 locked-final/private-test，不提交 Codabench。
- **训练日志必须保留且提交**：
  - 每个训练/评估/打包阶段的 raw stdout+stderr 完整保存在远程 `OUT_ROOT/logs/`；
  - 同步生成完整 `*.review.log` 到 `docs/sota迭代/reviews/overnight_integrated_20260905/`；只允许把机器绝对根路径替换为 `$DATA_ROOT/$KIT_ROOT/$OUT_ROOT/$CHECKPOINT`，不得删除 metric / warning / error 行；
  - 因 `.gitignore` 全局忽略 `*.log`，仅对这些 `*.review.log` 使用 `git add -f`；
  - checkpoint、ZIP、NPZ、raw tensor 不进入 Git。

## Deliverables

远程 artifact 至少包含：
- `runtime_snapshot.json`
- validation corrector / base head / corrected head（若有）训练日志、metrics、checkpoint
- full corrector（若 Gate 通过）训练日志、metrics、checkpoint
- `artifact_manifest.json`
- PRIMARY / BACKUP package + build report + clean smoke report
- terminal `status.json`

Git evidence 至少包含：
- `docs/sota迭代/reviews/overnight_integrated_20260905/README.md`
- 关键 validation / trajectory / SPS metrics CSV/JSON
- runtime / artifact provenance 的可审阅摘要
- **所有实际执行阶段的完整 `*.review.log`**
- 实际 `EXECUTION_COMMIT`、checkpoint SHA、package SHA、Gate 结果、PRIMARY/BACKUP 状态。

## Stop

这是预授权的 bounded 无人值守任务。实现、tests、runtime snapshot、preflight、smoke 全通过后可直接 detached 执行，不需要再次等待用户确认。长任务启动后记录 PID / command / log / artifact / `RUNNING`，不持续轮询；流程结束后自动完成 evidence commit + push，然后停止于 `REVIEW_REQUIRED`。