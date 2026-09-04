# Track 1 实验注册表与可复用基线

> 这是 Track 1 的全局实验记忆。所有新的 loss、架构（arch）、特征工程（FE）和训练策略实验，在设计、训练或解读结果前都必须阅读本文件与项目根 `MEMORY.md`。
>
> 这里记录的是**可比较性协议和已完成的参考结果**，不是 Codabench leaderboard，也不授权使用 locked final 或 private test 选模型。

## 1. 不可变的 ID 实验协议

除非实验明确写为 OOD 或正式 Codabench，本注册表定义的 Track 1 ID 实验一律使用下列协议：

| 项目 | 固定值 |
|---|---|
| 数据 | 82 条本地 PIV H5 trajectories；不得按 window 重新划分 |
| manifest | `artifacts/loss_optimization_v9_20260901_run1/evidence/manifests/id_seed20260901.json` |
| manifest SHA-256 | `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347` |
| split | 50 train / 16 dev / 16 locked final，按完整 trajectory 划分 |
| seed | `20260901`，除非预注册多 seed 实验明确新增 seed |
| 评分 | 用户下载的 Track 1 starting kit v9 `scoring.py`；只报告其公开的五项子分与 raw errors |
| 模型选择 | 仅用 dev；locked final 仅在方案与 checkpoint 锁定后做一次审计，绝不用于调参、early stop 或挑选架构 |

任何新实验必须在结果文件中写入：manifest SHA、起始 checkpoint SHA、starting-kit scorer SHA、seed、batch/effective batch、optimizer/LR、训练预算、checkpoint 选择规则、代码版本/未提交 diff 以及实际命令。若其中任一项不同，结果不是本注册表基线的直接同口径比较。

## 2. Research cleanliness 与两条实验线

每个 Track 1 条目必须标记以下之一；这是结果解释和 baseline 复用的前提。

| 标记 | 定义与允许用途 |
|---|---|
| `CLEAN` | 初始化不含来源未知的 real/PIV 学习，且所有 PIV 学习、可学习特征/归一化拟合只使用本地 50 条 train trajectory。用于离线方法选择和 clean causal improvement 的主张。CNO 默认链路为 `sim_pretrain → local train PIV → dev`。 |
| `OFFICIAL_WARM_START` | 使用 `sim_real_ft` 或其派生 checkpoint。适合比赛导向的对照、冲榜和 Codabench；不得作为 clean offline 方法增益的证据。 |
| `NON_CLEAN / OTHER` | teacher、蒸馏、数据来源、划分或其他处理无法证明 holdout 隔离。仅作诊断或明确标记的比赛尝试。 |

官方 starting kit 说明 `sim_pretrain` 是 sim-only pretraining，`sim_real_ft` 是 real-PIV-finetuned；但没有披露后者的 trajectory 覆盖范围相对于本地 manifest 的关系。因此不能排除 `sim_real_ft` 已见过本地 dev 或 locked-final 的风险。这是对可复核性边界的保守分类，**不是**对官方数据泄漏的指控，也不影响其在 Competition Track 的合法用途。

Clean Offline Research 是默认轨：loss、architecture、point modeling、FE、training strategy、ablation 与任何以 dev 判断方法有效性的实验均只能使用 `CLEAN` family。Competition / Submission 是独立轨：允许 `sim_real_ft + 已在 Clean 轨验证的方法`，但其结果不得反向污染 Clean 轨的方法选择。新架构没有可继承的 clean checkpoint 时，应随机初始化并只学习本地 train PIV；可引用 historical warm-start 的绝对性能作 performance anchor，但不得称为严格架构对照。

## 3. 如何复用，而不是重训基线

1. 先选择下表中与研究问题对应、且 cleanliness 相同的 **baseline family**，在新实验记录开头写明其 `reference ID`。
2. 若新实验只改变 loss、架构或训练策略，直接引用该 baseline 的既有 dev 结果；**不要为了凑表格再次训练对照组**。
3. 对 loss 或训练策略候选，必须从与 reference 相同的初始化 checkpoint 出发，并使用相同 split、seed、batch、optimizer、训练预算与 checkpoint 规则；唯一计划内变量是要测试的因素。跨 cleanliness 或初始化 family 的绝对数值不可作单变量因果比较。对无法加载 CNO checkpoint 的新架构，可复用同为 `CLEAN` 的 CNO-E0 既有结果而不重训 CNO 对照，但必须保持 split、loss、预算和评估口径，并把“初始化/架构不可完全等同”登记为比较限制。
4. 只有候选越过预注册的 dev 门槛，才允许做第二 seed 或更长预算；然后才可进行一次 locked-final audit。
5. 禁止用不同 family 的绝对分数宣称某个因素有效。例如 `sim_pretrain → N2` 的 FE 结果不能与 `sim_real_ft → E0` 的 loss 结果作因果比较。

## 4. 当前可复用 baseline catalogue

### 4.1 `T1-ID-LOSS-E0-90M-S20260901` — Historical Official-Warm-Start Reference

| 项目 | 内容 |
|---|---|
| 状态 | `OFFICIAL_WARM_START`；历史 / competition-oriented performance reference，**不再是 Clean Offline Research 默认参考** |
| 模型 | 官方 kit 三通道 CNO（`u/v/p`） |
| 初始化 | `sim_real_ft/sim_real_cno.pth`，SHA-256 `82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61` |
| loss E0 | `MSE + 0.05*TKE`；其它 loss 权重为 0 |
| optimizer | AdamW，`lr=1e-5`，batch=8，workers=2 |
| 训练规则 | seed `20260901`；同一 ID split；约 90.7 分钟、7498 updates；预先记录的 `best_iteration=7498` |
| dev 参考（last/best） | Rel-L2 `0.168923`，TKE `0.538475`，MVPE `0.136146`（iteration 7498） |
| 一次性 locked-final audit | Rel-L2 `0.162509`，TKE `0.483187`，MVPE `0.132557`；仅作已锁定方案审计，**禁止作为后续选择标准** |
| 训练 checkpoint | 远程 `/home/chyfuture/realpde_runs/loss_opt_v9_20260901_run1/long_E0_s20260901/model_best.pth`，SHA-256 `5d02c8da5bcbcbcc47917b0021b1007b2036931a3a51483772f7607deeb4aff6` |
| 记录 | `artifacts/loss_optimization_v9_20260901_run1/report.md`；训练记录 `raw/long_E0_s20260901/` |

解释：初始化使用官方 `sim_real_ft`，其 PIV fine-tuning trajectory coverage 相对于本地 ID split 未知；故该 family 不是 clean offline baseline。E1/E2/E3 的长训可降低 Rel-L2/MVPE，却使 locked-final TKE 相比 E0 恶化 11.5%/11.7%/18.2%。历史结果继续保留，但只能在同为 `OFFICIAL_WARM_START` 的 competition-oriented family 内引用；不要重跑它们作为“基线”。

### 4.2 `T1-ID-CLEAN-CNO-E0-S20260902` — Planned Clean 通用 CNO Reference

| 项目 | 冻结计划（尚未执行） |
|---|---|
| 状态 | `PLANNED`；`CLEAN`；没有结果、checkpoint 或 locked-final audit |
| 目标 | 建立今后 clean loss、training strategy 与保持 CNO 输入接口的架构实验的通用 reference |
| 模型与初始化 | 官方 kit 三通道 CNO（`u/v/p`）；`sim_pretrain/sim_cno.pth`，SHA-256 `af85374bfd06c0e386ec803d777396c21484978392213025697c5a7470106b6b` |
| PIV 数据 | 固定 manifest `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`；仅 50 train trajectory 学习，16 dev 选模，16 locked final 仅在全部方案锁定后一次性审计 |
| loss | E0：`MSE + 0.05*TKE`；其它 loss 权重为 0 |
| 训练协议 | 复刻 4.1 的 AdamW `lr=1e-5`、batch=8、workers=2、seed `20260901`、约 90.7 分钟 optimizer-active 预算、评估节奏与 dev-only checkpoint 选择规则；唯一核心变量是 `sim_real_ft → sim_pretrain` |
| 评分与记录 | Track 1 starting kit v9 `scoring.py`；运行前记录 scorer SHA、代码版本/未提交 diff、实际命令及实际 updates；不得将 wall-clock 预算误写为预定 updates |
| 执行限制 | 本登记不授权训练、locked-final audit 或 Codabench 提交；完成前不得填入预测指标 |

### 4.3 `T1-ID-FE-N2-30M-S20260901` — 特征工程 / 残差头参考

| 项目 | 内容 |
|---|---|
| 状态 | `CLEAN`；**FE 专用参考**，只用于残差头与可推理特征的消融，不是通用 CNO baseline |
| 主干初始化 | `sim_pretrain/sim_cno.pth`，SHA-256 `af85374bfd06c0e386ec803d777396c21484978392213025697c5a7470106b6b` |
| N2 loss | `MSE=1.0, TKE=0.05, Rel=0.027514, MVPE=0.009757` |
| 训练规则 | AdamW `lr=1e-5`，batch=18，seed `20260901`；optimizer 实训 30 分钟；last@30min；评估和写盘不计时 |
| FE-00 CNO-only | 1258 updates；Rel-L2 `0.190821`，TKE `0.644070`，MVPE `0.144258` |
| FE-00R Raw-Control | 同一 178 参数零初始化残差头，输入 raw `u/v/p`；1271 updates；Rel-L2 `0.183850`，TKE `0.636973`，MVPE `0.133649` |
| 产物 | 远程 `/home/chyfuture/realpde_fe_v21/artifacts/fe_v21_seed20260901_30m/`；审阅报告见 `artifacts/fe_v21_seed20260901_30m/ChatGPT_review_report.md` |

FE 结论：Temporal、SpatialPhysics、PixelPosition 都未通过同时保护 Rel-L2、TKE、MVPE 的预注册门槛。SpatialPhysics 有显著 trajectory-level Rel-L2/TKE 信号，但 MVPE 显著恶化；它只能作为后续诊断对象，不能直接合入主线或触发第二 seed。

### 4.4 `T1-ID-P0A-N2-2H-S20260901` — 历史参考，禁止作默认对照

该实验为 `CLEAN`：从 `sim_pretrain/sim_cno.pth` 出发，仅学习本地 train PIV。它将 20 个 P0-A 特征直接扩展至 CNO 输入，batch=8，约 2 小时、4100 updates。它的 dev replay 为 Rel-L2 `0.156106`、TKE `0.553362`、MVPE `0.125436`，但与 4.1/4.3 的输入架构和训练时长不同。

因此它仅用于避免重复探索已知的 P0-A 路线，**不能**用于衡量 loss、残差头特征或训练策略的增益。详见 `artifacts/b1_p0a_n2_20260901_record.md`。

### `T1-ID-FE04-RAWSPATIAL8-30M-S20260901` — COMPLETED / NO-GO

- Reference ID: `T1-ID-FE-N2-30M-S20260901`。
- Research cleanliness: `CLEAN`。
- Sole planned variable: 在固定 8-channel、178 参数、末层 zero-init 的 residual head 中，将 FE-00R 的五个零通道替换为 `du/dx, du/dy, dv/dx, dv/dy, derivative_valid`；`u/v/p` 与 FE-00R 相同，导数严格复用 FE-02 pipeline。
- Frozen protocol: manifest `42b710…c347`，sim-pretrain init SHA `af853…6b`，N2，AdamW `1e-5`，batch=18，seed `20260901`，30 分钟 optimizer-active、last checkpoint；仅 train/dev，禁止 locked final。
- Frozen spec: remote `/home/chyfuture/realpde_fe04_v11/artifacts/fe04_raw_spatial8_v11/fe04_locked_spec.json`，revision `V1.1-r1`，SHA `cc6ae493…e80d`。r0 的 TKE replay 发生 float32/float64 舍入实现问题，未产生有效诊断/训练结果；修复后重新冻结并从头诊断。
- Pre-training diagnostic: FE-02 spatial residual 的 mean/fluctuation 分解以及 v9 TKE exact replay 已完成于同一输出目录；v9 aggregate 是 659 个 scored window 的 per-window TKE relative-L2 均值，而非全局能量比。详细结果在 `diagnostic_spatial/`。
- Training result: batch=18，1269 updates，1800.70 active optimizer seconds，last@30m dev raw errors: Rel-L2 `0.187316`，TKE `0.649739`，MVPE `0.140675`；官方 v9 subscores Rel/TKE/MVPE/Time/SPS: `91.4363/75.4791/93.4285/84.7241/9.6102`。
- Paired vs FE-00R: trajectory macro Rel-L2 delta (control − FE04) `-0.003483`, 95% CI `[-0.005565,-0.001421]`; TKE `+0.021255`, CI `[+0.011899,+0.030325]`; MVPE `-0.008308`, CI `[-0.015150,-0.003641]`. FE-04 improves neither protected Rel nor MVPE; hard gate **failed**.
- Decision: **NO-GO**. Do not launch second seed, longer budget or locked-final audit. Evidence: remote `/home/chyfuture/realpde_fe04_v11/artifacts/fe04_raw_spatial8_v11/{report_fe04.md,fe04_result.json}`; no Codabench submission.

### `T1-ID-POINT-V0-7500-S20260901` — PLANNED

- Reference family: standalone `PERSIST` screening; a future Clean CNO is a descriptive performance reference only, not a strict causal architecture control.
- Research cleanliness: `CLEAN`. Point MLP is randomly initialized and all learned quantities use only the manifest's 50 train trajectories.
- Question: can a Point MLP with normalized grid positional coordinates and no spatial-field context predict 20 future `u/v` frames better than persistence?
- Frozen interface: input at each grid point is `[x_index, y_index, u_1, v_1, ..., u_20, v_20]` (42 values); a shared `42→256→256→256→128→40` GELU MLP emits 20 future `u/v` values. It has no neighborhood, CFD, metadata, global encoder, or physical features. `x/y` are normalized grid indices, not asserted physical coordinates. Submission-compatible output appends zero `p`.
- Variants: `PERSIST`, `POINT-DIRECT`, and `POINT-RESIDUAL`. Residuals are defined in the same velocity space as the frozen loss pipeline and are restored to absolute fields before loss/scoring.
- Frozen protocol: manifest `42b710…c347`; seed `20260901`; train unit is a complete 20→20 window; batch/effective batch `8`; fixed train-window sampling; AdamW `lr=1e-5`; `7500` optimizer updates for both learned variants; last@7500 only (no dev best-checkpoint selection); `MSE + 0.05*TKE` from the same differentiable v9-compatible implementation used by the Clean CNO family. Train/dev only; no locked final, OOD, or Codabench.
- Decision gate (v9 raw dev micro errors vs `PERSIST`): `STOP_PURE_POINT` if either Rel-L2 or MVPE improves by <5%, or TKE worsens by >10%; `GO_POINT_V1` if Rel-L2 and MVPE each improve ≥5% and TKE worsens ≤10%; `STRONG_GO_POINT_V1` if Rel-L2 and MVPE each improve ≥10% and TKE worsens ≤5%. Report paired trajectory-macro bootstrap on all 16 dev trajectories as stability evidence, but it is not a veto.
- Required review: official-v9 subscore replay through full submission-compatible inference; horizon micro diagnostics; residual spatial error map with documented validity mask; runtime timing of full in-memory inference path. The task ends after the review package; it does not authorize Point-V1.

### `T1-ID-POINT-V0-7500-S20260901` — COMPLETED / STOP

- Reference ID: `T1-ID-POINT-V0-7500-S20260901`; standalone `CLEAN` screening; no Clean CNO causal comparison.
- Frozen execution: identity/raw velocity normalization; seed `20260901`; batch 8 complete windows; 7,500 updates per learned variant; AdamW `1e-5`; last@7500; train 2,052 windows; dev 659 windows / 16 trajectories.
- Official v9 dev raw errors (PERSIST / DIRECT / RESIDUAL): Rel-L2 `0.135578 / 0.226415 / 0.147084`; TKE `1.000000 / 0.930389 / 0.917402`; MVPE `0.132707 / 0.282945 / 0.132756`.
- Official v9 subscores (PERSIST / DIRECT / RESIDUAL): Rel `93.6515 / 89.8305 / 93.1496`; TKE `66.6667 / 68.2503 / 68.5541`; MVPE `93.7775 / 87.6061 / 93.7754`; Time `99.7993 / 98.6594 / 98.6988`; SPS `16.8128 / 7.7596 / 13.8919`.
- Persistence gate for POINT-RESIDUAL: Rel-L2 improvement `-8.49%`, TKE improvement `+8.26%`, MVPE improvement `-0.04%`; failed the required 5% Rel-L2 and MVPE thresholds. Decision: **STOP_PURE_POINT**; no Point-V1, locked-final audit, or Codabench submission.
- Stability: paired trajectory-macro bootstrap over all 16 dev trajectories; report as evidence only, not a veto. Horizon is fixed window-micro; spatial map has 2,040/2,048 valid pixels.
- Engineering evidence: Direct train `1471.67 s`, Residual train `1467.66 s`; CPU/HDF5 loading dominated wall time while GPU utilization was low. Optimize the data pipeline before the next point-modeling run, without changing this result.
- Evidence: remote `/home/chyfuture/realpde_runs/point_v0_v2_s20260901/artifacts/full/`; local review package `artifacts/point_v0_v2_s20260901_review/`.

### `T1-ID-FE-DATA01-B1-S20260902` — REVIEW_REQUIRED

- Reference ID: `T1-ID-FE-N2-30M-S20260901` (feature-engineering data-side reference only; no model comparison or training was performed).
- Research cleanliness: `CLEAN`; runtime-only, input-side descriptive diagnosis.
- Question / sole planned variable: characterize the frozen Batch-1 raw, temporal, fluctuation and input-side TKE-proxy features before choosing any feature-fusion experiment.
- Frozen protocol: manifest `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`; 50 train / 16 dev trajectories only; `T_in=20`, stride `20`, starts `0,20,...`, valid when `start+20 <= length`; float32 and `ddof=0`. Re/AoA, physical coordinates, mask, CFD, target-derived quantities and locked final were excluded.
- Implementation / dirty-tree note: baseline commit `cbafb6cda724ff660ccda7646a728792f51b5f04`; repository had pre-existing uncommitted work and the new untracked tool `tools/realpde_feature_diagnostic_batch1.py` (SHA-256 `f3c67594314f624d9e7b2bb0ced9452b1cd2e223ddd0e29d491a2a971dc9e88b`).
- Exact command: `python code/tools/realpde_feature_diagnostic_batch1.py --data-archive data/train_real.tar.gz --manifest artifacts/loss_optimization_v9_20260901_run1/evidence/manifests/id_seed20260901.json --feature-doc ../../Downloads/RealPDE_Track1_特征汇总文档_V1.5_正式版.md --out-dir artifacts/feature_summary_batch1` (run from workspace root; the relative source-document location is local-only and is not hard-coded in the tool).
- Evidence: repository-relative `../artifacts/feature_summary_batch1/` contains protocol JSON (SHA-256 `75f14a746e2ea229c140ee7787539c22051087c8347753ddfa2da09b970b734d`), value/trajectory/correlation CSVs, report and filled feature document. Shareable bundle: `../artifacts/feature_summary_batch1_chatgpt.zip`, SHA-256 `40ba3b222d769e53bfd4e68961786524e19777ae4df104cae4b1b5df9024fc13`.
- Core observations: 2102 train and 675 dev windows, all feature values finite; `std_u_20² = u2_prime_mean` and `std_v_20² = v2_prime_mean` are float32-rounding-level identities; speed vs abs(u) Pearson is `0.99991` (train) / `0.99992` (dev). Raw u/speed central distributions are close, whereas v and fluctuation/delta/TKE-proxy tails differ descriptively between train and dev. Value moments/counts use all values; quantiles and signed ratios use a fixed 100k bounded reservoir and are explicitly labelled as estimates.
- Locked-final audit: not run; protocol metadata records `locked_final_accessed: false`.
- Decision: `REVIEW_REQUIRED`. This is a data-side shortlist, not evidence that any feature improves a model. No follow-on analysis, training, locked-final audit or Codabench submission is authorized pending a bounded ChatGPT/user `NEXT_ACTION`.

### `T1-ID-FE-SPATIAL-DATA01-S20260902` — REVIEW_REQUIRED

- Reference: `T1-ID-FE-DATA01-B1-S20260902`; pure data-side follow-up, no model training or model-effect claim.
- Research cleanliness: `CLEAN`; runtime-only pixel-space diagnostic.
- Question / sole planned variable: characterize `du_dx_pixel`, `du_dy_pixel`, `dv_dx_pixel`, `dv_dy_pixel`, and derived `vorticity_pixel` on frozen train/dev windows, including image-edge sensitivity.
- Frozen definition: spacing `1` pixel; centered finite difference at interior pixels; first-order forward at index 0 and first-order backward at the last index for each axis; float32; no smoothing, clipping, normalization, coordinates, mask, target, CFD, Re/AoA or locked-final access.
- Protocol: manifest `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`; 50 train / 16 dev trajectories; `T_in=20`, stride `20`; actual H×W `64×128`; 2102 / 675 windows; locked-final access `false`.
- Implementation / dirty-tree note: run saw HEAD `8ceb5fffa0948118f5f5c8e61302c90c20d36727`; repository remains dirty from unrelated user work and the new Spatial tool. The executed implementation SHA recorded in `spatial_definition_v1.json` is `2fa8cd5b2b33532c365673f04cd621e6d14f3b1188b6538ca56865306ab267ad`.
- Exact command: `python code/tools/realpde_spatial_diagnostic_batch1.py --data-archive data/train_real.tar.gz --manifest artifacts/loss_optimization_v9_20260901_run1/evidence/manifests/id_seed20260901.json --out-dir artifacts/spatial_diagnostic_batch1` (run from workspace root).
- Core observations: primitive gradient means/p95 are respectively du_dx `0.0007757/0.009284` train and `0.0008058/0.01028` dev; du_dy `-0.0004949/0.01444` and `-0.0005389/0.01526`; dv_dx `-0.00007716/0.003081` and `-0.00007935/0.002974`; dv_dy `0.0001030/0.003894` and `0.0001059/0.003779`. Vorticity mean/p95 is `0.0004177/0.01417` train and `0.0004595/0.01506` dev. All values are finite.
- Edge sensitivity: outer-edge/interior abs_mean ratios train/dev are du_dx `0.800/0.822`, du_dy `0.615/0.633`, dv_dx `0.572/0.573`, dv_dy `0.768/0.749`; abs_p95 ratios are all below `0.82`, so edge is not the dominant magnitude source under the frozen rule.
- Correlation / redundancy: vorticity vs `(dv_dx-du_dy)` Pearson/Spearman `1.0/1.0` (deterministic identity); vorticity vs du_dy is about `-0.978/-0.904`, and vs dv_dx `0.192/0.357` in train, qualitatively unchanged in dev.
- Decision: `REVIEW_REQUIRED`; final labels are du_dx/du_dy/dv_dx/dv_dy `KEEP` and vorticity `KEEP (derived summary)`. Do not treat this as model-effect evidence or authorize feature fusion, training, locked-final audit or Codabench.
- Evidence: repository-relative `../artifacts/spatial_diagnostic_batch1/` contains `spatial_definition_v1.json`, value/trajectory/edge/correlation CSVs and `report.md`.

## 5. 新实验登记模板

### `T1-ID-POINT-DATA-BENCH-20260902` — PAUSED / DIAGNOSTIC ONLY

- Reference: Point-V0 clean screening, engineering follow-up only; no model retraining or accuracy re-evaluation.
- Research cleanliness: `CLEAN`; train split only (50 trajectories), fixed seed `20260901`, batch `8`, fixed repeated window order.
- Question / sole planned variable: locate the approximately 196 ms/update data-pipeline cost and test bounded HDF5 worker/handle/RAM-cache candidates without changing window semantics.
- Profiles: A data-only and B Point-V0 training-path timing; B0 current loader first (`workers=2`, `pin_memory=True`, `persistent_workers=False`), then B1 worker scan, B2 worker-local handles, B3 trajectory RAM cache.
- Gates: exact `DATA_EQUIVALENCE` on 100 fixed train windows; report windows/s, latency, data wait ratio and RSS. No dev/final/scorer/Codabench access; no automatic Point-V1.
- Evidence: `docs/coordination/CHATGPT_HANDOFF_POINT_DATA_BENCHMARK.md`; remote run was intentionally paused after B0 at the user's request. Smoke evidence is sufficient to identify the HDF5 bottleneck; no complete candidate ranking is claimed.

### `T1-ID-POINT-V1-LOCAL3-20260902` — IN_PROGRESS

- Reference: Point-V0 clean screening and `T1-ID-POINT-DATA-BENCH-20260902`; no Point-V0 retraining, locked-final access, or Codabench.
- Research cleanliness: `CLEAN`; train-only Phase 1/2, then the pre-registered 16-trajectory dev gate/evaluation in Phase 3.
- Sole model variable: no spatial-field context → deterministic replicate-padded 3×3 local u/v history context; normalized grid x/y only; raw-space residual output.
- Frozen model/loss: `362→256→256→256→128→40`, GELU, AdamW, `MSE + 0.05*TKE`, batch 8, seed `20260901`, 50/16/16 manifest SHA `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`.
- Phase 1: coarse B3_RAM vs B3_PACKED (`20/100/1`), exact fixed seeded shuffled train-window order and DATA_EQUIVALENCE; only coarse winner receives formal `100/1000/3`.
- Phase 2: same initialization/window order, LR `1e-5` vs `1e-4`, 500 train updates; no dev LR selection. Phase 3A `last@1500` gate may continue the same model to `last@7500` only when the registered gate passes.
- Execution: `tools/realpde_point_v1_local3_runner.py`; remote run `/home/chyfuture/realpde_runs/point_v1_local3_s20260901`; Docker memory hard limit 48 GiB / swap 48 GiB; current stage is Phase 1 coarse.

### `T1-ID-POINT-V1-LOCAL3-20260902` — COMPLETED / STOP_LOCAL3_EARLY

- Phase 1 froze `POINT_RAM_PIPELINE_V1=B3_PACKED`: formal training-path `306.79 windows/s`, step latency `26.08 ms`, data-wait ratio `5.77%`, cache build `31.17 s`, cache `1.04 GB`, process-tree peak RSS `2.06 GB`; DATA_EQUIVALENCE passed exactly.
- Phase 2 froze `POINT-V1 LR=1e-4`: both 500-update runs finite; final total train loss `0.04198` (1e-4) vs `0.07075` (1e-5), with no NaN/Inf.
- Phase 3A: LOCAL3 residual, raw velocity, replicate-padded 3×3 context, 1500 updates, `last@1500`; train-side losses finite. Dev comparison versus PERSIST: PERSIST Rel-L2/TKE/MVPE `0.135578/1.000000/0.132707`; LOCAL3 `0.155252/0.832650/0.140822`.
- Screening gate deltas (Rel-L2/TKE/MVPE): `-14.51% / +16.74% / -6.12%`; Rel-L2 and MVPE did not improve, so `STOP_LOCAL3_EARLY`. No Phase 3B, no locked-final, no Codabench. `NEXT_CANDIDATE` is not authorized by this result.
- Evidence: remote `/home/chyfuture/realpde_runs/point_v1_local3_phase3_retry3_s20260901`; `phase3_local3/screening_gate.json`, `dev@1500_local3/evaluation.json`, `dev@1500_persist/evaluation.json`, `last@1500.pt`.

### `T1-ID-POINT-LOSS-GRAD-DIAG-20260902` — COMPLETED / DIAGNOSTIC ONLY

- Reference: `T1-ID-POINT-V1-LOCAL3-20260902`; existing `last@1500.pt`, no retraining, no optimizer step.
- Research cleanliness: `CLEAN`, train-only; B3_PACKED, batch 8, seed `20260901`, 32 fixed seeded-shuffled train batches.
- Frozen loss decomposition: separately compute `∇θ MSE` and `∇θ (0.05*TKE)` with trainable parameters only; no loss/LR/architecture sweep.
- Results: grad-ratio mean/median `57.949/44.764` (min/max `19.821/189.385`); cosine mean/median `-0.0853/-0.0731` (mixed batch directions); mean per-batch scalar `(0.05*TKE)/MSE` `103.472×`; all finite.
- Diagnostic label: `TKE_GRADIENT_STRONGLY_DOMINANT`. This supports a narrow objective-imbalance hypothesis but does not authorize changing lambda or training a balanced-loss model.
- Evidence: `docs/coordination/CHATGPT_HANDOFF_POINT_LOSS_GRADIENT_DIAGNOSTIC.md`; remote `/home/chyfuture/realpde_runs/point_loss_grad_diag_s20260901`; dev/locked-final/scorer/Codabench all `NO`.

每产生一个实质性候选或完整结果，在本文件末尾追加一条；不要回写或覆盖旧结论。

### `T1-ID-MF-C01-TKE2X-S20260901` / `T1-ID-MF-C01-CONDGAIN-S20260901` / `T1-ID-MF-C01-SPATIALGAIN-S20260901` — COMPLETED / REVIEW_REQUIRED

- Reference: `T1-ID-MF01-S20260904` at update 1500; existing `Control@1500` and `MF-01@1500` predictions are reused for Stage 0 only.
- Research cleanliness: `CLEAN`; frozen 50 train / 16 dev / 16 locked-final manifest, seed `20260901`, P0-A, `sim_pretrain`, official v9 scorer.
- Sole variables: E1 changes only N2 TKE weight `0.05 -> 0.10`; E2 adds only runtime-safe 5-scalar conditional gain `alpha=1+0.20*tanh(g)` with zero-init linear `5->1`; E3 adds only zero-init spatial `1x1 Conv2d 5->1` gain map with the same alpha range. No MF-02/spectral/independent backbone.
- Fixed protocol: 1500 optimizer updates, evaluations/checkpoints at 500/1000/1500, AdamW `lr=1e-5`, batch `8`, same window protocol and effective batch as MF-01.
- Execution commit: `f4dc26d`; implementation/test commits: `b151cb1`, `f4dc26d`.
- Results at 1500 (Rel-L2 / TKE / MVPE): E1 `0.190596 / 0.641192 / 0.158759`; E2 `0.183465 / 0.634980 / 0.155231`; E3 `0.183428 / 0.636492 / 0.154953`. Relative to MF-01: E1 `+1.161% / -0.614% / -1.117%`; E2 `-2.624% / -1.577% / -3.314%`; E3 `-2.644% / -1.343% / -3.488%` (lower error is better).
- Trajectory wins versus MF-01 (Rel-L2 / TKE / MVPE): E1 `2/16 / 5/16 / 8/16`; E2 `16/16 / 4/16 / 14/16`; E3 `16/16 / 3/16 / 14/16`.
- Stage 0 oracle was diagnostic-only on existing MF-01@1500 predictions; best unconstrained fixed spatial oracle Rel-L2/TKE was `0.185613 / 0.481970`, with alpha max `7.416`, so it is not a deployable result.
- Decision: E1/E2/E3 `NO_GO` for automatic continuation; aggregate Rel/MVPE gains in E2/E3 do not have stable TKE trajectory support. No locked-final/private-test or Codabench access.
- Evidence: `/home/chyfuture/realpde_runs/mf_energy_campaign01/` and `docs/coordination/CHATGPT_HANDOFF_MF_ENERGY_CAMPAIGN01.md`.

```markdown
### `T1-ID-<FACTOR>-<DATE>`

- Reference ID: `T1-ID-...`
- Research cleanliness: `CLEAN` / `OFFICIAL_WARM_START` / `NON_CLEAN / OTHER`
- Question / sole planned variable:
- Fixed protocol deviations: none / explicit reason
- Initialization checkpoint SHA:
- Model and input interface:
- Loss / optimizer / LR / batch / seed:
- Budget and checkpoint rule:
- Dev raw errors + official v9 subscores:
- Paired trajectory result / uncertainty:
- Locked-final audit: not run / one-time result after lock
- Decision: advance / no-go / diagnostic only
- Evidence paths and exact command:

### `T1-ID-MF01-CONTROL-S20260904` + `T1-ID-MF01-S20260904` — COMPLETED / REVIEW_REQUIRED

- Research cleanliness: `CLEAN`; fixed manifest `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`, 50 train / 16 dev / 16 locked-final.
- Question / sole planned variable: output-level Direct Future20 versus Mean/zero-mean-Fluctuation reconstruction; no loss, feature, branch-capacity, optimizer, LR, batch, seed or data changes.
- Shared initialization: official `sim_pretrain/sim_cno.pth`, SHA-256 `af85374bfd06c0e386ec803d777396c21484978392213025697c5a7470106b6b`; official v9 scorer is used.
- Fixed protocol: P0-A, N2 (`MSE + 0.05*TKE + 0.027514*Rel-L2 + 0.009757*MVPE`), AdamW `lr=1e-5`, batch `8`, workers `2`, seed `20260901`, 1500 updates, checkpoints `500/1000/1500`.
- MF-01 initialization smoke: reconstruction max absolute error `2.3841858e-7`; pressure max absolute error `0`; fluctuation temporal-mean max absolute error `3.1590463e-7`.
- Locked-final audit: not run. Codabench: not accessed.
- Code: `tools/realpde_mf01.py`; test: `tests/test_mf01_output.py`.
- Results: at 1500, MF-01 Rel-L2/TKE/MVPE `0.188409/0.645156/0.160552` versus Control `0.193675/0.633786/0.165178`; relative changes `-2.72%/+1.79%/-2.80%`. Trajectory wins Rel/TKE/MVPE `13/16`, `7/16`, `10/16`; all-three `3/16`.
- Decision: `MF01_NO_GO` / `REVIEW_REQUIRED`; decomposition improved mean and fluctuation diagnostics but official TKE did not improve. No MF-02 auto-launch.
- Evidence: `/home/chyfuture/realpde_runs/mf01_control_20260904/`, `/home/chyfuture/realpde_runs/mf01_s20260904/`, and `docs/coordination/CHATGPT_HANDOFF_MF01.md`.
```

## 6. 全局维护规则

- 新实验先更新本注册表的“计划”或“完成”条目，再开始高成本训练；更新应追加事实，不得以事后叙述改写原计划或抹去失败结果。
- 默认只维护已经有可复核产物的结果。未完成任务只能标记 `PLANNED` 或 `IN_PROGRESS`，不应被称为已完成 baseline 或成功候选。
- 产物、checkpoint、submission archive 仍留在 `artifacts/` 或远程路径；本文件只保存小型元数据、SHA、结果和可复现实验路径。
- `code/docs/submission_log.md` 仅记录真实 Codabench 提交；本注册表的离线数值绝不能填入其中。

### `T1-ID-FE-INCR-PERSIST-RIDGE-S20260902` — COMPLETED / REVIEW_REQUIRED

- Reference: `T1-ID-FE-DATA01-B1-S20260902` and `T1-ID-FE-SPATIAL-DATA01-S20260902`; minimal supervised incremental-value probe only.
- Research cleanliness: `CLEAN` with respect to the frozen local train/dev protocol; baseline is the previously registered deterministic `PERSIST` prediction, not a CNO checkpoint. No checkpoint inference, neural training, locked-final/private-test access or Codabench.
- Frozen protocol: manifest `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`; 50 train / 16 dev trajectories; complete 20→20 windows, stride 20; 2052 / 659 windows; runtime H×W `32×64`.
- Probe definition: per-pixel Raw-Control is last-frame `(u,v)` (dim 2). Temporal adds train-window mean/std/recent delta for both channels (dim 8 total). Spatial adds four last-frame pixel finite differences (dim 6 total). Joint dim 12. TKE and vorticity are not independent inputs; vorticity remains the deterministic derived value `dv_dx_pixel - du_dy_pixel`.
- Ridge: independent closed-form residual probe per group, residual target `target - PERSIST`, feature mean/std train-only over `4,202,496` rows, alpha `1e-2` scaled by train-row count; no dev tuning.
- Dev raw corrected-field errors (Rel-L2 / TKE / MVPE): Raw-Control `0.131353 / 0.987575 / 0.130701`; Raw+Temporal `0.118340 / 0.940578 / 0.109454`; Raw+Spatial `0.130241 / 0.973685 / 0.129355`; Raw+Temporal+Spatial `0.116346 / 0.936060 / 0.107228`.
- Delta vs Raw-Control (Rel-L2 / TKE / MVPE): Temporal `-0.013013 / -0.046997 / -0.021247`; Spatial `-0.001112 / -0.013889 / -0.001346`; Joint `-0.015007 / -0.051515 / -0.023473`. All are same-direction improvements; no metric trade-off was observed in this linear probe.
- Trajectory-macro win rates vs Raw-Control (Rel/TKE/MVPE): Temporal `0.875 / 1.000 / 0.938`; Spatial `0.812 / 1.000 / 0.750`; Joint `0.938 / 1.000 / 1.000`. This is stability evidence only, not a neural-model guarantee.
- Shortlist: Raw+Temporal, Raw+Spatial, and Raw+Temporal+Spatial `KEEP_FOR_MODEL_PROBE`; no additional feature catalog expansion. Final protocol state `REVIEW_REQUIRED`.
- Implementation SHA-256: `6e37aa883c6080df3ce60b15173f1d451002a50694c55aaff47605ca4eea3567`. Exact command and artifact inventory are in `docs/coordination/CHATGPT_HANDOFF_FE_INCREMENTAL_PROBE.md`; generated outputs remain under `../artifacts/fe_incremental_probe_s20260902/` and are not committed.

### `T1-ID-FE-INCR-FROZEN-CNO-E0-RIDGE-S20260902` — COMPLETED / REVIEW_REQUIRED

- Reference: registered `T1-ID-LOSS-E0-90M-S20260901` historical `OFFICIAL_WARM_START` E0 `model_best.pth`; frozen throughout. Checkpoint SHA-256 `5d02c8da5bcbcbcc47917b0021b1007b2036931a3a51483772f7607deeb4aff6`; starting-kit scorer SHA-256 `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`.
- Provenance / protocol: official vendored CNO3d `3→3`, input/output `[B,20,32,64,3]`, raw `u/v` plus zero pressure; exact frozen 50 train / 16 dev manifest SHA `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`; complete 20→20 windows, stride 20; 2052 / 659 windows; no final/private-test or Codabench.
- Implementation SHA-256 `73724987b0f471d74c803f07ce866273b76a1faedec2c1c6b31e75aa1db588b1`; run source commit `c51b2fbf6d8656c455872d68bb394106f2de18a1` with unrelated dirty/untracked work; exact smoke/formal commands and artifact inventory are in `docs/coordination/CHATGPT_HANDOFF_FE_INCREMENTAL_CNO_E0.md`.
- Formal runtime: GPU inference plus CPU closed-form ridge `834.0 s` (13.9 min); train-only feature standardization; alpha `1e-2*n`; four fixed groups only (Raw-Control dim 2, Raw+Temporal dim 8, Raw+Spatial dim 6, Joint dim 12); TKE-proxy and vorticity excluded as independent inputs.
- Frozen CNO raw dev Rel-L2 / TKE / MVPE: `0.168923 / 0.538475 / 0.136146`, matching the registered E0 reference. Corrected dev errors: Raw-Control `0.168162 / 0.594538 / 0.135999`; Temporal `0.161318 / 0.608969 / 0.135610`; Spatial `0.166255 / 0.624081 / 0.135474`; Joint `0.159876 / 0.631270 / 0.135046`.
- Deltas vs Raw-Control (Rel/TKE/MVPE): Temporal `-0.006843 / +0.014432 / -0.000390`; Spatial `-0.001907 / +0.029543 / -0.000525`; Joint `-0.008286 / +0.036732 / -0.000954`. Joint vs Temporal `-0.001442 / +0.022301 / -0.000564`; Spatial has independent Rel/MVPE value beyond Temporal but worsens TKE.
- Trajectory win rates vs Raw-Control (Rel/TKE/MVPE): Temporal `1.000/0.313/0.563`; Spatial `1.000/0.125/0.750`; Joint `1.000/0.250/0.625`. No all-three-metric stable gain; labels are Temporal/Spatial/Joint `LOW_INCREMENTAL_VALUE` under conservative protection, with Rel/MVPE signals review-only.
- Comparison: PERSIST ridge probe improved all three metrics, while this strong-CNO probe has Rel/MVPE gains and TKE penalties. Required conflict label: `FEATURE_VALUE_POSITIVE_BUT_FUSION_HISTORY_NEGATIVE`; historical FE-01/FE-02 fusion outcomes are not evidence that the underlying feature contains no information.
- Final decision: **STOP** automatic fusion training; final execution state `REVIEW_REQUIRED`. Artifacts remain local under `../artifacts/fe_incremental_probe_cno_e0_s20260902/`; generated outputs are not committed.

### `T1-ID-POINT-LOCAL3-BALANCED-L001-S20260902` — IN_PROGRESS

- Reference: `T1-ID-POINT-V1-LOCAL3-20260902` and `T1-ID-POINT-LOSS-GRAD-DIAG-20260902`; final bounded Point-MLP loss-balance experiment.
- Research cleanliness: `CLEAN`; train-only optimization followed by the preregistered 16-trajectory dev gate; locked-final and Codabench prohibited.
- Sole variable: frozen TKE coefficient `0.05 -> 0.001`, selected only from the train-only median gradient ratio `44.764` (`0.05/44.764 ≈ 0.00112`).
- Fixed protocol: random initialization, seed `20260901`, B3_PACKED, batch `8`, fixed seeded-shuffled train-window order, AdamW `1e-4`, LOCAL3 `362→256→256→256→128→40` GELU, replicate-padded 3×3 raw u/v history plus normalized x/y, raw residual output, p=0, 1500-step screen and gate-controlled continuation to 7500.
- Execution: standalone `tools/realpde_point_local3_balanced_runner.py`; remote run and exact command will be recorded in the completion handoff. Old `last@1500.pt` is not reused.
- Current state: implementation/smoke pending; no dev, locked-final or Codabench access yet.

### `T1-ID-POINT-LOCAL3-BALANCED-L001-S20260902` — COMPLETED / STOP_BALANCED_LOCAL3_EARLY

- Random-init LOCAL3 with the sole change `lambda_tke=0.001` completed 1500 updates on B3_PACKED and stopped at the preregistered screen; no 7500 continuation.
- Dev raw v9 errors (PERSIST / candidate): Rel-L2 `0.13557753 / 0.14545619`, TKE `1.00000000 / 0.85760111`, MVPE `0.13270709 / 0.13071164`; relative improvements `-7.286% / +14.240% / +1.504%` (Rel/TKE/MVPE). Since lower error is better, TKE improved by `14.240%` and passed its ≤10% degradation condition; the screen failed only on Rel-L2 improvement ≤0%.
- Train-only gradient snapshots were finite; median weighted-TKE/MSE gradient ratio was `8.888` at initialization and `1.631` at update 1500 (four fixed train batches each). No old checkpoint was reused.
- Checkpoint SHA-256 `95df4c9e69a287dc55290208a07a754e2cebf2706ead521e407e4f68347d6122`; runner SHA-256 `55eb38d0c7a99e3265adee9a8f7aa696f7caf198d606c099b240a2699ff41a5d`; manifest SHA unchanged `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`.
- Locked-final and Codabench: not accessed. Do not infer that all Point modeling is invalid; this only closes the preregistered balanced LOCAL3 candidate.
- Handoff: `docs/coordination/CHATGPT_HANDOFF_POINT_LOCAL3_BALANCED_L001.md`; small review artifacts under `../artifacts/point_local3_balanced_l001_s20260902_review/`.

### `T1-ID-HYBRID-CNO-POINT-H1-S20260902` — IN_PROGRESS

- Reference: `T1-ID-FE-N2-30M-S20260901` FE-00 CNO-only, used only as a frozen CLEAN FE-specific backbone; this is not a general CNO architecture comparison.
- Sole variable: add a zero-initialized LOCAL3 Point residual head to the frozen CNO. CNO parameters remain `requires_grad=False`, in `eval()` and outside the optimizer.
- Backbone: remote FE-00 CNO-only `last.pth`, SHA-256 `499ec748cc5db1b7f3ad24029a464e921ad31c3d383b69626ec540eba392903e`; initialization family `sim_pretrain -> local train PIV`, no `sim_real_ft`.
- Point head: input `402` (`40` CNO future uv + `360` replicate-padded LOCAL3 history + normalized x/y), MLP `402→256→256→256→128→40` GELU, final layer zero-init; CNO pressure channel copied unchanged.
- Fixed protocol: manifest SHA `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`, 50/16/16, B3_PACKED, seed `20260901`, batch `8`, AdamW `lr=1e-4`, `weight_decay=0`, uv MSE only, 1500-step screen and gate-controlled continuation to 7500.
- Screening gate versus the same frozen CNO: Rel-L2 improvement >0%, MVPE improvement >0%, TKE degradation ≤5%. No locked-final, Codabench, LOCAL5, extra features or H2.
- Current state: runner implementation/tests complete; FE-00 checkpoint resolved and remote smoke passed all frozen/zero-init/shape checks; formal detached launch pending.

### `T1-ID-HYBRID-CNO-POINT-H1-S20260902` — COMPLETED / STOP_HYBRID_POINT_H1_EARLY

- The frozen FE-00 CNO plus zero-initialized LOCAL3 Point residual head completed the preregistered 1500-update screen; no 7500 continuation was run.
- Same-frozen-CNO dev v9 errors (Rel-L2 / TKE / MVPE): backbone `0.19082105 / 0.64406884 / 0.14425756`; candidate `0.14110152 / 0.70690584 / 0.10270503`.
- Candidate relative changes (improvement positive, lower error is better): Rel-L2 `+26.056%`, TKE `-9.756%` (error worsened 9.756%), MVPE `+28.804%`. Rel-L2 and MVPE passed; TKE degradation exceeded the protected 5% limit, so the gate failed only on TKE.
- Fixed protocol: manifest SHA `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`; 50/16/16; B3_PACKED; seed `20260901`; batch `8`; AdamW `1e-4`, weight decay `0`; uv MSE-only Point-head training; frozen CNO parameters excluded from optimizer.
- FE-00 checkpoint SHA `499ec748cc5db1b7f3ad24029a464e921ad31c3d383b69626ec540eba392903e`; `last@1500` SHA `fb6735bff296cc53f028b894b74691697f7475f5bbfda9ae8ee0dcd70d1e3bd2`; runner SHA `9b3d3382496208496ddfa9ac382134095400f957fb4390bfc50904ebe7735b15`.
- Locked-final and Codabench were not accessed. H2, LOCAL5, joint training and loss changes are not authorized by this result.
- Handoff: `docs/coordination/CHATGPT_HANDOFF_HYBRID_CNO_POINT_H1.md`; remote evidence remains under `/home/chyfuture/realpde_runs/hybrid_cno_point_h1_s20260902/artifacts/` and small review files under `../artifacts/hybrid_cno_point_h1_s20260902_review/`.

### `T1-ID-HYBRID-CNO-POINT-H1-SCALE-S20260902` — COMPLETED / GO_H1_SCALE_CALIBRATION

- Replay-only residual-scale probe using the completed H1 `last@1500`; no neural-network training, optimizer step, loss change, or checkpoint update.
- Fixed alpha grid `0.0` through `1.0` in increments of `0.1`; alpha selected on train only by TKE degradation `<=5%`, then maximum Rel-L2 improvement with the preregistered tie-breaks. `alpha_star=0.5`.
- Dev v9 errors (same frozen CNO / original H1 alpha=1 / scaled H1 alpha=0.5), Rel-L2 / TKE / MVPE: `0.19082105 / 0.64406884 / 0.14425756`; `0.14110152 / 0.70690584 / 0.10270503`; `0.15642738 / 0.66956341 / 0.11520854`.
- Scaled candidate relative improvements: Rel-L2 `+18.024%`, TKE `-3.958%` (TKE error worsened 3.958%, within the 5% protection line), MVPE `+20.137%`; dev gate passed as `GO_H1_SCALE_CALIBRATION`.
- Fixed protocol: manifest SHA `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`; 50 train / 16 dev / 16 locked-final untouched; B3_PACKED; seed `20260901`; batch `8`; raw velocity; official v9 scorer. Alpha 0/1 replay max absolute difference was `0.0` and pressure was exact.
- CNO checkpoint SHA `499ec748cc5db1b7f3ad24029a464e921ad31c3d383b69626ec540eba392903e`; H1 checkpoint SHA `fb6735bff296cc53f028b894b74691697f7475f5bbfda9ae8ee0dcd70d1e3bd2`; scale runner SHA `6dde2508786836221359094b01a2ccb919dc0a937ca97efe5684bca6b916a5a8`.
- `dev accessed: YES` only after train selection; locked-final and Codabench: NO. H2, joint training, per-channel/per-horizon scaling, loss changes and new experiments remain unauthorized.
- Handoff: `docs/coordination/CHATGPT_HANDOFF_HYBRID_CNO_POINT_H1_SCALE.md`; review evidence remains under `../artifacts/hybrid_cno_point_h1_scale_s20260902_review/` and the remote run path.

### `T1-ID-HYBRID-CNO-POINT-H1-SCALE-STABILITY-S20260903` — COMPLETED / REVIEW_REQUIRED

- Fixed replay-only stability audit of the prior train-selected `alpha=0.5`; no training, optimizer step, alpha sweep, checkpoint update, locked-final access or Codabench.
- Same frozen CNO/H1 checkpoints and manifest; remote saved `dev_replay.npz` was scored in `realpde-pytorch-h5py:0831` with the official v9 scorer. Aggregate replay exactly matched Rel/TKE/MVPE `0.19082105/0.64406884/0.14425756` and `0.15642738/0.66956341/0.11520854`.
- Stability label: `H1_SCALE_STABILITY_SUPPORTIVE`. Trajectory win rates Rel/TKE/MVPE `100.0%/12.5%/100.0%`; medians `16.788%/-21.491%/36.430%`; paired trajectory bootstrap (10,000, seed `20260901`) 95% CIs `[15.953,21.943]` / `[-29.985,-14.339]` / `[35.183,37.478]` percent.
- Joint counts (trajectory-level): Rel+MVPE `16/16`; all three positive `2/16`; Rel>0, MVPE>0, TKE improvement≥-5% `3/16`. The earlier `16/16` was a documentation-generation error that confused the aggregate TKE gate with the trajectory-level gate. Official v9 aggregate TKE is the mean of all scored-window TKE relative-L2 ratios; each trajectory TKE percentage uses that trajectory's own error as denominator, so macro/median percentages need not match the aggregate percentage. Evidence: `../artifacts/hybrid_cno_point_h1_scale_stability_s20260903/`; handoff `docs/coordination/CHATGPT_HANDOFF_HYBRID_CNO_POINT_H1_SCALE_STABILITY.md`.
