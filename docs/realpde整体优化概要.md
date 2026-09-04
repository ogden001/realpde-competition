# RealPDE Track 1 整体优化概要

> 本文不是实验进度表，而是 Track 1 的**战略搜索空间总图**。
>
> 目的有两个：
> 1. 日常研发时，沿当前强方案继续榨分；
> 2. 每隔几天主动跳出当前局部最优，重新检查是否存在尚未充分探索的大方向。
>
> 具体实验事实、配置和指标以各方向概要、`track1_experiment_registry.md` 和 coordination handoff 为准。

- Deadline：**2026-09-28**
- 2026-09-03 ～ 2026-09-18：技术探索阶段
- 2026-09-18 后：模块 Merge、全量训练与 Submission 冲刺

---

## 0. 新一级研究会话的默认阅读顺序

新开 Modeling、Loss、Feature Engineering、Data、Sim2Real、OOD、Training 等一级研究会话时，不应只从本方向旧实验开始，也不应只围绕当前线上 SOTA 做局部微调。

默认先按以下顺序读取 GitHub 当前 `main`：

1. **战略地图**：`docs/realpde整体优化概要.md`
2. **当前线上战况**：`docs/sota迭代/README.md`
3. **最近正式提交**：`docs/submission_log.md`，以及当前 SOTA 主线对应的最新 coordination handoff
4. **本方向长期记忆**：对应方向的 `README.md` / `*概要.md`
5. **必要的实验事实**：`docs/track1_experiment_registry.md`、相关 handoff、`docs/coordination/STATUS.md`

新会话必须同时回答两类问题：

- **Exploration**：是否存在新的任务定义、模型结构、表示、训练或物理方法，可以打开新的性能曲线？
- **Competition relevance**：如果新方向成立，它未来如何与当前线上主线衔接，是替换 backbone、增加 branch、改变 prediction target、改变 objective，还是作为 auxiliary component？

原则是：**不能因为当前 SOTA 是 CNO 就只围绕 CNO 微调，也不能因为在做战略探索就忽略已经获得的线上和离线证据。**

---

## 0.5 技术探索阶段的“60 分原则”

当前阶段采用 **Breadth-first / 粗放式探索优先**。这里的“60 分”不是指指标只做到 60%，而是指：一个一级方向先完成从无到有的验证，优先拿走最容易获得的约 50%～80%潜在收益；一旦证明方向有价值，就先暂停，继续探索其他一级方向，而不是过早把单一细分方向优化到 90%～95%。

### 核心目标

技术探索阶段首先回答：

> **这个一级方向值不值得继续投资？最容易拿到的第一波收益有多大？**

而不是一开始就回答：

> “这个方向的最优结构、最优权重、最优 case 处理和全部机制细节是什么？”

当前最稀缺的资源不仅是 GPU，更是单人协调 ChatGPT、Codex、实验结果和全局方向判断的**研究管理带宽**。因此优先减少方向内的反复交互，把时间用于覆盖更多一级搜索空间。

### 默认执行节奏

- 每天原则上只重点推进 **2 个一级方向**；
- 每个新方向先做约 **1～2 小时级别的粗筛**，必要时可扩到 1.5～3 小时 Campaign；
- 一轮通常选择 **2～4 个代表性方案**，重点比较不同机制，不先做密集超参扫描；
- 先找“大台阶”，例如从没有 Feature 到引入有效 Feature、从普通 Loss 到新 Loss family、从现有预训练到新 pretraining strategy、从单一 backbone 到新 architecture family；
- 在方向尚未证明明显价值前，不优先做小权重扫描、细粒度 feature routing、case-specific 修补、复杂 gradient surgery 或多轮机制雕琢。

### 方向结束条件

粗筛完成后，一个方向优先进入以下三种状态之一：

- **`PROMISING`**：出现明显、有工程意义的正收益。记录当前最简单有效方案，先拿到第一波收益，然后 `PARK`，不自动继续深挖；
- **`WEAK_SIGNAL / PARKED`**：有小幅或冲突性信号，但暂时不足以占用更多研究管理带宽；
- **`NO_GO`**：没有明确收益，立即停止当前实现路线。

只有在主要一级方向已经完成一轮粗筛，或某个方向出现明显“大台阶”收益时，才进入第二阶段深挖，包括更细的 loss 权重、结构消融、case/horizon/spatial mechanism、长训和 full-data scaling。

### 粗筛阶段的数据分析级别

粗放探索并不等于只看一个总分，但默认分析应保持轻量：

1. official raw metrics / matched delta；
2. trajectory-level win count 与是否存在明显 metric 崩坏；
3. 简单 checkpoint / convergence curve；
4. 必要的 provenance 与 clean-control 检查。

只有两类结果默认升级到完整 Error Anatomy：

- **明显强收益**：需要确认收益不是假象，并判断是否值得后续深挖；
- **明显指标冲突**：例如一个核心指标大幅提升、另一个大幅恶化，需要快速判断这个方向是否还有低成本可救空间。

对 `+1%` 左右的小收益、轻微 trade-off、少量 case 差异，不默认进入多轮 trajectory / horizon / spatial / gradient 机制研究，优先记录后 `PARK`。

### 全局优先级原则

> **新的大方向从 0 → 1 的收益，优先于已验证细分方向从 0.8 → 0.9 的精修。**

在 Feature Engineering、Loss、Model Architecture、CFD/Pretraining、Training Strategy、Task Formulation 等主要一级方向尚未完成粗筛前，应避免在任何单一子方向连续投入大量轮次。当前阶段的目标是先画完整“收益地图”，再把第二阶段资源集中到最有价值的 2～3 个方向。

---

## 1. 当前总体判断

当前线上最强工程主线是：**P0-A + N2 + CNO + full-data late continuation + calibrated SPS bounds**。截至 2026-09-04，正式 Codabench SOTA 为 **`76.149726`**：Rel-L2 `93.434384` / TKE `77.588799` / MVPE `92.519561` / Time `86.998134` / SPS `27.545059`。SPS 已从上一版 `11.431650` 恢复，当前更值得继续寻找的是在保持 Rel-L2/MVPE/SPS 的同时提升 TKE 或形成新的结构性收益。

**Task Formulation 已出现一个明确的粗筛正结果：Mean / Fluctuation（MF）在 matched 3000-update CLEAN 对照中三项 raw error 均优于 Direct CNO。** MF@3000 相对 Direct@3000 的 Rel-L2 / TKE / MVPE 分别改善 `6.541% / 1.971% / 14.019%`，trajectory wins 为 `16/16 / 8/16 / 15/16`。因此 MF 第一阶段结论已定为 **`PROMISING_PARKED`**：方向价值已经确认，但按“60 分原则”当前不再继续做 RMS decoupling、spectrum、MF-02 等机制级精修，先转向其它一级方向。

**Sim2Real / CFD 也已完成两轮粗筛，结论为 `WEAK_SIGNAL / PARKED`。** Raw CFD temporal transfer 不受支持；official `sim_pretrain` frozen representation 在 matched tiny-probe 下三项 PIV dev 指标均明显差于 random frozen control，因此停止把当前 official CFD representation 当作 PIV forecasting 主增量来源。18 个 CFD-only conditions 仍保留为 OOD / coverage 分析资产，但其价值为 `MODERATE`，不足以单独支撑 long CFD pretraining 或复杂 Sim2Real Campaign。

数据基线已完成独立审计：对全部 82 条 PIV 做 input-side split audit 后结论为 `SPLIT_OK`，当前 50/16/16 没有明显分布切偏，Final 也不存在清晰 Train coverage gap。因此后续无需因为 split 怀疑而重划数据，研发仍以 50 Train / 16 Dev 为主。唯一长期注意事项是 Train `6300_0.h5` 与 Final `7575_0.h5` 的 Past20 输入完全重复，因此未来若正式评估 locked-final，应同时报告 `Final-all16` 与 `Final-unique15`。

这说明当前工作应同时存在两种模式：

### Exploitation：沿已知强点继续榨分

- P0-A + N2 的 checkpoint / continuation / tail refinement
- TKE 专项 objective
- feature fusion 的稳定实现
- submission 前的 SPS / bounds / runtime / package 优化
- 每日把已经论证清楚的 KEEP/GO 项合成一个 competition candidate

### Exploration：周期性重新打开搜索空间

不能把整个比赛收缩成“继续调 CNO、Loss、Feature”。需要持续检查：

- 任务本身是否应该重构；
- 当前模型是否对 distribution-tail / hard trajectories 足够稳健；
- 时空建模结构是否合理；
- turbulence 是否应该用频谱、模态或 fluctuation 表示；
- 物理约束是否只停留在 TKE；
- 训练范式是否还有自监督、课程学习、多任务等大空间；
- SPS 是否最终需要显式 uncertainty modeling；
- 已 PARKED 的 CFD/Sim2Real 只有出现新的 calibration、OOD failure linkage 或明确机制证据时才重新打开。

---

## 2. 八个一级战略方向

| 一级方向 | 核心问题 | 主要子方向 | 当前状态 | 战略优先级 |
|---|---|---|---|---|
| **1. Task Formulation** | 我们是不是把任务定义得过于简单？ | Direct 20→20、Residual/Delta、Mean+Fluctuation、Multi-horizon、coarse-to-fine | **Mean/Fluctuation 第一阶段已验证 `PROMISING_PARKED`；MF@3000 相对 Direct@3000 三项 raw error 均改善。其它 formulation 尚未系统比较。** | **PARKED after first-pass** |
| **2. Sim2Real / CFD 利用** | CFD 是否能真正帮助 PIV forecasting / OOD？ | raw transfer、Pseudo-PIV、representation、OOD coverage | 两轮粗筛完成：raw temporal transfer 不支持；official frozen CFD representation 不支持；18 个 CFD-only condition 仅有 moderate OOD coverage 价值。 | **WEAK_SIGNAL / PARKED** |
| **3. Generalization / OOD** | 模型是否对当前数据分布尾部与未知工况足够稳健？ | trajectory profile、bad-case linkage、leave-condition-out、edge-condition holdout、稳健模型选择 | 82 条 input-side split audit 已完成且 `SPLIT_OK`；CFD-only 18 conditions 可作为额外 OOD/coverage 分析资产。 | **P1 Exploration** |
| **4. Model Architecture** | 一个 CNO 是否同时承担了太多时空与尺度建模职责？ | CNO/FNO/Transformer family、Local+Global、Spatial+Temporal、Multi-scale、dual-path | CNO 是当前性能锚点；Point/LOCAL3 已停止；结构空间仍未充分打开 | **P1 Exploration** |
| **5. Representation** | Raw/手工特征之外，是否存在更适合流体的表示？ | Raw、runtime-safe physics feature、Frequency/Spectrum、POD/Modal、low/high-frequency decomposition | Feature Discovery 已关闭；official CFD frozen representation 路线 STOP，但与 CFD 无关的 representation 搜索仍开放。 | **P1 Exploration** |
| **6. Physics / Objective** | 如何同时保护平均流、波动结构和局部物理？ | Rel-L2/TKE/MVPE、Mean/Fluctuation、Vorticity/Gradient、spectral energy、multi-task physical heads | N2 已有效；TKE 与 Rel/MVPE trade-off 仍是核心矛盾 | **P0 Exploit + Explore** |
| **7. Training Strategy** | 是否还有比继续加 update 更有效的训练范式？ | Curriculum、noise robustness、self-supervised PIV pretraining、multi-task、LR schedule、SWA/model soup | 30.9k 长训已进入平台振荡区；简单暴力长训优先级下降；long CFD pretraining 当前不做 | **P1** |
| **8. Reliability / Inference** | 如何把物理能力稳定转化为最终分？ | uncertainty、SPS calibration、bounds、pressure handling、runtime、packaging | explicit bounds 已把 SPS 恢复到 `27.545059`；仍需提交前固定 calibration 与 runtime/package 保护 | **P0 Submission** |

---

## 3. 各方向当前证据与下一层问题

### 3.1 Task Formulation

当前线上主线仍是直接：

`past 20 frames → future 20 frames`

但 **Mean + Fluctuation decomposition 已经完成第一阶段公平验证，并从“候选假设”升级为明确正方向。** 该 formulation 显式拆成：

`X(t) = mean(X) + X'(t)`

分别建模平均状态与零均值波动场。

固定 CLEAN 50/16、P0-A、N2、相同 seed / optimizer continuation 的 matched 3000-update 对照：

| Model | Rel-L2 | TKE | MVPE |
|---|---:|---:|---:|
| Direct@3000 | 0.175829 | 0.594649 | 0.151631 |
| MF@3000 | **0.164327** | **0.582928** | **0.130374** |

MF@3000 相对 Direct@3000：

- Rel-L2：`-6.541%`，trajectory wins `16/16`；
- TKE：`-1.971%`，trajectory wins `8/16`；
- MVPE：`-14.019%`，trajectory wins `15/16`。

当前稳定判断：

1. **MF 方向本身有明确价值。** Rel-L2 与 MVPE 是明显的大台阶收益，TKE aggregate 也同向改善。
2. **早期 MF@1500 明显 under-converged。** 后续分析显示继续训练的主要收益来自 Mean reconstruction，Fluctuation 也改善但稳定性较弱。
3. **TKE 是 MF 后续第二阶段的主要薄弱项。** RMS/amplitude objective 曾出现 `15/16` trajectory TKE wins，但会损伤 Mean/MVPE；简单 scorer-aligned TKE、高能区 weighting、conditional/spatial gain 未形成可保留方案。
4. **按“60 分原则”现在停止继续精修。** 当前状态为 **`PROMISING_PARKED`**。不自动继续 RMS decoupling、spectrum、MF-02、temporal head 等实验，先完成其它一级方向的粗筛。

因此 Task Formulation 的当前结论不是“MF 仍待验证”，而是：**MF 已证明值得保留，第一阶段收益已经拿到；未来第二阶段若重新打开，目标是保持 Mean/MVPE 优势的同时继续提高 Fluctuation/TKE。**

其它 formulation 仍可在后续 breadth-first 阶段按需粗筛：

- **Residual / Delta forecasting**：预测相对 persistence、最后一帧、temporal mean 或 coarse predictor 的增量；
- **Multi-horizon**：短期、中期、长期 horizon 分阶段或多头预测；
- **Coarse-to-fine**：先预测大尺度结构，再恢复细尺度 fluctuation。

详细证据见 [Modeling](modeling/modeling概要.md)、[MF Direction Closeout](coordination/CHATGPT_HANDOFF_MF_DIRECTION_CLOSEOUT.md) 和 [Experiment Registry](track1_experiment_registry.md)。

### 3.2 Sim2Real / CFD 利用

正式 runtime 只能依赖当前 20 帧 `u/v` 与 tensor shape；`p` 为兼容占位，Re/AoA/CFD/metadata 等不能成为推理特征。这一边界保持不变。

该一级方向已完成两轮 breadth-first 粗筛，当前正式状态：**`WEAK_SIGNAL / PARKED`**。

#### 数据资产

- CFD：100 trajectories，每条 1000 frames；
- PIV：82 trajectories；
- 共同 condition：82；
- CFD-only condition：18；
- stride=20 的 Past20→Future20 windows：CFD `4900`，PIV `3383`，仅约 `1.448×`。

因此 CFD 的独特价值主要不是“数量级更多 forecast 样本”，而是更长 trajectory 与额外 condition coverage。

#### Domain gap

对 82 个共同 Re/AoA condition 做 phase-free 比较：

- normalized spatial spectrum gap 约为 PIV 邻近工况 variation 的 `3.0×`；
- dominant-frequency gap `3.2×`；
- ACF lag-1/5 gap 约 `2.4× / 2.2×`；
- raw Mean/TKE norm gap 极大，但首先反映 amplitude / normalization / observation-operator 差异，不能直接写成 CFD 物理场错误很多倍。

因此当前支持的结论是：**raw temporal CFD→PIV transfer 不受支持，TKE 存在负迁移风险。** 这不等价于“CFD 一定伤害 TKE”。

#### Representation probe

REP-01 冻结 official `sim_pretrain` CNO 主干，只训练同预算 tiny linear probe：

| Arm | Rel-L2 ↓ | TKE ↓ | MVPE ↓ |
|---|---:|---:|---:|
| Official CFD representation | 0.9026 | 32.0180 | 1.1509 |
| Random frozen representation | **0.7680** | **13.0809** | **0.7800** |

三项均明显更差，因此 **`CFD_REPRESENTATION_NOT_SUPPORTED`**。这只否定 official CFD frozen representation + tiny probe 这一具体低成本路线，不等价于证明所有 CFD representation 方法都不可能有效；但按 60 分原则，当前没有理由继续为此设计复杂 self-supervised / teacher-student / adversarial Campaign。

#### OOD coverage

18 个 CFD-only conditions：interpolation `8/18`、edge `7/18`、extrapolation `3/18`。主要补充高 Re tail 与局部 Re×AoA 缺口，结论 **`CFD_OOD_COVERAGE_MODERATE`**。

当前路线状态：

- **STOP**：raw temporal transfer、raw CFD+PIV mixed curriculum、phase-aligned residual/teacher、official frozen CFD representation、long CFD pretraining、teacher/student、adversarial alignment；
- **PARK**：Calibrated Pseudo-PIV 与重新设计的 CFD self-supervised representation，除非出现可靠 calibration / OOD failure linkage / 新机制证据；
- **KEEP**：官方 `sim_pretrain` 仍可作为现有 CLEAN 链路中的合法 initialization；18 个 CFD-only conditions 保留为 OOD / coverage / robustness 分析资产。

详细长期记忆见 [Sim2Real / CFD 利用概要](sim2real/sim2real概要.md)。

### 3.3 Generalization / OOD

现有冻结 protocol：82 条 PIV，50 train / 16 dev / 16 locked-final，按完整 trajectory 隔离。

2026-09-04 已完成一次全 82 条的 **input-side、target-blind Split Audit**：

- 结论：`SPLIT_OK`，无需重新划分；
- AoA 在 Dev / Final 的覆盖完全一致，Re 仅有轻微分布差异；
- Dev 与 Final 均有 `4/16 OOD_LIKE`，tail-case 比例一致；
- PCA 中 Train / Dev / Final 相互交错，没有 Dev / Final 独占的联合分布区域；
- Final 最大 nearest-Train descriptor distance 为 `1.775`，低于 distance-2 boundary，没有明显 Train coverage gap；
- cross-split duplicate audit 仅发现一组 exact pair：Train `6300_0.h5` ↔ Final `7575_0.h5`，42 个 Past20 `u/v` windows 完全一致；除此之外无 `<0.1` near-duplicate。

因此 Generalization/OOD 的核心问题已经从：

> “50/16/16 是否切歪？”

转成：

> “哪些 distribution-tail / hard trajectories 会让当前模型的 Rel-L2、TKE、MVPE 失效，以及什么机制能提升这些 case 的 robustness？”

后续优先做 **Dataset Profile × model bad-case × metric/horizon behavior** 的关联分析。Leave-condition-out / edge-condition holdout 仍可以作为战略 robustness evaluator，但不再以修复当前 split 为目的，也不应自动替代日常 50/16 protocol。

CFD-only 的 18 个 conditions 可以作为额外 OOD/coverage 分析资产，但 OOD-01 只给出 `MODERATE` 价值，不应自动升级成 CFD training 数据主线。

日常训练和 checkpoint 选择仍只基于 Train / Dev。Locked-final 已完成 input-side audit，但 Future20 / 模型指标仍保持非日常选模信息；如果未来明确授权做 Final 模型评估，应同时报告 `Final-all16` 与 `Final-unique15`，避免 exact duplicate 夸大独立泛化证据。

详细事实见 [Dataset Profile](data/DATASET_PROFILE.md)、[Duplicate Audit](data/DUPLICATE_AUDIT.md) 与 [CFD OOD-01](coordination/CHATGPT_HANDOFF_SIM2REAL_OOD01.md)。

### 3.4 Model Architecture

当前证据：

- CNO 仍是比赛性能锚点；
- 纯 Point MLP / LOCAL3 已停止；
- H1 作为完整 residual correction 的 TKE 稳定性不足，但存在稳定 Rel/MVPE 信号；
- 历史 UNet 线上 hidden physical score 不如 CNO。

这只说明这些具体实现的结果，不代表架构搜索空间关闭。

更值得研究的结构问题：

1. **Local + Global**：局部 gradient/shear/vortex branch + 全局 neural operator branch。
2. **Spatial + Temporal 解耦**：空间 encoder/operator 与 temporal model 分开承担职责。
3. **Multi-scale**：coarse mean structure 与 fine turbulent structure 分尺度建模。
4. **Dual-path Mean/Fluctuation**：与任务 formulation 联动，而不是单纯换 backbone。

短期不建议做无目标的大规模 architecture sweep。

### 3.5 Representation

Feature Discovery 已关闭，意味着“不继续堆新的手工 feature catalog”，不意味着 representation 研究结束。

Sim2Real REP-01 只说明：**official CFD `sim_pretrain` frozen representation 不能作为当前 PIV forecasting 的主增量来源。** 它不关闭其它 representation 搜索。

当前重点应从 feature 数量转向表示方式：

- **Frequency/Spectrum representation**：时域/空域频谱、low/high-frequency decomposition；
- **POD / Modal representation**：把流场分解为主要空间模态及其时间系数；
- **Physical + Spectral dual representation**：同时在物理空间和频域约束/预测；
- **PIV self-supervised / learned latent representation**：优先从真实 PIV 自身寻找更可迁移的 learned representation，而不是默认依赖 CFD alignment。

P0-A 已证明 runtime-safe Temporal/Spatial/统计信息有价值，下一步更应研究“网络如何使用信息”，而不是继续发明 feature 21/22/23。

### 3.6 Physics / Objective

当前 N2：

`MSE + 0.05*TKE + 0.027514*Relative-L2 + 0.009757*MVPE`

已经在线下和 Codabench 显示价值，但后续不应只做权重扫描。

更大的方向：

1. **Fluctuation-aware objective**：直接约束 `u'=u-mean(u)`、`v'=v-mean(v)`。
2. **Mean-flow consistency**：显式保护 temporal mean，避免为了 TKE 牺牲整体流场。
3. **Vorticity / gradient / shear objective**：保护局部流动结构。
4. **Spectral energy objective**：防止 pixel error 看似下降但高频 turbulent energy 被抹平。
5. **Multi-task physical heads**：forecasting 为主任务，同时预测 mean/TKE/vorticity/modal coefficient 等辅助目标。

这一方向是当前最可能直接改善 TKE 上限的主线。

### 3.7 Training Strategy

P0-A + N2 的固定 50/16 validation 已扩展到 30,900 updates：

- 10.3k 后仍有收益；
- 约 22k 后进入平台/振荡区；
- balanced dev candidate：`26240`；
- Rel-L2 best：`27880`；
- TKE best：`30340`；
- MVPE best：`26240`。

因此继续原 LR 暴力拉长不再是主要 exploration 方案。全量 competition continuation 到 `43260` 已用于当前线上 SOTA，但这属于 submission exploitation，不改变 validation 长训已进入平台区的判断。

后续训练策略分两类：

**低成本 exploitation**：

- checkpoint averaging / SWA / model soup；
- low-LR tail refinement；
- 更合理的 checkpoint selection。

**新范式 exploration**：

- short→long horizon curriculum；
- coarse→fine / mean→fluctuation curriculum；
- noise robustness training；
- self-supervised PIV pretraining；
- multi-task learning。

当前不再启动 long CFD pretraining、raw CFD/PIV mixed curriculum 或复杂 Sim2Real training Campaign。

详细长训证据见 [Training](training/training概要.md) 与 [30.9k handoff](coordination/CHATGPT_HANDOFF_P0A_N2_VALIDATION_30900.md)。

### 3.8 Reliability / Inference / Submission

当前最好正式结果：**`76.149726`**。

Candidate：`P0-A + N2 + CNO + full@43260 + explicit SPS bounds`

- Rel-L2 `93.434384`
- TKE `77.588799`
- MVPE `92.519561`
- Time `86.998134`
- SPS `27.545059`
- Final `76.149726`

显式 bounds：`half_width = 0.0075 + 0.02 * abs(prediction)`，已把上一版 full@15300 的 SPS `11.431650` 恢复到 `27.545059`。

当前提交策略：

- **SPS 不单独消耗提交机会**；
- 当天模型优化完成后，在 submission 前一起完成 SPS/bounds/pressure/runtime/package 校准；
- 再进行一次正式 Codabench 提交。

长期看，SPS 方向不应只理解成固定 bounds 调参，还可以研究：

- prediction uncertainty；
- heteroscedastic interval；
- quantile / calibrated interval；
- small ensemble / uncertainty proxy。

短期先做 submission-level calibration，长期再决定是否需要 learned uncertainty。

---

## 4. 当前最值得提高权重的战略空白

| 战略空白 | 为什么重要 | 当前动作 |
|---|---|---|
| **Distribution-tail / bad-case robustness** | Split audit 已确认数据划分本身无明显偏置，下一步应解释为什么某些 tail trajectories 仍会造成 TKE/Rel/MVPE 失败 | 联合 `Dataset Profile × trajectory metrics × horizon behavior`，定位模型机制问题；18 个 CFD-only conditions 仅作为辅助 OOD/coverage asset |
| **Sim2Real / CFD 数据利用** | 两轮粗筛已完成，没有发现值得继续占用主研发带宽的大台阶 | **`WEAK_SIGNAL / PARKED`**；停止 raw transfer / official frozen representation / long CFD training；Pseudo-PIV 等仅在新 calibration 证据出现时重开 |
| **Mean / Fluctuation 重构** | 第一阶段已确认是明确正方向：MF@3000 相对 Direct@3000 在 Rel-L2 / TKE / MVPE 全部改善，尤其 Mean/MVPE 收益明显 | **`PROMISING_PARKED`**；保留为第二阶段重点候选，当前不继续机制级精修 |
| **Spectral / fluctuation dynamics** | MF 的 TKE 收益小于 Rel/MVPE，波动场仍是未来可能的上限问题 | 当前只保留研究钩子，先完成其它一级方向粗筛后再决定是否进入第二阶段 |
| **Local+Global / Spatial+Temporal 架构** | 当前单 CNO 同时承担空间、时间、尺度和 turbulence 表示 | 只做有明确假设的 bounded probe，不做盲目 sweep |
| **Uncertainty → SPS** | 固定 bounds 已恢复 SPS，但可能只是低阶解法 | 提交前先 calibration；后续看收益决定是否模型化 |

---

## 5. 当前战术执行层

战略地图负责“不要漏方向”；战术层负责“今天做什么”。两者不要混在一起。

### 当前 Exploitation 主线

1. P0-A + N2 现有 checkpoint / averaging / tail refinement；
2. TKE / fluctuation-aware objective；
3. 已确认 Temporal/Spatial 信息的 TKE-preserving fusion；
4. 当日最佳 candidate 的 full-data competition refit；
5. submission 前 SPS + smoke + package；
6. Codabench 正式提交。

### 当前暂停或低优先级

- 继续扩 Feature catalog；
- 纯 Point / LOCAL3 / LOCAL5；
- 回到旧 UNet 路线；
- **Mean/Fluctuation 方向的 RMS decoupling、spectrum、MF-02、temporal head 等第二阶段精修，待其它一级方向完成粗筛后再回收；**
- **Sim2Real / CFD 独立研发 Campaign：raw temporal transfer、raw CFD+PIV mixed curriculum、official frozen CFD representation、teacher/student、adversarial alignment、long CFD pretraining；**
- **Calibrated Pseudo-PIV 当前 PARK，只有出现可靠 observation/calibration evidence 才重新打开；**
- 无假设的大规模 architecture sweep；
- 原 LR 继续无限长训；
- 只扫 N2 各项 scalar 权重；
- 在主要一级方向尚未完成粗筛前，对单一细分方向连续多轮做机制级精修；
- 重新设计 50/16/16 split 或重复 Dataset Profile；
- 以自定义 final proxy 代替官方 scorer / Codabench。

### 当前线上 SOTA 迭代

每日 competition candidate 的汇总、全量训练、SPS、打包和正式提交，统一记录在 [SOTA 迭代](sota迭代/README.md)。

---

## 6. 周期性全局复盘规则

建议每完成 2～3 个主要实验，或每 2～3 天，重新打开本文件一次，不直接从 `NEXT_ACTION` 出发。

复盘时固定问四个问题：

1. **当前最优方案只是 ID dev 更强，还是更可能泛化？**
2. **有没有一个一级方向连续多天没有被重新审视？**
3. **最近的失败是“方向无效”，还是“当前实现方式无效”？**
4. **下一轮算力应该用于 Exploitation，还是应该买一次 Exploration 的信息？**

只有回答完这四个问题，再确定新的 GPU 任务。

---

## 7. 方向文档索引

- [Physics](physics/physics概要.md)
- [Modeling](modeling/modeling概要.md)
- [Feature Engineering](feature_engineering/feature_engineering概要.md)
- [Loss / Metrics](loss/loss概要.md)
- [Data](data/data概要.md)
- [Training](training/training概要.md)
- [Sim2Real / CFD](sim2real/sim2real概要.md)
- [Inference / Submission](inference/inference概要.md)
- [SOTA 迭代](sota迭代/README.md)
- [Experiment Registry](track1_experiment_registry.md)
- [Coordination Status](coordination/STATUS.md)

当前执行状态以 `docs/coordination/STATUS.md` 为准；本文件负责战略搜索空间，不替代实验注册表和具体 handoff。