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

## 1. 当前总体判断

当前最强工程主线已经比较清楚：**CNO + P0-A runtime-safe features + N2 objective**。它在固定 50/16 validation 上同时改善 Rel-L2、TKE、MVPE，并在 Codabench 上把 TKE 提升到 `78.355520`；但最终分仍受 SPS 明显拖累。

数据基线也已经完成一次独立审计：对全部 82 条 PIV 做 input-side split audit 后结论为 `SPLIT_OK`，当前 50/16/16 没有明显分布切偏，Final 也不存在清晰 Train coverage gap。因此后续无需因为 split 怀疑而重划数据，研发仍以 50 Train / 16 Dev 为主。唯一长期注意事项是 Train `6300_0.h5` 与 Final `7575_0.h5` 的 Past20 输入完全重复，因此未来若正式评估 locked-final，应同时报告 `Final-all16` 与 `Final-unique15`。

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
- CFD 是否被充分利用；
- 当前模型是否对 distribution-tail / hard trajectories 足够稳健；
- 时空建模结构是否合理；
- turbulence 是否应该用频谱、模态或 fluctuation 表示；
- 物理约束是否只停留在 TKE；
- 训练范式是否还有自监督、课程学习、多任务等大空间；
- SPS 是否最终需要显式 uncertainty modeling。

---

## 2. 八个一级战略方向

| 一级方向 | 核心问题 | 主要子方向 | 当前状态 | 战略优先级 |
|---|---|---|---|---|
| **1. Task Formulation** | 我们是不是把任务定义得过于简单？ | Direct 20→20、Residual/Delta、Mean+Fluctuation、Multi-horizon、coarse-to-fine | Direct 20→20 为当前主线；其它 formulation 尚未系统比较 | **P0 Exploration** |
| **2. Sim2Real / CFD 利用** | 训练阶段是否真正利用了赛题最独特的 CFD+PIV 资产？ | CFD pretraining objective、CFD→PIV degradation/domain randomization、latent alignment、teacher/student | 目前主要依赖官方 warm-start / pretrain，系统性 Sim2Real 设计不足 | **P0 Exploration** |
| **3. Generalization / OOD** | 模型是否对当前数据分布尾部与未知工况足够稳健？ | trajectory profile、bad-case linkage、leave-condition-out、edge-condition holdout、稳健模型选择 | 82 条 input-side split audit 已完成且 `SPLIT_OK`；下一步重点从“怀疑 split”转向模型 robustness | **P1 Exploration** |
| **4. Model Architecture** | 一个 CNO 是否同时承担了太多时空与尺度建模职责？ | CNO/FNO/Transformer family、Local+Global、Spatial+Temporal、Multi-scale、dual-path | CNO 是当前性能锚点；Point/LOCAL3 已停止；结构空间仍未充分打开 | **P1 Exploration** |
| **5. Representation** | Raw/手工特征之外，是否存在更适合流体的表示？ | Raw、runtime-safe physics feature、Frequency/Spectrum、POD/Modal、low/high-frequency decomposition | Feature Discovery 已关闭，但“表示学习”远未关闭 | **P1 Exploration** |
| **6. Physics / Objective** | 如何同时保护平均流、波动结构和局部物理？ | Rel-L2/TKE/MVPE、Mean/Fluctuation、Vorticity/Gradient、spectral energy、multi-task physical heads | N2 已有效；TKE 与 Rel/MVPE trade-off 仍是核心矛盾 | **P0 Exploit + Explore** |
| **7. Training Strategy** | 是否还有比继续加 update 更有效的训练范式？ | Curriculum、noise robustness、self-supervised PIV pretraining、multi-task、LR schedule、SWA/model soup | 30.9k 长训已进入平台振荡区；简单暴力长训优先级下降 | **P1** |
| **8. Reliability / Inference** | 如何把物理能力稳定转化为最终分？ | uncertainty、SPS calibration、bounds、pressure handling、runtime、packaging | SPS 是当前线上最大显性短板；提交日必须处理 | **P0 Submission** |

---

## 3. 各方向当前证据与下一层问题

### 3.1 Task Formulation

当前主线是直接：

`past 20 frames → future 20 frames`

这一路线简单、稳定，但它默认让一个网络同时学习平均流、瞬态波动和长短期时间演化。

后续值得重新打开的 formulation：

1. **Residual / Delta forecasting**：预测相对 persistence、最后一帧、temporal mean 或 coarse predictor 的增量。
2. **Mean + Fluctuation decomposition**：显式拆成 `X(t)=mean(X)+X'(t)`，分别建模 mean flow 与 turbulent fluctuation。
3. **Multi-horizon**：短期、中期、长期 horizon 分阶段或多头预测。
4. **Coarse-to-fine**：先预测大尺度结构，再恢复细尺度 fluctuation。

其中 **Mean + Fluctuation** 当前价值最高，因为 H1、Temporal/Spatial correction 多次出现“Rel-L2/MVPE 改善但 TKE 受损”的共同模式。

### 3.2 Sim2Real / CFD 利用

正式 runtime 只能依赖当前 20 帧 `u/v` 与 tensor shape；`p` 为兼容占位，Re/AoA/CFD/metadata 等不能成为推理特征。这一边界保持不变。

但 runtime 不使用 CFD，不等于训练阶段只需要官方预训练 checkpoint。

值得系统探索：

1. **CFD forecasting pretraining**：预训练目标直接与最终 20→20 forecasting 对齐。
2. **CFD→PIV degradation/domain randomization**：人为加入 measurement noise、spatial smoothing、amplitude perturbation、missing/noisy structure，让 simulation 更接近真实 PIV。
3. **CFD/PIV latent alignment**：让 encoder 学到跨域共享的 flow representation。
4. **CFD teacher → PIV student**：利用 CFD 的完整性做 representation / dynamics teacher，而不是 inference 输入。

这是目前整体路线里最明显的战略空白之一。

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

日常训练和 checkpoint 选择仍只基于 Train / Dev。Locked-final 已完成 input-side audit，但 Future20 / 模型指标仍保持非日常选模信息；如果未来明确授权做 Final 模型评估，应同时报告 `Final-all16` 与 `Final-unique15`，避免 exact duplicate 夸大独立泛化证据。

详细事实见 [Dataset Profile](data/DATASET_PROFILE.md) 与 [Duplicate Audit](data/DUPLICATE_AUDIT.md)。

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

当前重点应从 feature 数量转向表示方式：

- **Frequency/Spectrum representation**：时域/空域频谱、low/high-frequency decomposition；
- **POD / Modal representation**：把流场分解为主要空间模态及其时间系数；
- **Physical + Spectral dual representation**：同时在物理空间和频域约束/预测；
- **Learned latent representation**：与 CFD/PIV alignment、自监督训练结合。

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

因此继续原 LR 暴力拉长到 40k/50k 不再是优先方案。

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

详细长训证据见 [Training](training/training概要.md) 与 [30.9k handoff](coordination/CHATGPT_HANDOFF_P0A_N2_VALIDATION_30900.md)。

### 3.8 Reliability / Inference / Submission

当前最好正式结果仍是简单 CNO 包：`75.58455`。

P0-A + N2 full 15,300 updates：

- Rel-L2 `93.023539`
- TKE `78.355520`
- MVPE `91.894417`
- Time `88.430528`
- SPS `11.431650`
- Final `71.153839`

说明物理能力提升没有自动转化成更高 final score。

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
| **Distribution-tail / bad-case robustness** | Split audit 已确认数据划分本身无明显偏置，下一步应解释为什么某些 tail trajectories 仍会造成 TKE/Rel/MVPE 失败 | 联合 `Dataset Profile × trajectory metrics × horizon behavior`，定位模型机制问题；不重划 split |
| **Sim2Real / CFD 数据利用** | 赛题独有的数据资产，目前主要只通过官方 checkpoint 间接使用 | 系统梳理 CFD pretrain / degradation / alignment 可行性 |
| **Mean / Fluctuation / Spectral 重构** | 多个实验反复出现 Rel/MVPE 与 TKE trade-off | 作为下一代 task/objective 的重点候选 |
| **Local+Global / Spatial+Temporal 架构** | 当前单 CNO 同时承担空间、时间、尺度和 turbulence 表示 | 只做有明确假设的 bounded probe，不做盲目 sweep |
| **Uncertainty → SPS** | 固定 bounds 可能只是 SPS 的低阶解法 | 提交前先 calibration；后续看收益决定是否模型化 |

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
- 无假设的大规模 architecture sweep；
- 原 LR 继续无限长训；
- 只扫 N2 各项 scalar 权重；
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
- [Inference / Submission](inference/inference概要.md)
- [SOTA 迭代](sota迭代/README.md)
- [Experiment Registry](track1_experiment_registry.md)
- [Coordination Status](coordination/STATUS.md)

当前执行状态以 `docs/coordination/STATUS.md` 为准；本文件负责战略搜索空间，不替代实验注册表和具体 handoff。
