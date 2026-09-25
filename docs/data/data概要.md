# Data 优化概要

## 1. 整体思路

数据侧的核心是严格 trajectory-level 隔离、明确 Sim/PIV 的可用信息边界，并让所有窗口、归一化、特征拟合只从 train split 获得。

同时维护一份与具体实验无关的 Dataset Profile。后续重要实验做 case analysis 时，默认先加载该分布画像，再判断 bad case 属于模型机制问题还是 Train / Dev coverage / distribution-tail 问题。

当前 Clean 研发默认数据协议冻结为 **Train51 / Seen-Dev12 / unseen-AoA10 Holdout18**，三者按完整 trajectory 隔离；**不再额外划分 Locked-final**。历史 82 条 PIV 的 50/16/16 split audit 仅作为历史分布与 duplicate 证据保留，不再是当前 Clean 研发的默认 split。CFD / PIV Sim2Real 也已完成两轮粗筛，CFD 当前主要保留为 OOD / condition coverage 分析资产，不作为直接 PIV forecasting 数据主线。

## 2. 当前结论

| 技术方向 | 内容概要 | 关键实验结果 | 状态 | 详细文档 |
|---|---|---|---|---|
| Clean 研发数据协议 | 当前默认使用 **Train51 / Seen-Dev12 / unseen-AoA10 Holdout18**，按完整 trajectory 隔离；不再额外划分 Locked-final。Train51 用于训练；Seen-Dev12 用于高频开发评估、checkpoint 与方案选择；AoA10 Holdout18 只用于未见攻角的低频泛化审计，不参与反复调参。 | Train/Seen-Dev/Holdout trajectory-level disjoint；Train/Seen-Dev 覆盖 AoA 0/5/15/20，Holdout 为未见 AoA=10。 | **FROZEN / KEEP** | [Clean campaign](../clean_baseline_final_campaign/README.md) |
| 历史 50/16/16 split / duplicate audit | 旧 82-trajectory split 的 input-side / duplicate 审计继续作为历史数据资产保留，但不再定义当前 Clean 研发 split。 | 历史唯一 exact pair：Train `6300_0.h5` ↔ old-Final `7575_0.h5`；其余历史审计结论可用于理解数据分布，但不再产生 Locked-final 使用规则。 | **HISTORICAL / CLOSED** | [Dataset Profile](DATASET_PROFILE.md) / [Duplicate Audit](DUPLICATE_AUDIT.md) |
| AoA10 Holdout 使用边界 | Holdout18 用于回答“在训练未见过的 AoA=10 工况上是否仍泛化”。 | 长训过程中不要求每 1000 updates 高频评估；长训结束后，对**事先确定的关键 checkpoint**批量跑 Rel-L2/TKE/MVPE/point 与 by-horizon 审计。允许用它判断泛化是否保持，但不得围绕 Holdout 结果继续细粒度 checkpoint/超参搜索。 | **FROZEN / AUDIT ONLY** | [Clean campaign](../clean_baseline_final_campaign/README.md) |
| Runtime-safe features | 正式推理不使用 Re、AoA、physical x/y、CFD、HDF5 metadata、body mask、future/target；不从零值推断 mask。 | Feature Discovery 的运行边界已冻结。Re/AoA/CFD 可用于训练期或离线分析，但不能成为正式 runtime feature。 | KEEP | [Feature summary](../feature_engineering/README.md) |
| CFD 数据资产 | 正式 competition CFD 为 100 trajectories，PIV 为 82；共同 condition 82，CFD-only 18。 | stride=20 的 Past20→Future20 windows：CFD `4900`、PIV `3383`，仅约 `1.448×`；CFD 的主要新增价值是更长 trajectory 与 condition coverage，而不是数量级更多样本。 | **KEEP AS ASSET** | [Inventory](../coordination/CHATGPT_HANDOFF_SIM2REAL_DATA_INVENTORY.md) |
| CFD-only OOD coverage | 18 个 CFD-only conditions 作为参数空间 / flow-space coverage 资产。 | interpolation `8/18`、edge `7/18`、extrapolation `3/18`；主要补高 Re tail 与局部 Re×AoA 缺口，不是大面积新 regime。 | **MODERATE / KEEP** | [OOD-01](../coordination/CHATGPT_HANDOFF_SIM2REAL_OOD01.md) |
| Sim2Real 配对与 domain gap | 不能仅按 AoA/Re 假设 CFD 与 PIV 唯一或逐帧配对；phase-free 统计显示 raw temporal transfer 风险高。 | normalized spatial spectrum gap 约为 PIV 邻近工况 variation 的 `3.0×`，dominant-frequency gap `3.2×`；TKE raw scale gap 极大，当前只支持“存在负迁移风险”，不应写成 CFD 一定伤害 TKE。 | **WEAK_SIGNAL / PARKED** | [Sim2Real summary](../sim2real/sim2real概要.md) |
| Dataset Profile | 已建立长期数据分布画像；旧 50/16/16 与 82 条 input-side audit 保留为历史画像，当前 Clean 研发协议以 51/12/18 为准。 | 后续 case analysis 直接复用已有画像；除非 manifest / 输入协议发生实质变化，不重复 profiling。 | **FROZEN** | [Dataset Profile](DATASET_PROFILE.md) |

## 3. 后续数据侧规则

当前无独立 Data 执行任务。后续实验遵循：

1. 当前 Clean 研发固定使用 **Train51 / Seen-Dev12 / unseen-AoA10 Holdout18**；不再额外划分 Locked-final。
2. **Train51**：用于训练与优化；所有 train-only 拟合、窗口枚举和训练期统计只能来自 Train。
3. **Seen-Dev12**：允许高频评估，用于训练曲线观察、checkpoint 选择、方案比较和 Stage-B 等研发决策。
4. **AoA10 Holdout18**：只做未见攻角泛化审计。对长训任务，默认在长训完成后对预先确定的关键 checkpoint 批量评估；不要每 1000 updates 高频查看，也不要看完结果后围绕 Holdout 继续做细粒度 checkpoint/超参搜索。
5. **Codabench**：属于外部提交验证，不替代 Seen-Dev / AoA10 的研发角色；是否提交由独立 submission 流程决定。
6. 历史 50/16/16 split、旧 Final 与 duplicate audit 只作为历史分析资产，不再产生当前 Clean 研发的 Locked-final 规则。
7. 重要 bad-case / robustness 分析应关联 `DATASET_PROFILE.md` 中的 trajectory distribution 信息。
8. CFD/PIV 不做逐帧 residual 或 phase pairing；CFD-only 18 conditions 仅作为 OOD / coverage / robustness 分析资产保留。
9. 只有 manifest、窗口协议、输入定义或 descriptor 定义发生实质变化时，才刷新 Dataset Profile。

## 4. 相关文档

- [Dataset Profile](DATASET_PROFILE.md)
- [Cross-Split Duplicate Audit](DUPLICATE_AUDIT.md)
- [Sim2Real / CFD 利用概要](../sim2real/sim2real概要.md)
- [CFD / PIV Data Inventory](../coordination/CHATGPT_HANDOFF_SIM2REAL_DATA_INVENTORY.md)
- [CFD / PIV Domain Gap](../coordination/CHATGPT_HANDOFF_SIM2REAL_DOMAIN_GAP.md)
- [CFD OOD-01](../coordination/CHATGPT_HANDOFF_SIM2REAL_OOD01.md)
- [Data Analysis Skill](SKILL.md)
- [当前数据任务](NEXT_ACTION.md)
- [实验分析 Skill](../experiment_analysis/SKILL.md)
- [Track 1 实验注册表](../track1_experiment_registry.md)
- [Feature 数据侧复核](../coordination/CHATGPT_HANDOFF_FE_DATA01.md)
