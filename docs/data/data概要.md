# Data 优化概要

## 1. 整体思路

数据侧的核心是严格 trajectory-level 隔离、明确 Sim/PIV 的可用信息边界，并让所有窗口、归一化、特征拟合只从 train split 获得。

同时维护一份与具体实验无关的 Dataset Profile。后续重要实验做 case analysis 时，默认先加载该分布画像，再判断 bad case 属于模型机制问题还是 Train / Dev coverage / distribution-tail 问题。

当前 Dataset Baseline / Split Audit 已完成并冻结，不再继续扩展数据划分分析，除非 manifest、窗口协议或数据定义发生实质变化。

## 2. 当前结论

| 技术方向 | 内容概要 | 关键实验结果 | 状态 | 详细文档 |
|---|---|---|---|---|
| 冻结 ID split | 默认使用 82 条 PIV 的 50 train / 16 dev / 16 locked final，按完整 trajectory 划分。 | Full 82-trajectory input-side audit 结论 `SPLIT_OK`；AoA 三组覆盖一致，Re 仅轻微差异，Dev/Final 均 4/16 `OOD_LIKE`，PCA 无 Dev/Final 独占区域，Final 最大 nearest-Train distance `1.775 < 2`。无需重划分。 | **FROZEN / KEEP** | [Dataset Profile](DATASET_PROFILE.md) |
| Cross-split duplicate | 对 82 条 trajectory 的 Past20 `u/v` 做跨 split exact / near duplicate audit。 | 唯一 exact pair：Train `6300_0.h5` ↔ Final `7575_0.h5`，42 个 Past20 windows 逐元素完全一致；除此之外无 `<0.1` near-duplicate，其他最近距离均 `>=0.573602`。 | **CLOSED** | [Duplicate Audit](DUPLICATE_AUDIT.md) |
| Final 使用边界 | Final 已做一次 input-side、target-blind 分布审计；不因此成为模型选择集。 | Final Future20 未读取，未运行模型、未计算指标。后续默认仍只在 Dev 上选模；若经明确授权做 Final 模型评估，应同时报告 `Final-all16` 与排除 `7575_0` 的 `Final-unique15`。 | **KEEP** | [Dataset Profile](DATASET_PROFILE.md) / [Duplicate Audit](DUPLICATE_AUDIT.md) |
| Runtime-safe features | 正式推理不使用 Re、AoA、physical x/y、CFD、HDF5 metadata、body mask、future/target；不从零值推断 mask。 | Feature Discovery 的运行边界已冻结。Re/AoA 可用于离线数据审计，但不能成为正式 runtime feature。 | KEEP | [Feature summary](../feature_engineering/README.md) |
| Sim2Real 配对 | 不能仅按 AoA/Re 假设 CFD 与 PIV 唯一配对；应保留条件与配对溯源。 | 项目长期数据规则。 | KEEP | — |
| Dataset Profile | 已建立长期数据分布画像，包含 50/16 开发画像与 82 条 input-side split audit。 | `SPLIT_OK`；数据划分审计关闭。后续 case analysis 直接复用，不重复 profiling。 | **FROZEN** | [Dataset Profile](DATASET_PROFILE.md) |

## 3. 后续数据侧规则

当前无独立 Data 执行任务。后续实验遵循：

1. 继续使用冻结的 50 / 16 / 16 manifest，不因本次 audit 重划 split。
2. 日常训练与模型选择仍只使用 Train / Dev；locked-final 不参与 checkpoint 或超参数选择。
3. 重要 bad-case / robustness 分析应关联 `DATASET_PROFILE.md` 中的 trajectory distribution 信息。
4. 如未来明确授权 Final 模型评估，同时报告 `Final-all16` 与 `Final-unique15`，避免唯一 exact duplicate 对独立泛化结论造成误导。
5. 只有 manifest、窗口协议、输入定义或 descriptor 定义发生实质变化时，才刷新 Dataset Profile。

## 4. 相关文档

- [Dataset Profile](DATASET_PROFILE.md)
- [Cross-Split Duplicate Audit](DUPLICATE_AUDIT.md)
- [Data Analysis Skill](SKILL.md)
- [当前数据任务](NEXT_ACTION.md)
- [实验分析 Skill](../experiment_analysis/SKILL.md)
- [Track 1 实验注册表](../track1_experiment_registry.md)
- [Feature 数据侧复核](../coordination/CHATGPT_HANDOFF_FE_DATA01.md)
