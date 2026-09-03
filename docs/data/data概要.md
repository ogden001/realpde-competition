# Data 优化概要

## 1. 整体思路

数据侧的核心是严格 trajectory-level 隔离、明确 Sim/PIV 的可用信息边界，并让所有窗口、归一化、特征拟合只从 train split 获得。

## 2. 当前结论

| 技术方向 | 内容概要 | 关键实验结果 | 状态 | 详细文档 |
|---|---|---|---|---|
| 冻结 ID split | 默认使用 82 条 PIV 的 50 train / 16 dev / 16 locked final，按完整 trajectory 划分。 | Manifest SHA `42b710…c347`。 | KEEP | [Registry](../track1_experiment_registry.md) |
| Runtime-safe features | 不使用 Re、AoA、physical x/y、CFD、HDF5 metadata、body mask、future/target；不从零值推断 mask。 | Feature Discovery 的运行边界已冻结。 | KEEP | [Feature summary](../feature_engineering/README.md) |
| Sim2Real 配对 | 不能仅按 AoA/Re 假设 CFD 与 PIV 唯一配对；应保留条件与配对溯源。 | 项目长期数据规则。 | KEEP | — |

## 3. TODO

| 技术方向 | 内容概要 | 优先级 |
|---|---|---|
| 数据使用审计 | 新实验记录 manifest、窗口协议、train-only 拟合与是否访问 dev/final。 | P0 |

## 4. 相关文档

- [Track 1 实验注册表](../track1_experiment_registry.md)
- [Feature 数据侧复核](../coordination/CHATGPT_HANDOFF_FE_DATA01.md)
