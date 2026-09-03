# RealPDE Track 1 整体优化概要

> Track 1 技术优化的总导航和当前战况汇总。这里只保留阶段性结论；具体配置、完整指标和证据以各方向概要及其链接的实验记录为准。

- Deadline：**2026-09-28**
- 2026-09-03 ～ 2026-09-18：技术探索阶段
- 2026-09-18 后：模块 Merge、全量训练与 Submission 冲刺

| 优化方向 | 核心问题 | 当前概要结论 | 优先级 | 方向概要 |
|---|---|---|---|---|
| 物理过程理解 | 任务边界与可用物理约束 | 正式 runtime 只能依赖当前 20 帧 `u/v` 和 tensor shape；`p` 为兼容占位，外部工况/几何信息不可作为推理特征。 | P0 | [Physics](physics/physics概要.md) |
| 建模范式与模型结构 | 采用什么预测范式 | CNO 是当前比赛性能锚点；纯 Point/LOCAL3 路线已在既定筛选中停止。H1 缩放保留 Rel-L2/MVPE 收益但 TKE 稳定性不足，仍待 review。 | P0 | [Modeling](modeling/modeling概要.md) |
| 特征工程 | 哪些信息有价值、如何融合 | Feature Discovery 已关闭；Temporal/Spatial 有残差信号，但强 CNO 上的融合/校正仍有 TKE 代价，自动 fusion 训练停止、待 review。 | P1 | [Feature Engineering](feature_engineering/README.md) |
| Loss / 指标 / 离线评估 | 如何对齐比赛目标 | 必须用官方 v9 scorer 同时保护 Rel-L2、TKE、MVPE；不能以自定义 proxy 或单一 Rel-L2 选模。 | P0 | [Loss](loss/loss概要.md) |
| 数据分析与数据建设 | PIV/CFD 如何理解和使用 | 冻结 trajectory-disjoint 50/16/16 ID protocol；训练时仅用 train 拟合，禁止用 metadata 或 target 构造 runtime 特征。 | P0 | [Data](data/data概要.md) |
| 训练策略 | 如何提升最终训练效果 | Clean 与 Official Warm-start/Competition 两条证据线必须隔离；通用 clean CNO E0 baseline 尚属 planned，不得虚构结果。 | P0 | [Training](training/training概要.md) |
| 推理与提交工程 | SPS、速度、打包等 | 公开榜最好结果仍为简单 CNO 包 `75.58455`；starting kit 未公开 final-score 公式，本地 proxy 仅作诊断。 | P1 | [Inference](inference/inference概要.md) |
| 实验体系 | 可比较与可追溯 | 实验注册表、协调状态和 handoff 已构成现有事实记录；新任务先读取方向概要与对应证据。 | P0 | [Experiments](coordination/experiments概要.md) |

当前执行状态以 [coordination/STATUS.md](coordination/STATUS.md) 为准；它是短交接索引，不替代本概要或实验注册表。
