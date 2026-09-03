# RealPDE Track 1 整体优化概要

> Track 1 技术优化的总导航和当前战况汇总。这里只保留阶段性结论；具体配置、完整指标和证据以各方向概要及其链接的实验记录为准。

- Deadline：**2026-09-28**
- 2026-09-03 ～ 2026-09-18：技术探索阶段
- 2026-09-18 后：模块 Merge、全量训练与 Submission 冲刺

| 优化方向 | 核心问题 | 当前概要结论 | 优先级 | 方向概要 |
|---|---|---|---|---|
| 线上 SOTA 迭代 | 如何利用每日提交机会持续刷新正式分数 | 每天汇总截至当天已经论证清楚的 KEEP/GO 优化项，形成单一 competition candidate；完成全量训练、SPS、smoke、打包和正式提交。当前主线为 P0-A + N2 full continuation + SPS recovery。 | P0 | [SOTA 迭代](sota迭代/README.md) |
| 物理过程理解 | 任务边界与可用物理约束 | 正式 runtime 只能依赖当前 20 帧 `u/v` 和 tensor shape；`p` 为兼容占位，外部工况/几何信息不可作为推理特征。 | P2 | [Physics](physics/physics概要.md) |
| 建模范式与模型结构 | 采用什么预测范式 | CNO 是当前比赛性能锚点；纯 Point/LOCAL3 路线已停止。H1 保留稳定 Rel-L2/MVPE 信号，但 trajectory-level TKE 保护不足，暂不自动扩展。 | P0 | [Modeling](modeling/modeling概要.md) |
| 特征工程 | 哪些信息有价值、如何融合 | Feature Discovery 已关闭；P0-A 已证明 runtime-safe Temporal/Spatial/统计特征能改善主模型，但后续重点应转向 fusion 与 TKE 保护，而不是继续扩展特征目录。 | P1 | [Feature Engineering](feature_engineering/feature_engineering概要.md) |
| Loss / 指标 / 离线评估 | 如何对齐比赛目标 | 必须用官方 v9 scorer 同时保护 Rel-L2、TKE、MVPE；不能以自定义 proxy 或单一 Rel-L2 选模。P0-A + N2 已在线下和 Codabench 显示明显 TKE 收益。 | P0 | [Loss](loss/loss概要.md) |
| 数据分析与数据建设 | PIV/CFD 如何理解和使用 | 冻结 trajectory-disjoint 50/16/16 ID protocol；训练时仅用 train 拟合，禁止用 metadata 或 target 构造 runtime 特征。 | P1 | [Data](data/data概要.md) |
| 训练策略 | 如何提升最终训练效果 | P0-A + N2 的 50/16 validation 已扩展到 30,900 updates；10.3k 后仍改善，约 22k 后进入平台/振荡区。`26240` 是当前 balanced dev checkpoint candidate，不自动等同于最佳提交点。 | P1 | [Training](training/training概要.md) |
| 推理与提交工程 | SPS、速度、打包等 | 公开榜最好结果仍为简单 CNO 包 `75.58455`；P0-A + N2 full 的 TKE 提升到 `78.355520`，但 SPS 降到 `11.431650`，SPS recovery 是当前最高价值排查项。 | P0 | [Inference](inference/inference概要.md) |
| 实验体系 | 可比较与可追溯 | 实验注册表、协调状态和 handoff 构成现有事实记录；新任务先读取方向概要与对应证据。 | Support | [Experiments](coordination/experiments概要.md) |

当前执行状态以 [coordination/STATUS.md](coordination/STATUS.md) 为准；它是短交接索引，不替代本概要或实验注册表。
