# 线上 SOTA 迭代

## 1. 目标

本目录用于 Track 1 的**每日线上 SOTA 冲刺**。

核心目标不是独立研究某一个技术方向，而是利用每天有限的 Codabench 提交机会，把截至当天已经论证清楚、具备正向证据且能够稳定部署的优化项合并到当前最佳候选中，完成训练、SPS 优化、打包和线上提交，持续刷新正式分数。

各技术方向仍在自己的目录中完成研究和验证，例如 Modeling、Loss、Feature Engineering、Training、Inference。本目录只负责**跨方向收口与提交**。

## 2. 基本原则

1. **只合并已经有证据的改动。** 不在 submission candidate 上临时加入未经验证的新结构、新 Loss 或新 Feature。
2. **当前线上最佳结果始终作为锚点。** 新版本必须明确说明相对上一版新增了什么。
3. **每天尽量形成一个可提交版本。** 当天没有足够可靠的新模型改动时，也可以只做 SPS、推理或打包优化。
4. **SPS 是每次提交前的固定步骤。** SPS 优化与模型训练分开处理，不污染模型实验结论。
5. **避免浪费提交次数。** 本地 official scorer、SPS/interval 检查、runtime smoke 和 clean-package smoke 通过后再提交。
6. **线上结果只用于确认整体效果。** 不把 Codabench 当作高频超参数搜索器。

## 3. 每日流程

```text
各方向最新实验结论
        ↓
筛选当天可合并的 KEEP / GO 项
        ↓
确定单一 SOTA candidate recipe
        ↓
必要的小规模 A/B 或收尾实验
        ↓
全量 competition refit / continuation
        ↓
SPS / bounds 优化
        ↓
official smoke + runtime + package check
        ↓
Codabench 提交
        ↓
记录线上结果并更新下一轮基线
```

## 4. 当前线上锚点

截至 2026-09-04：

- 当前最好正式结果：`75.584550`
- package：`submission_cno_tke1200_bounds_rel00.zip`
- 最新 P0-A + N2 full@15,300：`71.153839`
- P0-A + N2 的主要正向信号：线上 TKE `78.355520`
- 当前主要问题：SPS `11.431650`，同时 full-data 训练深度尚未对齐 validation 的 late-training 区间

正式提交历史见：[`../submission_log.md`](../submission_log.md)。

## 5. 当前 SOTA 主线

当前优先路线：

**P0-A features + N2 loss + CNO + full-data continuation + SPS recovery**。

50/16 validation 已显示 P0-A + N2 在 10.3k 后仍持续改善，并在约 22k validation updates 后进入平台/振荡区。由于 full-data windows 更多，full-data update 数不能直接与 validation update 数一一对应，后续 competition refit 应优先按数据遍历量对齐。

当前不作为 SOTA 主线自动合并：

- H1 / Point hybrid：Rel-L2、MVPE 有信号，但 trajectory-level TKE 保护不足；
- 新 Feature Fusion：底层信息有价值，但强 CNO 上稳定 fusion 收益尚未建立；
- 新 Loss：N2 仍是当前经过验证的主力 objective。

## 6. 记录规则

每天完成一次正式提交后，在本文件追加一个简短记录：

```text
日期：
Candidate：
相对上一版新增：
训练 checkpoint：
SPS / bounds：
Codabench：Final / Rel-L2 / TKE / MVPE / Time / SPS
结论：KEEP / ROLLBACK
下一轮主要问题：
```

详细实验数据继续记录在对应优化方向文档和 `track1_experiment_registry.md`，本目录不重复保存大段实验报告。
