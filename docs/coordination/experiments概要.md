# Experiments 优化概要

## 1. 整体思路

现有实验体系由实验注册表（可比较协议与结果）、`coordination/STATUS.md`（当前交接状态）和 handoff（审阅证据）组成。新实验不迁移历史文件，只在这些既有入口增量沉淀事实。

## 2. 当前结论

| 技术方向 | 内容概要 | 关键实验结果 | 状态 | 详细文档 |
|---|---|---|---|---|
| 可比性协议 | 新实验必须登记 baseline family、manifest、起始 checkpoint、scorer、seed、预算、命令和证据。 | 50/16/16 dev-only 选择规则已冻结。 | KEEP | [Registry](../track1_experiment_registry.md) |
| 交接状态 | 当前无执行任务；H1 scale 与 FF-00 均处于 `REVIEW_REQUIRED`。 | 禁止自动启动 H2、joint training、LOCAL5 或 FF-01/02/03。 | REVIEW | [STATUS](STATUS.md) |
| 长期概要 | 只有稳定结论才上升至方向概要；重要方向改变才更新总概要。 | 本次建立最小导航骨架。 | KEEP | [整体概要](../realpde整体优化概要.md) |

## 3. TODO

| 技术方向 | 内容概要 | 优先级 |
|---|---|---|
| 下一次交接 | ChatGPT/Sol review 后返回一个有目标、资源边界、禁止动作和验收条件的 `NEXT_ACTION`。 | P0 |

## 4. 相关文档

- [Track 1 实验注册表](../track1_experiment_registry.md)
- [当前协调状态](STATUS.md)
- [FF-00 handoff](CHATGPT_HANDOFF_FEATURE_FUSION_FF00.md)
