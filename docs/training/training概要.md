# Training 优化概要

## 1. 整体思路

训练策略先保证证据可比较：固定 split、seed、预算与 checkpoint 规则，只改变预注册变量；同时严格区分 Clean Offline Research 与 Official Warm-start / Competition。

## 2. 当前结论

| 技术方向 | 内容概要 | 关键实验结果 | 状态 | 详细文档 |
|---|---|---|---|---|
| 双轨证据规则 | `sim_pretrain → local train PIV` 是 clean 主轨；`sim_real_ft` 仅作 competition-oriented 参考，不能反向作为 clean 因果证据。 | 官方 real fine-tuning 覆盖相对本地 holdout 未知。 | KEEP | [Registry](../track1_experiment_registry.md) |
| Clean CNO E0 | 通用 clean CNO E0 baseline 已登记但尚未完成。 | 仅 `PLANNED`，不得写成已有性能。 | REVIEW | [Registry](../track1_experiment_registry.md) |
| 长训练门槛 | 高成本训练前需实现、smoke test、登记 protocol；未经明确授权不访问 locked final/Codabench。 | 当前无 active execution task。 | KEEP | [STATUS](../coordination/STATUS.md) |

## 3. TODO

| 技术方向 | 内容概要 | 优先级 |
|---|---|---|
| 下一轮训练 | 仅执行 ChatGPT/Sol 明确授权、带 baseline family 和资源边界的单一任务。 | P0 |

## 4. 相关文档

- [Track 1 实验注册表](../track1_experiment_registry.md)
- [协调状态](../coordination/STATUS.md)
