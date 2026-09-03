# Training 优化概要

## 1. 整体思路

训练策略先保证证据可比较：固定 split、seed、预算与 checkpoint 规则，只改变预注册变量；同时严格区分 Clean Offline Research 与 Official Warm-start / Competition。

## 2. 当前结论

| 技术方向 | 内容概要 | 关键实验结果 | 状态 | 详细文档 |
|---|---|---|---|---|
| 双轨证据规则 | `sim_pretrain → local train PIV` 是 clean 主轨；`sim_real_ft` 仅作 competition-oriented 参考，不能反向作为 clean 因果证据。 | 官方 real fine-tuning 覆盖相对本地 holdout 未知。 | KEEP | [Registry](../track1_experiment_registry.md) |
| Clean CNO E0 | 通用 clean CNO E0 baseline 已登记但尚未完成。 | 仅 `PLANNED`，不得写成已有性能。 | REVIEW | [Registry](../track1_experiment_registry.md) |
| P0-A + N2 长程 validation | `OFFICIAL_WARM_START` 50/16 validation 已从 10,300 延伸到 30,900 updates。10.3k 后仍持续改善，约 22k 后进入平台/振荡区，没有出现明确长期过拟合崩坏。 | 最优 Rel-L2 `0.112398@27880`；最优 TKE `0.492848@30340`；最优 MVPE `0.084671@26240`。`26240` 是当前 balanced dev candidate。 | REVIEW | [30.9k handoff](../coordination/CHATGPT_HANDOFF_P0A_N2_VALIDATION_30900.md) |
| 长训练门槛 | 高成本训练前需实现、smoke test、登记 protocol；未经明确授权不访问 locked final/Codabench。 | 当前没有必要自动继续到 40k/50k；优先分析现有 late checkpoints。 | KEEP | [STATUS](../coordination/STATUS.md) |

## 3. TODO

| 技术方向 | 内容概要 | 优先级 |
|---|---|---|
| Late-checkpoint 选择 | 后续若研究提交/SPS，优先比较现有 `26240 / 27880 / 30340` checkpoints；不要仅按单一 dev 指标或自定义 final proxy 选模。 | P0 |
| 下一轮训练 | 仅执行 ChatGPT/Sol 明确授权、带 baseline family 和资源边界的单一任务；当前不自动延长训练预算。 | P1 |

## 4. 相关文档

- [P0-A + N2 30.9k validation handoff](../coordination/CHATGPT_HANDOFF_P0A_N2_VALIDATION_30900.md)
- [Track 1 实验注册表](../track1_experiment_registry.md)
- [协调状态](../coordination/STATUS.md)
