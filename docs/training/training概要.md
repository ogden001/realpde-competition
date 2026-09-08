# Training 优化概要

## 1. 整体思路

训练策略先保证证据可比较：固定 split、seed、预算与 checkpoint 规则，只改变预注册变量；同时严格区分 Clean Offline Research 与 Official Warm-start / Competition。

对所有正式 Future20 时序实验，overall 指标不再视为充分证据。凡结果用于 `KEEP / PROMISING / NO_GO / MERGE_WORTHY` 等科研判断，必须按 [Experiment By-Horizon / 逐帧分析协议](../EXPERIMENT_BY_HORIZON_PROTOCOL.md) 同时保留 overall、trajectory-level 与 `t+1...t+20` 的逐帧证据。逐帧 TKE / MVPE 只作为明确标注的机制拆解，不冒充 official 单帧分数。

CFD / PIV Sim2Real 已完成两轮粗筛。当前不再把“重新做 CFD long pretraining”作为独立训练主线；官方 `sim_pretrain` 仍可作为现有 CLEAN 链路中的合法 initialization，但其 frozen representation 没有显示 PIV forecasting 增量。

## 2. 当前结论

| 技术方向 | 内容概要 | 关键实验结果 | 状态 | 详细文档 |
|---|---|---|---|---|
| 双轨证据规则 | `sim_pretrain → local train PIV` 是 clean 主轨；`sim_real_ft` 仅作 competition-oriented 参考，不能反向作为 clean 因果证据。 | 官方 real fine-tuning 覆盖相对本地 holdout 未知。 | KEEP | [Registry](../track1_experiment_registry.md) |
| Official sim_pretrain provenance | 官方 checkpoint 可确认 architecture、20→20 task 与 5000-iteration checkpoint 记录，但 exact sampling/loss/noise/mask/normalization/optimizer 等不完整。 | 结论 `INSUFFICIENT_PROVENANCE`；RealPDEBench 代码“支持某 augmentation”不能写成 competition checkpoint “已实际使用”。 | **KEEP WITH LIMIT** | [Recipe audit](../coordination/CHATGPT_HANDOFF_SIM2REAL_OFFICIAL_RECIPE.md) |
| CFD representation transfer | 冻结 official `sim_pretrain` CNO，只训练同预算 tiny linear probe，与同架构 random frozen CNO 对照。 | CFD representation：Rel-L2 `0.9026` / TKE `32.0180` / MVPE `1.1509`；random control：`0.7680 / 13.0809 / 0.7800`，三项均更差。 | **STOP / PARKED** | [REP-01](../coordination/CHATGPT_HANDOFF_SIM2REAL_REP01.md) |
| Clean CNO E0 | 通用 clean CNO E0 baseline 已登记但尚未完成。 | 仅 `PLANNED`，不得写成已有性能。 | REVIEW | [Registry](../track1_experiment_registry.md) |
| P0-A + N2 长程 validation | `OFFICIAL_WARM_START` 50/16 validation 已从 10,300 延伸到 30,900 updates。10.3k 后仍持续改善，约 22k 后进入平台/振荡区，没有出现明确长期过拟合崩坏。 | 最优 Rel-L2 `0.112398@27880`；最优 TKE `0.492848@30340`；最优 MVPE `0.084671@26240`。`26240` 是当前 balanced dev candidate。 | REVIEW | [30.9k handoff](../coordination/CHATGPT_HANDOFF_P0A_N2_VALIDATION_30900.md) |
| DW-01 Dense-All temporal supervision | 将 50-train temporal pool 从 canonical fixed stride-20 扩展为所有合法 Past20→Future20 starts；Dev 固定协议不变。该策略只增加真实 PIV 合法监督，不引入 synthetic/domain gap。 | 40,488 windows（vs 2,052 canonical；`19.731x`），30k raw `0.110092 / 0.479399 / 0.080831`。Matched 7.5k 对 RW-00 fixed 三项 aggregate 均改善，frame Rel-L2 `18/20` horizons 更好；7.5k→30k frame Rel-L2 `20/20` 改善。h19 存在 tail spike，且 20k→30k trajectory-level TKE 仅 `2/16` 改善，故 checkpoint 选择仍需保留 robustness 审计。 | **DEFAULT_TRAINING_PROTOCOL / MERGE_CANDIDATE** | [DW-01 review](../experiments/dw01_dense_all_20260907/README_FOR_CHATGPT.md) |
| 长训练门槛 | 高成本训练前需实现、smoke test、登记 protocol；未经明确授权不访问 locked final/Codabench。 | 当前没有必要自动继续到 40k/50k；也不启动 long CFD pretraining、raw CFD/PIV mixed curriculum 或 teacher/student Campaign。 | KEEP | [STATUS](../coordination/STATUS.md) |

### Dense-All 决策规则

Dense-All 已完成 aggregate、trajectory-level 与 Future20 by-horizon 复核，证据已足以支持工程决策。后续 competition-oriented / SOTA 训练默认采用 Dense-All，不再单独消耗 GPU 做“Dense-All 是否仍有效”的重复 A/B。

这里的判断不是把“更多数据必然更好”当成数学定理，而是基于本任务已有直接证据：新增样本全部来自真实 PIV 合法时序窗口，matched control 与逐帧结果均显示稳定正收益。因此下一步研究问题应从“Dense 是否有效”转为“如何把 Dense 的训练预算、checkpoint selection 和已有 SOTA 组件一起组织好”。

## 3. TODO

| 技术方向 | 内容概要 | 优先级 |
|---|---|---|
| Dense-All SOTA merge integration | 下一次进行 SOTA Merge Worthiness Review 时，Dense-All 作为默认 train-window protocol 直接进入候选主线，不再安排额外 canonical-vs-Dense 验证。需要重点设计 Dense epoch / update 预算与 checkpoint selection。 | P0 |
| h19 late-tail 机制 | Future20 末段不是平滑单调 drift，h19 在多个 checkpoint/control 上出现尖峰，优先怀疑 phase/amplitude mismatch 或 temporal event alignment。作为机制方向记录，但暂不抢占主线 merge 资源。 | P1 |
| Late-checkpoint 选择 | 后续若研究提交/SPS，除 aggregate 外必须结合 by-horizon 与 trajectory TKE robustness；不要仅按单一 dev 指标或自定义 final proxy 选模。 | P0 |
| 下一轮训练 | 仅执行 ChatGPT/Sol 明确授权、带 baseline family 和资源边界的单一任务；当前不自动延长训练预算。 | P1 |
| CFD / Sim2Real | 当前状态 `WEAK_SIGNAL / PARKED`。除非出现可靠 CFD→PIV calibration、明确 OOD failure linkage 或新的低风险利用机制，否则不启动新的 CFD 训练 Campaign。 | PARKED |

## 4. 相关文档

- [Experiment By-Horizon / 逐帧分析协议](../EXPERIMENT_BY_HORIZON_PROTOCOL.md)
- [DW-01 Dense-All review](../experiments/dw01_dense_all_20260907/README_FOR_CHATGPT.md)
- [Sim2Real / CFD 利用概要](../sim2real/sim2real概要.md)
- [Official Sim2Real Recipe Audit](../coordination/CHATGPT_HANDOFF_SIM2REAL_OFFICIAL_RECIPE.md)
- [REP-01](../coordination/CHATGPT_HANDOFF_SIM2REAL_REP01.md)
- [Round 2 Decision](../coordination/CHATGPT_HANDOFF_SIM2REAL_ROUND2_DECISION.md)
- [P0-A + N2 30.9k validation handoff](../coordination/CHATGPT_HANDOFF_P0A_N2_VALIDATION_30900.md)
- [Track 1 实验注册表](../track1_experiment_registry.md)
- [协调状态](../coordination/STATUS.md)
