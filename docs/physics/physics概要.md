# Physics 优化概要

## 1. 整体思路

Track 1 的正式预测任务是利用过去 20 帧 PIV 流场预测未来 20 帧 PIV 流场。Sim2Real 主要体现在 CFD 与真实 PIV 数据的训练期利用、预训练、迁移学习和分布差异，而不是正式推理阶段直接进行 CFD→PIV 逐帧映射。

物理理解在当前阶段的作用，是划清正式推理可用信息与不可用信息，并同时保护平均流、波动结构与真实 PIV observation domain。CFD 与 PIV 属于相关但不同的观测/数值域，不能默认“PIV = CFD + Gaussian noise”。

## 2. 当前结论

| 技术方向 | 内容概要 | 关键实验结果 | 状态 | 详细文档 |
|---|---|---|---|---|
| Runtime 物理信息边界 | 正式输入可靠信息是当前窗口 `u/v` 与 tensor shape；`p=0` 是接口兼容占位。 | FF-00 固化了 `[B,20,32,64,3]` 接口与禁止信息清单。CFD/Re/AoA 只允许训练期或离线分析使用。 | KEEP | [FF-00 handoff](../coordination/CHATGPT_HANDOFF_FEATURE_FUSION_FF00.md) |
| 波动结构保护 | Rel-L2/MVPE 改善可伴随 TKE 恶化，不能以点误差替代物理判断。 | H1 缩放 Rel-L2/MVPE 分别改善 18.024%/20.137%，TKE error 恶化 3.958%。 | REVIEW | [H1 scale handoff](../coordination/CHATGPT_HANDOFF_HYBRID_CNO_POINT_H1_SCALE.md) |
| CFD/PIV phase-free domain gap | 相同 Re/AoA 只用于 condition-level 比较，不假设逐帧 phase 对齐。 | normalized spatial spectrum gap 约为邻近 PIV variation 的 `3.0×`，dominant-frequency gap `3.2×`，短 lag ACF gap 约 `2.2–2.4×`。 | **WEAK_SIGNAL / PARKED** | [Domain Gap](../coordination/CHATGPT_HANDOFF_SIM2REAL_DOMAIN_GAP.md) |
| CFD→PIV TKE transfer | CFD/PIV raw fluctuation/TKE amplitude 差异极大，且 spatial spectrum 也存在明显 gap。 | 当前结论仅为“未经 calibration 的 CFD TKE / temporal supervision 存在负迁移风险”，不能升级成“CFD 一定伤害 TKE”。 | **HIGH RISK / STOP RAW TRANSFER** | [Sim2Real summary](../sim2real/sim2real概要.md) |
| Pseudo-PIV | 通过 scale correction、blur/downsampling、noise/filter 等模拟 PIV observation operator。 | 当前没有可靠 calibration evidence 说明该怎样从 CFD 生成 PIV-like dynamics；未被证伪，但不具备继续 sweep 的证据。 | **PARKED** | [Round 2 Decision](../coordination/CHATGPT_HANDOFF_SIM2REAL_ROUND2_DECISION.md) |

## 3. 当前物理判断

1. **CFD 与 PIV 不做逐帧 residual。** 同 Re/AoA 不代表瞬时 phase 对齐，逐帧差值可能主要反映 phase mismatch。
2. **raw amplitude gap 不能直接解释成物理误差。** Mean/TKE norm 的巨大跨域差异还混合了单位、normalization、PIV measurement、filter/smoothing、geometry/mask 等因素。
3. **temporal/spectral shape gap 仍然真实存在。** 即使避开 raw amplitude，CFD/PIV 的 normalized spectrum、dominant frequency、ACF 仍高于邻近 PIV condition variation，因此 raw temporal teacher 不受支持。
4. **CFD 的剩余物理价值主要是 condition coverage。** 18 个 CFD-only conditions 有限补高 Re tail 和局部 Re×AoA 缺口，可作为未来 OOD/robustness 分析资产。

## 4. TODO

| 技术方向 | 内容概要 | 优先级 |
|---|---|---|
| 物理约束 | 仅在明确授权的候选中检验能同时保护 TKE 的 runtime-safe 物理约束。 | P1 |
| CFD / Sim2Real | 当前不继续 temporal transfer / TKE teacher / Pseudo-PIV sweep；只有出现可靠 observation/calibration evidence 时重新打开。 | PARKED |

## 5. 相关文档

- [Sim2Real / CFD 利用概要](../sim2real/sim2real概要.md)
- [CFD / PIV Domain Gap](../coordination/CHATGPT_HANDOFF_SIM2REAL_DOMAIN_GAP.md)
- [CFD OOD-01](../coordination/CHATGPT_HANDOFF_SIM2REAL_OOD01.md)
- [特征工程长期总结](../feature_engineering/README.md)
- [Track 1 实验注册表](../track1_experiment_registry.md)
