# Sim2Real / CFD 利用概要

## 1. 当前战略结论

Sim2Real / CFD 一级方向已完成两轮 breadth-first 粗筛，当前状态：**`WEAK_SIGNAL / PARKED`**。

当前不再继续投入独立 CFD 训练 Campaign。CFD 的主要保留价值是 **OOD / condition coverage（分布外与工况覆盖）分析资产**；当前没有证据支持把 raw CFD temporal dynamics（原始 CFD 时间动力学）直接作为 PIV forecasting teacher，也没有证据支持 official `sim_pretrain` 的 frozen representation（冻结表征）能为 PIV forecasting 提供直接增量。

这不是“CFD 永远无用”的结论，而是按照 60 分原则：当前最便宜、最自然的 Sim2Real 路线未出现足够大的正信号，因此先 PARK，把研发带宽转向其它一级方向。

## 2. 比赛与研究边界

正式 inference 可靠输入仍只有：

- 当前 Past20 PIV `u/v`；
- tensor shape；
- `p=0` 仅作为接口兼容占位。

CFD、Re、AoA、metadata、HDF5 mask 等只能用于训练期、数据分析、预训练或 OOD 研究，不能成为正式 runtime feature。

CFD / PIV 不能仅按相同 Re/AoA 假设逐帧或唯一 trajectory 配对；phase-aligned residual（逐帧相位对齐残差）不成立。

## 3. Round 1：数据资产与 Domain Gap

### 3.1 数据规模

正式 competition 数据：

- CFD：100 trajectories，每条 1000 frames；
- PIV：82 trajectories；其中 79 条为 868 frames，另外 3 条较短；
- CFD/PIV 共同 condition：82；
- CFD-only condition：18；
- PIV-only condition：0；
- Past20→Future20、stride=20：CFD `4900` windows，PIV `3383` windows，比例约 `1.448×`。

因此 CFD 的独特价值主要不是“数量级更大的 forecast 样本”，而是更长 trajectory 和额外 condition coverage。

### 3.2 Phase-free domain gap

在 82 个共同 Re/AoA condition 上，不做逐帧 phase 对齐：

| Descriptor | CFD↔PIV gap | PIV↔PIV 邻近工况 variation | Gap / reference |
|---|---:|---:|---:|
| Mean-flow norm relative gap | 4.716 | 0.087 | 54.0× |
| TKE-norm relative gap | 189.841 | 0.173 | 1095× |
| Spatial spectrum L1 | 0.109 | 0.036 | 3.0× |
| ACF lag-1 gap | 0.0054 | 0.0023 | 2.4× |
| ACF lag-5 gap | 0.0498 | 0.0225 | 2.2× |
| Dominant-frequency gap | 0.1843 | 0.0576 | 3.2× |

Mean/TKE raw scale gap 首先说明 CFD/PIV 存在明显 amplitude / normalization / observation-operator 差异，不能直接解释成 CFD 物理场“错误很多倍”。但 normalized spatial spectrum 与 temporal statistics 仍有明显跨域差异，因此 **raw temporal transfer（原始时间动力学直接迁移）不受支持**。

TKE 尤其需要谨慎：当前证据支持“存在负迁移风险”，不支持“CFD 一定伤害 TKE”的更强结论。

Round 1 verdict：`REPRESENTATION_ONLY_MORE_PLAUSIBLE` / `CFD_TEMPORAL_TRANSFER_NOT_SUPPORTED`。

## 4. Official recipe provenance

对官方 `sim_pretrain` 可确认：

- competition task 为 20→20；
- CNO/FNO/Transolver architecture 可由 checkpoint / starting kit 确认；
- available checkpoint 记录到 5000 iterations。

但 exact sampling、loss、noise、mask、normalization、optimizer/scheduler/batch 等训练 recipe 为 `PARTIAL / UNKNOWN`。

因此最终结论为：**`INSUFFICIENT_PROVENANCE`**。

`third_party/RealPDEBench` 中“代码支持 Gaussian/Poisson/optical noise、pressure masking 等”不能等价写成“competition `sim_pretrain` 已实际使用这些策略”。

## 5. Round 2：Representation 与 OOD 粗筛

### 5.1 REP-01：CFD Representation Probe

问题：official `sim_pretrain` 学到的 frozen CNO representation，是否比同架构 random frozen representation 更适合 PIV forecasting？

Matched tiny linear probe、100 updates、PIV dev：

| Arm | Rel-L2 ↓ | TKE rel-L2 ↓ | MVPE ↓ |
|---|---:|---:|---:|
| Official CFD representation | 0.9026 | 32.0180 | 1.1509 |
| Random frozen representation | **0.7680** | **13.0809** | **0.7800** |

三项均明显更差，结论：**`CFD_REPRESENTATION_NOT_SUPPORTED`**。

需要严格限定这个结论：它只否定“当前 official `sim_pretrain` + frozen backbone + tiny probe”这条最自然低成本路线；不能外推成“任何重新设计的 CFD self-supervised representation 都不可能有效”。按照 60 分原则，当前没有理由继续为此设计更复杂 pretraining。

Latent pooled mean 的 CFD↔PIV gap 确实比 raw field 更小，但这种 latent alignment 没有转化为 PIV downstream forecasting 增量。因此后续不能把“latent distribution 更接近”本身当作成功标准。

### 5.2 OOD-01：CFD-only Condition Coverage

18 个 CFD-only conditions：

- `15225_{0,5,10,15,20}`；
- `27975_{0,5,10,15,20}`；
- `3750_15`、`17775_0`、`22875_{5,20}`、`24150_5`、`25425_5`、`26700_{5,20}`。

按相同 AoA 的 PIV Re support 判断：

- interpolation：`8/18`；
- edge：`7/18`；
- extrapolation：`3/18`。

这些数据有限补充了高 Re tail 和部分 Re×AoA 缺口，但不是大面积新 flow regime。结论：**`CFD_OOD_COVERAGE_MODERATE`**。

因此 CFD-only conditions **KEEP AS OOD / analysis asset**，但不足以单独支持 long CFD pretraining 或 temporal Sim2Real。

## 6. 当前路线状态

### STOP

- raw CFD Past20→Future20 temporal transfer；
- raw CFD + PIV mixed training / curriculum；
- phase-aligned CFD→PIV residual；
- phase-aligned CFD teacher；
- official CFD frozen representation → PIV forecast；
- 当前阶段的 teacher/student、adversarial alignment、long CFD pretraining。

### PARK

- **Calibrated Pseudo-PIV（经真实差异校准的伪 PIV）**：不是被证伪，而是当前缺少单位、尺度、PIV observation operator、noise/filter 的可靠 calibration evidence。只有出现独立 calibration 证据时才重新打开。
- 重新设计的 CFD self-supervised / auxiliary representation pretraining：理论上未被证伪，但当前信息价值不足，不继续投入。

### KEEP

- 官方 `sim_pretrain` 仍可作为现有 CLEAN 实验链路中的合法 initialization；“停止 Sim2Real 研发”不等于禁止使用已经存在的官方 sim-only checkpoint。
- 18 个 CFD-only conditions 作为 OOD / robustness / coverage 分析资产保留。

## 7. 何时重新打开本方向

只有出现以下任一新证据时，才建议重新打开 Sim2Real：

1. 得到可靠 CFD→PIV observation/calibration model，可实证缩小 spectrum/TKE/temporal gap；
2. 后续 OOD 研究证明当前 PIV-only model 的主要失败集中在 CFD-only coverage 所覆盖的边缘工况；
3. 新 backbone / task formulation 出现明确机制，使 CFD 的低频结构或 condition coverage 可以低风险利用；
4. 官方补充了 `sim_pretrain` 完整 training provenance，暴露出明显未利用的低成本空间。

否则保持 **`WEAK_SIGNAL / PARKED`**。

## 8. 相关证据

- [Round 1 Data Inventory](../coordination/CHATGPT_HANDOFF_SIM2REAL_DATA_INVENTORY.md)
- [Round 1 Domain Gap](../coordination/CHATGPT_HANDOFF_SIM2REAL_DOMAIN_GAP.md)
- [Official Recipe Audit](../coordination/CHATGPT_HANDOFF_SIM2REAL_OFFICIAL_RECIPE.md)
- [Round 1 Decision](../coordination/CHATGPT_HANDOFF_SIM2REAL_ROUND1_DECISION.md)
- [REP-01](../coordination/CHATGPT_HANDOFF_SIM2REAL_REP01.md)
- [OOD-01](../coordination/CHATGPT_HANDOFF_SIM2REAL_OOD01.md)
- [Round 2 Decision](../coordination/CHATGPT_HANDOFF_SIM2REAL_ROUND2_DECISION.md)
