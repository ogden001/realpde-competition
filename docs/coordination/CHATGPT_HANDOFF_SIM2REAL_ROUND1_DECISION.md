# RealPDE Track 1 — CFD / PIV Sim2Real Round-1 Decision

## Round-1 verdict

`WEAK_SIGNAL`

CFD 的高价值部分是 condition coverage、长轨迹和 phase-free representation/OOD；但 raw CFD→PIV temporal transfer 没有得到支持，TKE 还有明显负迁移风险。A/B/C 均未训练、未读取 locked-final/private-test、未提交 Codabench。

## 推荐路线（最多两条）

### 1. `CFD representation / auxiliary pretrain → PIV forecast`

下一实验：30–60 分钟 frozen-representation probe。冻结官方 `sim_pretrain/sim_cno.pth` 的 CNO 主干，在 clean 50-train / 16-dev、同一 20→20 接口上只训练一个很小的线性或零初始化 residual adapter；对 CFD/PIV 仅比较 encoder representation 的均值/协方差与 condition-level separability，再用 PIV dev 的 Rel-L2/TKE/MVPE 做一次 bounded screen。禁止新 backbone、teacher/student、adversarial alignment、locked-final 和长训。

判据：如果 representation probe 在 PIV dev 上相对同预算 PIV-only control 同时没有明显伤害 TKE/MVPE，并且 CFD/PIV latent gap 小于 raw field gap，才允许进入下一轮辅助预训练；否则 CFD 保留为 OOD/coverage 数据。

### 2. `Stop temporal Sim2Real，CFD 仅保留 OOD / representation 用途`

下一实验：30–45 分钟 descriptor/OOD utility check。复用本轮 phase-free 的 82 intersection 与 18 CFD-only condition，检查 CFD descriptor/冻结 encoder representation 是否能解释 PIV train/dev 的 condition-level error 或覆盖缺口；只做 train/dev 的 condition-level 统计和可视化，不训练 forecast，不做 ratio sweep，不读取 final。

这条路线是默认安全落点：即使 probe 没有增益，也能保留 CFD 的 coverage/representation 价值而不污染当前 CNO forecast 主线。

## 本轮明确停止

- 停止未经校准的 `Forecast-aligned CFD pretrain → PIV fine-tune`：raw mean/TKE scale gap 过大，temporal PSD/ACF gap 高于 PIV 邻近 variation。
- 停止当前版本的 `CFD → calibrated Pseudo-PIV → PIV fine-tune`：校准尚未由官方单位、观测算子和 PIV noise/filter 证据支撑；TKE 负迁移风险未解除。只有先完成独立 calibration audit，才可重新打开。
- 停止 `CFD + PIV curriculum`、逐帧 CFD/PIV residual、phase-aligned teacher、teacher/student 和 adversarial domain adaptation。
- 不重复开发或长训 official sim_pretrain baseline；其完整训练 recipe provenance 仍为 `INSUFFICIENT_PROVENANCE`。

## 最终措辞

- CFD 对 PIV mean flow：**有条件的表征/coverage 价值**，不是 raw unit transfer 证据。
- CFD 对 PIV TKE：**当前存在负迁移风险**。
- CFD 对 PIV temporal dynamics：**当前不支持直接迁移**。
- Round-1 不授权下一轮自动实验；完成本 handoff 后停止，等待 ChatGPT / Sol review。
