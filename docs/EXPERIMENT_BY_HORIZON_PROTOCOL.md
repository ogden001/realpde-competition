# Experiment By-Horizon / 逐帧分析协议

本文规定 Track 1 所有正式 Future-K 时序预测实验的逐帧分析要求。目标是避免只看 Future20 汇总指标而忽略误差随预测时距的结构变化。

## 1. 适用范围

凡正式实验输出多步未来序列，例如 Past20 → Future20，进入 ChatGPT / Sol 结果复核前都必须提供 By-Horizon / 逐帧分析。

这项分析属于正式实验的最小证据，不再是可选的 Level-1 深挖项。短 smoke / implementation parity test 可以豁免，但一旦结果用于 `KEEP / PROMISING / NO_GO / MERGE_WORTHY` 等科研判断，就必须补齐。

默认使用 Dev；不得为了逐帧分析访问 locked-final。

## 2. Future20 的强制输出

对 `h = 1..20`，至少计算以下诊断量。

### 2.1 Frame Rel-L2

对每个 Dev window 的第 h 个未来帧，仅使用 measured velocity channels `u/v`：

`||pred_h - target_h||_2 / max(||target_h||_2, eps)`

先得到 window-level 值，再做 aggregate 与 trajectory-level 汇总。

### 2.2 Frame RMSE

对第 h 帧的 `u/v` 场计算 RMSE，用于观察绝对误差随 horizon 的增长速度。

### 2.3 TKE by-horizon decomposition

这不是“官方单帧 TKE 分数”，而是 official Future20 TKE 的逐 horizon 机制拆解。

对每个 window，先分别用完整 Future20 计算 prediction / target 的 temporal mean：

`mean_pred = mean_t(pred_t)`

`mean_target = mean_t(target_t)`

再定义第 h 帧相对完整 Future20 temporal mean 的 fluctuation energy contribution：

`E_pred_h = 0.5 * sum_uv((pred_h - mean_pred)^2)`

`E_target_h = 0.5 * sum_uv((target_h - mean_target)^2)`

至少输出：

- `tke_contrib_rel_l2`：第 h 帧 energy map 的相对 L2 误差；
- `tke_contrib_ratio`：prediction energy / target energy。

必须在 README 中明确标记为 diagnostic decomposition，不得写成 official per-frame TKE score。

### 2.4 MVPE probe by-horizon

沿用 official v9 MVPE 的 probe 空间位置 / geometry，但取消 Future20 temporal averaging，逐帧计算 probe velocity relative error。

该指标用于判断 probe-region velocity error 是否随 horizon 加速恶化。

必须标记为 diagnostic probe metric，不得写成 official per-frame MVPE score。

## 3. 聚合层级

每个正式实验至少保留两张表：

1. `by_horizon.csv`
   - 20 行，对应 `t+1 ... t+20`；
   - aggregate Frame Rel-L2 / RMSE / TKE decomposition / MVPE-probe diagnostic；
   - paired A/B 时同时给 baseline、candidate、delta / improvement。

2. `by_trajectory_horizon.csv`
   - `trajectory × horizon`；
   - 至少包含 Frame Rel-L2 和与实验核心假设直接相关的逐帧诊断量。

推荐额外生成 `by_horizon.png`，但 CSV / JSON 是正式审计事实源。

## 4. Paired A/B 与长训练实验

### Paired A/B

若 baseline / candidate 使用同一 Dev replay：

- 必须逐 horizon 报告 candidate 相对 baseline 的 improvement；
- 标记哪些 horizon 改善、哪些退化；
- README 至少总结 early / middle / late horizon 三段的差异。

### Long training / continuation

不要求对每个 checkpoint 都重放 Future20。

至少分析：

- 最终 / 决策 checkpoint；
- 一个能代表较早阶段的关键 checkpoint，若 checkpoint-selection / late-training behavior 是实验结论的一部分；
- 若存在严格 matched baseline checkpoint，则优先加入同一分析。

## 5. 结果解释要求

Sol / Codex 的实验 README 不能只写“overall 变好”。至少回答：

1. 收益主要出现在 `t+1..t+N` 的早期、中期还是后期？
2. error growth rate 是否下降？
3. 是否出现 long-horizon drift / variance collapse / energy amplitude bias？
4. aggregate 改善是否掩盖某些 horizon 明显退化？
5. paired A/B 的逐帧收益是否跨 trajectory 稳定？

逐帧分析是诊断证据，不允许据单个 horizon 做 trajectory-specific / horizon-specific 手工修补，除非后续另行预注册控制实验。

## 6. Artifact 与执行原则

- 优先复用已有 prediction artifact；
- 若历史实验只保存 checkpoint，没有 prediction replay，可固定原 Dev / scorer / checkpoint 做 prediction-only replay；
- 不因补逐帧分析重新训练模型；
- replay 必须记录 checkpoint SHA、manifest SHA、scorer SHA 和 execution commit；
- prediction replay 与原 experiment aggregate metrics 应做 parity check；明显不一致则 `REVIEW_REQUIRED / INVALID_REPLAY`。

## 7. 最终原则

**所有正式 Future20 实验：overall + trajectory-level + by-horizon 是默认最小结果证据。**

**官方 Rel-L2 / TKE / MVPE 的汇总定义保持不变；逐帧 TKE / MVPE 只能作为明确标注的机制诊断量。**
