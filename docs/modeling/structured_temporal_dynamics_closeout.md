# Structured Temporal Dynamics 方向收口

> Track 1 Modeling 方向长期记忆。本文汇总 Structured Temporal Dynamics（结构化时序动力学）从假设、R1/R2 粗筛到 Vorticity 长程终局验证的完整过程。实验细节仍以各 coordination handoff 和 `track1_experiment_registry.md` 为准。

## 1. 初始问题

Track 1 当前主任务是 `Past20 u/v → Future20 u/v`。最初怀疑有两个潜在瓶颈：

1. Future20 被一次性直接回归，模型是否没有充分利用未来 20 帧内部的连续时序关系；
2. 直接预测 u/v 是否过于困难，是否可以通过速度增量、涡量等中间物理变量提供更好的监督。

因此本方向不是单纯寻找一个 temporal layer，而是验证：

- Temporal target / supervision 是否有价值；
- 显式 Future20 temporal interaction 是否是当前 CNO 的主要结构瓶颈；
- Physics auxiliary target 是否能稳定帮助 u/v forecasting。

## 2. R1：低成本机制粗筛

R1 固定 50 Train / 16 Dev、P0-A、N2、stride20、Direct@1500 起点，连续比较：

- C0：Direct CNO control；
- T1：ΔUV Temporal Supervision，增加 Future20 帧间速度增量 MSE；
- T2：Vorticity Supervision，增加 `ω = dv/dx - du/dy` 的涡量 MSE；
- T3：Future20 output-space Temporal Mixer。

R1 因共享 GPU 使用 micro-batch2 + accumulation4，effective batch8。@3000 raw errors：

| Arm | Rel-L2 | TKE | MVPE | 相对 C0 直观结果 |
|---|---:|---:|---:|---|
| C0 | 0.197545 | 0.633871 | 0.170807 | control |
| T1 ΔUV | 0.186753 | 0.654265 | 0.170106 | Rel 明显改善，但 TKE 明显恶化 |
| T2 Vorticity | 0.199202 | 0.630783 | 0.170158 | 弱正/混合信号 |
| T3 Temporal Mixer | 0.198035 | 0.634338 | 0.166572 | 没有形成稳定收益 |

### R1 结论

- ΔUV 证明 temporal-change supervision 能改变模型，但出现明确 Rel/TKE trade-off；
- Vorticity 出现值得复核的弱正信号；
- output-level temporal correction 没有证明价值；
- R1 T3 还存在 optimizer-state confound，因此不能用来否定 temporal architecture。

状态：方向保持开放，进入真正 latent-level architecture 粗筛。

Handoff：`docs/coordination/CHATGPT_HANDOFF_STRUCTURED_TEMPORAL_DYNAMICS_R1.md`

## 3. R2：真正的 latent temporal architecture 粗筛

R2 回到 physical batch8 / no accumulation，并修正 optimizer fairness：先恢复 Direct@1500 backbone optimizer state，再把新 temporal module 放入独立 param group。

Arms：

- C0：canonical Direct control；
- V1：Vorticity Supervision；
- A1：Latent Temporal Conv，在原 CNO `project` 前做 temporal-only Conv3D；
- A2：Latent Temporal Attention，在每个 spatial location 上对 Future20 latent 做单层 temporal-only MHA。

@3000 raw errors：

| Arm | Rel-L2 | TKE | MVPE | 相对 C0 |
|---|---:|---:|---:|---|
| C0 | 0.161343 | 0.557971 | 0.131496 | control |
| V1 Vorticity | 0.156023 | 0.555572 | 0.121585 | Rel +3.30%，TKE +0.43%，MVPE +7.54% |
| A1 Temporal Conv | 0.159602 | 0.556817 | 0.127943 | 小幅正信号，约 1%～3% |
| A2 Temporal Attention | 0.161400 | 0.559184 | 0.131574 | 基本无增益 |

Horizon 诊断中 A1 的 far horizon 信号略强，但绝对收益仍小；A2 在 Early/Mid/Late/Far 均没有形成结构性改善。

### R2 结论

- **Latent Temporal Attention = NO_GO**；
- **Latent Temporal Conv = WEAK_SIGNAL_PARKED**，不足以继续 architecture 深挖；
- **Vorticity Supervision = PROMISING short-run signal**，而且不改变 inference architecture，因此值得做一次长程终局验证；
- 当前没有证据支持“缺少显式 Future20 temporal architecture 是 CNO 的主要性能瓶颈”。

Handoff：`docs/coordination/CHATGPT_HANDOFF_STRUCTURED_TEMPORAL_DYNAMICS_R2.md`

## 4. Vorticity 长程终局验证

短训 V1 信号较强，但 MF 曾出现 short-run 很好、long-run wash out 的经验，因此本方向最后只验证一个问题：

> Vorticity Supervision 的优势能否在长训练后稳定保持，并达到值得进入 SOTA merge pool 的强度？

原 batch8 C0-LONG 已先训练到 15000，但因 GPU 长期共享且旧 V1 尚未启动，最终改成一组严格 matched 的 low-memory paired final：

- C0-LOWMEM / V1-LOWMEM 均从各自 R2@3000 model + optimizer checkpoint 恢复；
- micro-batch4 + accumulation2，effective batch8；
- CUDA per-process cap 12 GiB；
- 两臂 peak reserved 均约 9.80 GiB；
- V1 固定 `lambda_vort=15.5385751724`；
- 训练到 absolute update12000；
- locked-final / full-data / SPS / Codabench 均未访问。

### Long-run matched results

Improvement 定义为 `(C0 - V1) / C0 * 100`，正数表示 V1 更好。

| Update | Rel improvement | TKE improvement | MVPE improvement |
|---:|---:|---:|---:|
| 3000 | +3.297% | +0.430% | +7.537% |
| 4500 | +4.402% | -0.254% | +9.133% |
| 6000 | +3.243% | -0.564% | -0.048% |
| 9000 | +2.722% | -1.657% | +6.933% |
| 12000 | +3.948% | +1.599% | +2.059% |

Late gate 固定使用 6000 / 9000 / 12000：

- median Rel-L2 improvement = `+3.242654%`，PASS；
- median TKE improvement = `-0.564134%`，PASS；
- median MVPE improvement = `+2.059027%`，FAIL，要求 `>=4%`；
- 2-of-3 checkpoint rule = FAIL，仅 @12000 同时满足 Rel>0、MVPE>0、TKE>=-1%。

因此机械 gate：

**`FINAL_GATE = PARK`**

@12000 independent official-v9 replay：C0/V1 三项 raw-error 最大差异均为 `0.0`，结果可重复。

Trajectory stability（6000 / 9000 / 12000）：

- Rel wins：`16/16 / 14/16 / 16/16`；
- TKE wins：`5/16 / 15/16 / 14/16`；
- MVPE wins：`5/16 / 12/16 / 9/16`；
- all-three：`1/16 / 11/16 / 7/16`。

Handoff：`docs/coordination/CHATGPT_HANDOFF_VORTICITY_LOWMEM_FINAL.md`

## 5. 最终科研判断

### 5.1 Structured Temporal Dynamics

**最终状态：`CLOSED / WEAK_SIGNAL`**。

直觉结论：

> 当前 CNO 并不是因为“完全没有显式建模 Future20 内部时序关系”而卡住。它已经能够隐式学习相当一部分时间关系。给模型增加更强 temporal module，没有出现预期中的大台阶。

具体证据：

- ΔUV temporal supervision 有信号，但以 TKE trade-off 为代价；
- output Temporal Mixer 无稳定收益；
- latent Temporal Conv 只有 1%～3% 弱信号；
- latent Temporal Attention 基本无收益。

因此不再继续：

- Temporal Transformer；
- SSM；
- autoregressive / block rollout；
- 更多 Temporal Conv / Attention 深度或宽度扫描；
- ΔUV 权重扫描。

### 5.2 Vorticity Supervision

**最终状态：`PARK`，不是 SOTA merge candidate。**

Vorticity 的物理直觉得到部分支持：Rel-L2 长程改善非常稳定，说明显式 vortex/local-rotation supervision 确实改变了模型学习方向；但 MVPE 收益随训练阶段明显波动，late median 只有 `+2.06%`，不足以达到比赛级大台阶。

因此不再继续：

- `lambda_vort` 扫描；
- Vorticity + Temporal module 组合；
- Vorticity full-data / Codabench submission；
- 其它仅为挽救当前 Vorticity gate 的精修。

未来只有出现新的独立机制证据时才重新打开。

## 6. 对全局研发的含义

本方向的价值主要是**排除了一个看似很大的局部最优陷阱**：继续为当前 CNO 堆 Temporal Transformer / autoregressive mechanism 的赔率不高。

接下来研究带宽优先转向仍未充分覆盖的一级变量：

1. Random Window / Phase Augmentation；
2. Backbone Family：CNO / FNO / Transolver；
3. POD / Modal Dynamics；
4. 其它真正改变 representation / data regime 的方向。

遵循 60 分原则，不再为本方向追加 R4。

## 7. 关键证据索引

- R1：`docs/coordination/CHATGPT_HANDOFF_STRUCTURED_TEMPORAL_DYNAMICS_R1.md`
- R2：`docs/coordination/CHATGPT_HANDOFF_STRUCTURED_TEMPORAL_DYNAMICS_R2.md`
- Long progress：`docs/coordination/CHATGPT_HANDOFF_VORTICITY_LONG_FINAL_PROGRESS.md`
- Low-memory final：`docs/coordination/CHATGPT_HANDOFF_VORTICITY_LOWMEM_FINAL.md`
- Modeling overview：`docs/modeling/modeling概要.md`
- Experiment registry：`docs/track1_experiment_registry.md`
- Remote final artifacts：`/home/chyfuture/realpde_runs/vorticity_lowmem_final_20260915/`
