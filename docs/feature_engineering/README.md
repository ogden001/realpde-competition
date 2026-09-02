# RealPDE Track 1 特征工程方向总结

**方向：Feature Engineering**  
**适用赛道：RealPDE Competition Track 1**  
**文档性质：长期技术记忆 / 阶段性研究总结**  
**当前状态：Feature Discovery 已收口，Feature Fusion 研究中**  
**最后更新：2026-09-02**

---

# 1. 文档目的

本文档是 RealPDE Track 1 项目中 **Feature Engineering 方向的长期技术记忆**。

它不是某一次实验的 handoff，也不是单次任务报告。

本文档用于长期回答以下问题：

1. 为什么要做 Feature Engineering；
2. Track 1 正式推理阶段允许使用哪些信息；
3. 我们已经研究过哪些 Feature；
4. 哪些 Feature 被证明具有信息价值；
5. 哪些 Feature 属于数学重表达或冗余；
6. Feature Discovery 阶段做过哪些关键实验；
7. 当前如何理解 Feature Value 与 Feature Fusion Value；
8. 历史 Feature Fusion 为什么没有成功；
9. Feature Fusion 下一阶段应该重点研究什么；
10. 当前实验推进到了哪里；
11. 后续该方向如何组织实验与 Codex 执行。

后续 Feature Engineering 方向出现新的稳定结论时，应更新本文档。

单次实验的详细指标、命令、artifact provenance 等仍应保留在：

- `docs/coordination/`
- experiment registry
- experiment-specific report / handoff

本文档只保留已经具有阶段意义的事实、判断和研究路线。

---

# 2. Feature Engineering 的方向定位

RealPDE Track 1 的基本任务可以抽象为：

```text
过去 20 帧流场
        ↓
预测未来 20 帧流场
```

正式推理输入本质上是二维速度场序列。

当前可靠的物理输入主要是：

* `u`
* `v`

虽然模型接口存在 `p` channel，但在当前数据和官方 runtime 协议中：

```text
p
```

主要应视为兼容占位。

如果原始输入中不存在真实 pressure，loader 会填入 0。

因此：

> 不应把 `p` 当成一个可靠的真实压力物理场进行 Feature Engineering。

Feature Engineering 的核心目标不是不断增加人工变量，而是：

> **从当前 20 帧 `u/v` 中提炼模型可能没有充分利用的时间结构、波动结构和空间结构，并研究这些信息如何帮助未来 20 帧预测。**

整个 Feature Engineering 方向应分成两个阶段。

---

# 3. Feature Engineering 的两个阶段

## 3.1 Feature Discovery

研究问题：

> 什么信息值得构造？

重点回答：

* 什么 Feature 是 runtime-safe 的；
* 什么 Feature 表示真正不同的信息；
* 什么 Feature 只是数学重表达；
* 什么 Feature 在 train/dev 数据分布上稳定；
* 什么 Feature 对未来预测具有 incremental information；
* 强模型是否已经完全吸收这些信息。

这一阶段回答的是：

> **WHAT：有什么信息值得保留。**

当前状态：

```text
Feature Discovery = CLOSED
```

目前没有必要继续无边界扩展 Feature Catalog。

---

## 3.2 Feature Fusion

研究问题：

> 已经找到的信息应该如何交给模型？

包括：

* 输入层 concat；
* latent fusion；
* multi-branch；
* conditioning；
* gating；
* FiLM；
* residual correction；
* auxiliary objective；
* feature-aware loss。

这一阶段回答的是：

> **HOW：模型应该如何利用这些信息。**

当前 Feature Engineering 的主要研究矛盾已经转移到 Feature Fusion。

---

# 4. Runtime 硬约束

所有最终进入正式模型的 Feature 必须是 runtime-safe。

正式推理阶段可靠获得：

```text
当前 20 帧 u/v
+
tensor shape
```

允许使用：

* 当前 20 帧 `u/v`
* 当前输入窗口内部时间统计
* pixel-space difference
* tensor shape 推导出的像素位置
* 当前输入自身构造的 derived quantity

禁止正式 Feature 依赖：

* Re
* AoA
* physical x/y
* CFD
* private HDF5 metadata
* private geometry
* hidden body mask
* future trajectory
* target
* locked-final
* private test

此外：

> 不允许通过 `u == 0` 或 `v == 0` 自动推断 solid / invalid region。

当前官方输入并没有显式、可靠的 body mask 可供正式 inference 使用。

因此当前 Feature Engineering 研究的是：

```text
Input-derived Prior
```

而不是：

```text
External Metadata Engineering
```

---

# 5. 当前 Feature Information Map

经过目前的 Feature Discovery，真正值得保留的信息可以压缩为：

```text
Raw Flow
+
Mean State
+
Fluctuation Strength
+
Recent Dynamics
+
Spatial Structure
```

如果只讨论相对于 raw `u/v` 新增的 Feature Engineering 信息，则进一步压缩为：

```text
Mean
+
Fluctuation
+
Recent Dynamics
+
Spatial Structure
```

当前综合优先级为：

```text
Temporal Prior
    >
Spatial Prior
    >
Derived Physical Summary
```

---

# 6. Raw Flow

基础输入：

```text
u
v
```

它们表示二维流场两个速度方向上的分量。

Raw Flow 是整个 Feature 体系的根信息。

后续几乎所有 Feature：

* mean
* std
* delta
* gradient
* vorticity
* TKE proxy

都可以由 raw `u/v` 计算得到。

因此 Raw Flow 本身不是需要筛选的 Feature Candidate。

它是：

> **Feature Engineering 的信息底座。**

---

# 7. Temporal Prior

Temporal 是当前证据最明确、优先级最高的一类 Feature。

冻结 Feature Package：

```text
TEMPORAL6
```

包含：

```text
mean_u_20
mean_v_20

std_u_20
std_v_20

delta_u
delta_v
```

Temporal6 可以进一步理解为三个完全不同的时间维度：

```text
Mean
+
Fluctuation
+
Recent Dynamics
```

---

# 8. Mean State

定义：

```text
mean_u_20
mean_v_20
```

即最近 20 帧：

```text
mean_u = Mean_t(u)
mean_v = Mean_t(v)
```

Mean 描述的是：

> **过去一段时间中，这个位置整体处于什么流动状态。**

例如：

两个位置最后一帧的 `u/v` 完全相同，

并不意味着：

```text
mean_u_20
mean_v_20
```

也相同。

因此 Mean 提供的是：

* temporal aggregation
* background flow
* mean-flow structure
* bias information

历史 residual decomposition 也发现：

> 一部分 Rel-L2 改善与 mean / bias correction 高度相关。

因此 Mean State 是当前最值得保留的 Temporal Prior 之一。

---

# 9. Fluctuation Strength

定义：

```text
std_u_20
std_v_20
```

表示过去 20 帧中：

> **该位置速度变化有多剧烈。**

例如两个位置：

```text
mean_u ≈ 1.0
```

但可能分别是：

```text
0.99 1.01 1.00 1.00 ...
```

和：

```text
0.5 1.5 0.6 1.4 ...
```

Mean 相同，

但后者具有明显更强的 temporal fluctuation。

因此：

```text
Mean
```

回答：

> 整体是什么状态？

而：

```text
Std
```

回答：

> 这个状态有多不稳定？

Std 与以下物理过程存在直接联系：

* 非定常流动
* wake fluctuation
* local disturbance
* turbulence-like fluctuation
* TKE

这一类 Feature 对本比赛尤其重要。

因为 Track 1 官方指标中明确包含：

```text
TKE
```

因此：

> **Fluctuation information 不能被简单看作普通统计量。**

它实际上对应比赛最重要的动态结构之一。

---

# 10. Recent Dynamics

当前冻结定义：

```text
delta_u = u_last - u_prev
delta_v = v_last - v_prev
```

Recent Dynamics 描述：

> **在预测开始之前的最后时刻，流场正在怎样变化。**

Temporal6 中三个信息维度可以这样理解：

| Feature | 回答的问题          |
| ------- | -------------- |
| Mean    | 最近一段时间整体是什么状态  |
| Std     | 最近一段时间波动有多强    |
| Delta   | 预测开始前正在往哪个方向变化 |

数据诊断结果显示：

```text
corr(delta_u, mean_u)
```

和：

```text
corr(delta_v, mean_v)
```

都很低。

因此：

> Recent Dynamics 与 Mean State 是明显不同的信息维度。

这也是 Temporal6 不应该进一步被压缩成单一时间统计量的重要原因。

---

# 11. Spatial Prior

当前冻结：

```text
SPATIAL4
```

包含：

```text
du_dx_pixel
du_dy_pixel
dv_dx_pixel
dv_dy_pixel
```

这些量描述：

> **当前流场在空间邻域中变化得有多快。**

Raw Flow 告诉模型：

```text
这个点速度是多少
```

Spatial Gradient 告诉模型：

```text
这个点周围速度是怎样变化的
```

因此可以帮助描述：

* shear
* wake boundary
* vortex neighborhood
* strong-gradient region
* local spatial variation
* spatial structure transition

---

# 12. Spatial Gradient 的冻结定义

当前 Spatial Gradient 是：

```text
pixel-space derivative
```

而不是严格物理坐标意义上的：

```text
physical derivative
```

冻结规则：

```text
pixel spacing = 1
```

内部像素：

```text
centered difference
```

最外层：

```text
forward / backward difference
```

同时：

* 不 smoothing
* 不 clipping
* 不使用 body mask
* 不引入 physical x/y
* 不做额外 geometry correction

因此后续文档应统一称为：

```text
du_dx_pixel
du_dy_pixel
dv_dx_pixel
dv_dy_pixel
```

避免误写为严格物理单位下的速度梯度。

---

# 13. Fluctuation Field

定义：

```text
u' = u - mean_u
v' = v - mean_v
```

对应经典流场分解：

```text
instantaneous flow
=
mean flow
+
fluctuation
```

`u'/v'` 有很好的物理解释价值。

但是数学上：

只要已有：

```text
raw u/v
+
mean_u/mean_v
```

就可以严格恢复：

```text
u'
v'
```

因此：

> `u'/v'` 不属于新的 independent information。

它应该被理解为：

```text
Physical Re-expression
```

而不是新的 Feature Family。

---

# 14. TKE Proxy

当前输入侧定义：

```text
TKE_proxy
=
0.5 * (std_u^2 + std_v^2)
```

它可以理解为：

> 当前输入 20 帧窗口中，局部 velocity fluctuation energy 的摘要。

它与比赛 TKE 指标具有明显的物理概念联系。

但它完全由：

```text
std_u
std_v
```

计算得到。

因此：

> TKE Proxy 不是新的 primitive information。

更适合未来作为：

* physical summary
* auxiliary target
* conditioning signal
* gating information
* loss
* diagnostic
* fluctuation protection signal

而不是继续扩充输入 Feature Catalog。

---

# 15. Vorticity

当前定义：

```text
vorticity_pixel
=
dv_dx_pixel - du_dy_pixel
```

它描述：

> 局部流场旋转或涡结构的强弱。

其物理解释非常直观。

但由于严格由：

```text
dv_dx_pixel
du_dy_pixel
```

计算得到，

因此：

> Vorticity 是 Derived Spatial Summary。

不是新的 primitive Feature Family。

---

# 16. Speed

定义：

```text
speed = sqrt(u^2 + v^2)
```

Feature Data Diagnostic 发现：

```text
corr(speed, abs(u))
≈
0.9999
```

说明当前数据分布中：

> velocity magnitude 基本被 `u` 主方向速度支配。

因此：

```text
speed
```

有解释价值，

但独立新增信息极少。

当前状态：

```text
LOW PRIORITY
```

---

# 17. u2_prime_mean / v2_prime_mean

已经确认：

```text
u2_prime_mean = std_u^2
v2_prime_mean = std_v^2
```

在当前定义下属于严格数学冗余。

因此：

```text
u2_prime_mean
v2_prime_mean
```

不再具有 independent Feature Candidate 身份。

---

# 18. 当前冻结的 Feature Catalog

## Primitive Information

### Raw

```text
u
v
```

---

### TEMPORAL6

```text
mean_u_20
mean_v_20
std_u_20
std_v_20
delta_u
delta_v
```

---

### SPATIAL4

```text
du_dx_pixel
du_dy_pixel
dv_dx_pixel
dv_dy_pixel
```

---

## Derived / Re-expression

```text
u'
v'
TKE_proxy
vorticity_pixel
```

---

## Low Priority / Redundant

```text
speed
u2_prime_mean
v2_prime_mean
```

---

# 19. Feature Discovery 阶段：Batch1 Data Diagnostic

实验：

```text
T1-ID-FE-DATA01-B1-S20260902
```

Protocol：

```text
50 train trajectories
16 dev trajectories
```

纯输入窗口诊断：

```text
T_in = 20
stride = 20
```

总窗口：

```text
2102 train
675 dev
```

原始诊断空间尺寸：

```text
64 × 128
```

该实验：

* 不使用 target
* 不使用 CFD
* 不使用 Re
* 不使用 AoA
* 不使用 private geometry
* 不使用 locked-final
* 不训练模型

---

# 20. Batch1 Data Diagnostic 主要结论

所有核心 Feature：

```text
finite_ratio = 1.0
```

说明数值稳定。

同时发现：

```text
corr(speed, abs(u))
≈
0.99991 train
≈
0.99992 dev
```

因此：

```text
speed
```

高度冗余。

同时：

```text
std_u^2
```

与：

```text
u2_prime_mean
```

严格一致。

```text
std_v^2
```

与：

```text
v2_prime_mean
```

严格一致。

因此这两个 fluctuation-energy field 不需要作为独立 Feature。

另外：

```text
corr(delta_u, mean_u)
≈ -0.014
```

```text
corr(delta_v, mean_v)
≈ -0.098
```

说明：

> Recent Dynamics 与 Mean State 提供明显不同的信息。

最终 Batch1 shortlist：

```text
KEEP:
raw u/v
mean
std
delta
u'/v'
TKE proxy

LOW VALUE / REDUNDANT:
speed
u2_prime_mean
v2_prime_mean
```

---

# 21. Spatial Diagnostic

实验：

```text
T1-ID-FE-SPATIAL-DATA01-S20260902
```

使用：

```text
du_dx_pixel
du_dy_pixel
dv_dx_pixel
dv_dy_pixel
vorticity_pixel
```

数据：

```text
50 train
16 dev
```

窗口：

```text
2102 train
675 dev
```

空间尺寸：

```text
64 × 128
```

不训练模型。

---

# 22. Spatial Diagnostic 主要结论

四个 primitive gradient：

```text
du_dx
du_dy
dv_dx
dv_dy
```

全部：

* finite
* 数值稳定
* train/dev 量级总体一致

Outer-edge 与 interior 的统计比较发现：

> image outer edge 并没有形成异常大的 gradient magnitude。

因此：

> 当前 pixel-space finite difference 并不存在明显由图像边缘造成的主导 artifact。

Vorticity 也验证：

```text
vorticity
=
dv_dx - du_dy
```

严格一致。

因此最终 Spatial 判断：

```text
SPATIAL4 = KEEP
```

而：

```text
vorticity = KEEP AS DERIVED SUMMARY
```

---

# 23. PERSIST Incremental Probe

实验：

```text
T1-ID-FE-INCR-PERSIST-RIDGE-S20260902
```

目的：

> 在不依赖复杂神经网络的情况下，判断 Feature 是否真的包含未来预测信息。

Baseline：

```text
PERSIST
```

即：

> 把最后一个 observed frame 重复作为未来 20 帧预测。

Supervised protocol：

```text
20 input
→
20 target
```

stride：

```text
20
```

数据：

```text
50 train
16 dev
```

完整窗口：

```text
2052 train
659 dev
```

runtime resolution：

```text
32 × 64
```

---

# 24. PERSIST Probe Feature Packages

Raw-Control：

```text
last u
last v
```

Temporal：

```text
last u/v
+
mean_u/v
+
std_u/v
+
delta_u/v
```

Spatial：

```text
last u/v
+
4 primitive gradients
```

Joint：

```text
Raw
+
Temporal
+
Spatial
```

使用：

```text
closed-form ridge
```

预测：

```text
future target - PERSIST
```

即 residual。

---

# 25. PERSIST Probe 结果

Dev：

| Package              |       Rel-L2 |          TKE |         MVPE |
| -------------------- | -----------: | -----------: | -----------: |
| Raw-Control          |     0.131353 |     0.987575 |     0.130701 |
| Raw+Temporal         |     0.118340 |     0.940578 |     0.109454 |
| Raw+Spatial          |     0.130241 |     0.973685 |     0.129355 |
| Raw+Temporal+Spatial | **0.116346** | **0.936060** | **0.107228** |

Temporal 相比 Raw-Control：

```text
Rel-L2 -0.013013
TKE    -0.046997
MVPE   -0.021247
```

Joint 相比 Raw-Control：

```text
Rel-L2 -0.015007
TKE    -0.051515
MVPE   -0.023473
```

Trajectory-level win rate：

Temporal：

```text
Rel-L2 0.875
TKE    1.000
MVPE   0.938
```

Joint：

```text
Rel-L2 0.938
TKE    1.000
MVPE   1.000
```

因此 PERSIST Probe 给出非常明确的信息：

> **Temporal Feature 中存在真实的 future predictive information。**

Spatial 单独增量较弱，

但总体方向一致。

Joint 最好。

---

# 26. Frozen Strong CNO Incremental Probe

实验：

```text
T1-ID-FE-INCR-FROZEN-CNO-E0-RIDGE-S20260902
```

目的：

> 判断强 CNO 已经看到完整 raw 20-frame sequence 后，Temporal / Spatial Feature 是否仍然具有模型尚未充分利用的 residual signal。

Frozen CNO baseline：

```text
T1-ID-LOSS-E0-90M-S20260901
```

Checkpoint SHA：

```text
5d02c8da5bcbcbcc47917b0021b1007b2036931a3a51483772f7607deeb4aff6
```

Frozen CNO raw dev：

```text
Rel-L2 0.168923
TKE    0.538475
MVPE   0.136146
```

---

# 27. Frozen Strong CNO Probe 结果

| Package              |       Rel-L2 |      TKE |         MVPE |
| -------------------- | -----------: | -------: | -----------: |
| Raw-Control          |     0.168162 | 0.594538 |     0.135999 |
| Raw+Temporal         |     0.161318 | 0.608969 |     0.135610 |
| Raw+Spatial          |     0.166255 | 0.624081 |     0.135474 |
| Raw+Temporal+Spatial | **0.159876** | 0.631270 | **0.135046** |

Temporal vs Raw-Control：

```text
Rel-L2 -0.006843
TKE    +0.014432
MVPE   -0.000390
```

Spatial vs Raw-Control：

```text
Rel-L2 -0.001907
TKE    +0.029543
MVPE   -0.000525
```

Joint vs Raw-Control：

```text
Rel-L2 -0.008286
TKE    +0.036732
MVPE   -0.000954
```

注意：

> error 越低越好。

因此：

```text
Rel-L2 ↓
MVPE   ↓
```

表示改善。

而：

```text
TKE ↑
```

表示恶化。

---

# 28. Frozen CNO Probe 的核心解释

这个实验最重要的结论不是：

> Temporal / Spatial 无效。

恰恰相反。

Temporal / Spatial 在强 CNO residual 上仍然能够改善：

```text
Rel-L2
MVPE
```

说明：

> 强 CNO 并没有完全利用掉这些信息。

尤其 Joint：

```text
TEMPORAL6 + SPATIAL4
```

获得最大 Rel-L2 residual 改善。

Spatial 在 Temporal 之外也仍然存在额外：

```text
Rel-L2 / MVPE
```

signal。

---

# 29. TKE Trade-off

Frozen CNO Probe 同时暴露出严重问题：

所有 ridge residual correction 都伤害 TKE。

甚至：

```text
Raw-Control correction
```

本身就让：

```text
TKE

0.538475
→
0.594538
```

因此不能简单解释为：

> Feature 导致 TKE 变差。

更合理的解释是：

> **这种简单 residual correction mechanism 本身容易改善 point-wise / mean-like error，同时破坏 temporal fluctuation structure。**

因此 Feature Engineering 阶段形成一个非常重要的统一判断：

```text
FEATURE_VALUE_POSITIVE_BUT_FUSION_HISTORY_NEGATIVE
```

含义是：

> Feature information value 为正。

但：

> Feature Fusion implementation value 尚未被证明。

---

# 30. 历史 Neural Feature Fusion

历史 Neural Fusion 已经做过：

## FE-00

CNO baseline：

```text
Rel-L2 0.190821
TKE    0.644070
MVPE   0.144258
```

---

## FE-00R Raw-Control

相同 CNO：

```text
+
178-param residual head
```

只输入 raw information。

结果：

```text
Rel-L2 0.183850
TKE    0.636973
MVPE   0.133649
```

---

## FE-01 Temporal

Temporal Feature 注入同一种 residual fusion：

```text
Rel-L2 0.189399
TKE    0.641316
MVPE   0.140459
```

没有优于 Raw-Control。

---

## FE-02 SpatialPhysics

结果：

```text
Rel-L2 0.177478
TKE    0.640643
MVPE   0.140967
```

Rel-L2 明显改善，

但 MVPE 恶化。

没有通过 multi-metric gate。

---

# 31. 如何解释历史 FE-01 / FE-02

历史实验只能说明：

> **那一种 Feature Fusion Implementation 没有稳定成功。**

不能解释成：

```text
Temporal Feature 无价值
```

或者：

```text
Spatial Feature 无价值
```

因为后续 PERSIST / Frozen CNO Incremental Probe 已经证明：

Temporal / Spatial 中确实存在预测增量信息。

因此后续任何文档必须严格区分：

```text
Feature Value
```

和：

```text
Fusion Implementation Value
```

这是 Feature Engineering 方向的重要解释边界。

---

# 32. Feature Discovery 最终结论

目前 Feature Discovery 已经基本回答：

> 哪些 Feature 值得保留？

最终信息体系为：

```text
Raw
+
Temporal
+
Spatial
```

其中：

```text
TEMPORAL6
=
mean_u_20
mean_v_20
std_u_20
std_v_20
delta_u
delta_v
```

```text
SPATIAL4
=
du_dx_pixel
du_dy_pixel
dv_dx_pixel
dv_dy_pixel
```

Derived：

```text
u'
v'
TKE_proxy
vorticity_pixel
```

Low Priority / Redundant：

```text
speed
u2_prime_mean
v2_prime_mean
```

因此：

```text
Feature Discovery = CLOSED
```

---

# 33. 当前不建议继续扩展的 Feature

没有新的明确证据前，不自动继续构造：

* Local Mean / Local Std
* Wake-specific Feature
* FFT
* spectral Feature
* POD
* Laplacian
* higher-order derivatives
* 大量手工 neighborhood statistics
* 大量 Feature combination

原因不是这些 Feature 一定没有价值。

而是：

> 当前主要瓶颈已经从“还有什么 Feature”变成“怎么使用已经找到的 Feature”。

继续扩充 Feature Catalog 很容易导致 Feature Explosion。

---

# 34. 为什么 Feature Engineering 仍然有意义

目前所有核心 Feature 最终都可以由 raw `u/v` 数学计算出来。

理论上：

> CNO 应该有能力自己学习 mean / std / delta / gradient。

但是：

```text
信息存在于 raw input
```

并不等于：

```text
模型已经高效、稳定地利用了这些信息
```

实际模型受到：

* 数据量
* optimization
* architecture
* training budget
* inductive bias

限制。

Frozen CNO Probe 已经直接证明：

> CNO 后仍然存在 Temporal / Spatial residual information。

因此 Feature Engineering 的价值正在从：

```text
人工制造更多变量
```

转变为：

```text
发现模型尚未充分利用的结构
+
设计更合理的 inductive bias
```

---

# 35. Feature Fusion 的核心研究问题

Feature Fusion 的核心目标不是：

> 让 Feature 更强地直接修改 prediction。

而是：

> **如何利用 Temporal / Spatial prior，同时保护 temporal fluctuation structure。**

当前最典型的失败模式是：

```text
Rel-L2 ↓
MVPE   ↓

但是

TKE    ↑
```

这表示：

> point-wise prediction 更准，但时间波动结构反而变差。

因此 Feature Fusion 必须同时考虑：

```text
Mean Flow
+
Point Accuracy
+
Fluctuation Structure
+
TKE
```

---

# 36. Feature Fusion 技术路线

目前优先保留三个方向。

---

## FF-01：Conditioning / Gating

基本结构：

```text
Raw u/v
   ↓
CNO backbone
   ↓
latent z
   ↑
Temporal / Spatial Feature
   ↓
Condition Encoder
   ↓
gamma / beta / gate
```

典型形式：

```text
z'
=
z * (1 + gamma)
+
beta
```

Feature 不直接生成：

```text
velocity residual
```

而是：

> 调节主模型内部 representation。

这与历史 residual correction 有本质区别。

Feature 在这种设计里是：

```text
prior / condition
```

而不是：

```text
second predictor
```

当前优先级较高。

---

# 37. FF-02：Multi-Branch Prior Fusion

基本结构：

```text
Raw Flow
   ↓
Main CNO Branch
   ↓
z_raw
        \
         Fusion
        /
Temporal / Spatial Prior
   ↓
Lightweight Prior Branch
   ↓
z_prior
```

核心思想：

> Raw Flow 和人工 Prior 分别形成 representation，然后在 latent space 融合。

Prior Branch 必须：

* lightweight
* 参数显著少于主干
* 不成为第二套完整 predictor

未来重点比较：

```text
Temporal
```

和：

```text
Temporal + Spatial
```

判断 Spatial 是否值得增加额外结构复杂度。

---

# 38. FF-03：Feature-aware Objective

这一方向甚至不需要把 Feature 当作模型输入。

模型仍然只输入：

```text
raw u/v
```

但训练目标显式约束：

```text
Mean
Std
TKE
Gradient
Vorticity
```

例如：

```text
L
=
L_velocity
+
lambda_tke * L_tke
+
lambda_std * L_std
+
lambda_grad * L_grad
```

本质上是把 Feature Engineering 从：

```text
Input Engineering
```

推进到：

```text
Representation / Objective Engineering
```

---

# 39. FF-00 Protocol Freeze

Feature Fusion 已完成：

```text
FF-00
Fusion Protocol & Baseline Freeze
```

FF-00 冻结：

```text
TEMPORAL6
SPATIAL4
TEMPORAL6_SPATIAL4
```

并冻结：

* manifest
* scorer
* 20 → 20 protocol
* 32×64 runtime
* matched Raw-Control 原则
* dev-only selection
* trajectory-level stability evidence
* Rel-L2 / TKE / MVPE 三指标保护原则

---

# 40. FF-00 Baseline Provenance

Historical strong baseline：

```text
T1-ID-LOSS-E0-90M-S20260901
```

属于：

```text
OFFICIAL_WARM_START
```

Checkpoint SHA：

```text
5d02c8da5bcbcbcc47917b0021b1007b2036931a3a51483772f7607deeb4aff6
```

历史训练 source commit：

```text
UNKNOWN / NOT RECOVERED
```

项目已接受：

```text
BASELINE_PROVENANCE_EXCEPTION_ACCEPTED
```

含义：

> 可以把这个 checkpoint 当成 immutable artifact baseline 继续使用。

但：

> 不允许声称历史训练过程可以从 Git source 完整复现。

另外：

`OFFICIAL_WARM_START`

更适合：

```text
competition-oriented comparison
```

而不是：

```text
clean causal method selection
```

因此 Feature Fusion 的方法筛选原则上应优先使用 CLEAN initialization。

---

# 41. Feature-aware Loss Duplication Audit

FF-00 同时检查了历史 Loss Optimization。

已经实质覆盖：

## Mean consistency

历史 E3 已包含：

```text
Mean loss
```

并没有解决 TKE trade-off。

因此：

> 不应简单重复 Mean loss。

---

## TKE consistency

已经被：

```text
E0 / E1 / E2 / E3
N0 / N1 / N2
```

广泛测试。

因此：

> 不应把同样的 TKE loss 换个 Feature-aware 名字重新跑。

---

## Fluctuation consistency

历史 E3 已经包含：

```text
pred - mean(pred)
vs
target - mean(target)
```

属于 fluctuation-field consistency。

但并不是显式：

```text
std_u
std_v
```

moment matching。

因此：

```text
Explicit std-moment loss
```

仍可视为潜在不同方向。

---

## Gradient consistency

代码中存在相关 helper，

但目前没有发现一个完整、注册过的官方 v9 独立 Gradient-loss 实验。

因此：

```text
Gradient consistency
```

仍是潜在新候选。

---

## Vorticity consistency

尚无正式独立实验。

但它是：

```text
dv_dx - du_dy
```

的 derived physical summary。

如果未来研究，应避免把它重新解释为新 Feature Family。

---

# 42. FF-01 初次执行

2026-09-02 曾启动：

```text
FF-01 CLEAN Feature Conditioning / Gating
```

目标：

> 使用 TEMPORAL6 / SPATIAL4 对 CNO latent representation 做轻量 FiLM / gating。

计划三组：

```text
G0
Matched Raw-Control

G1
TEMPORAL6

G2
TEMPORAL6 + SPATIAL4
```

Condition Encoder 输入宽度统一：

```text
10 channels
```

用于保证三组参数量一致。

---

# 43. FF-01 已完成的有效工程工作

虽然正式实验被终止，但以下工程工作有效：

1. 找到了 CNO 中一个稳定的 bottleneck conditioning point；
2. 实现了固定的 G0 / G1 / G2 Feature construction；
3. 实现了 TEMPORAL6；
4. 实现了 SPATIAL4；
5. 实现了 lightweight FiLM / gating module；
6. final projection 使用 zero-init；
7. 初始 conditioning 是严格 no-op；
8. 单元测试通过；
9. Feature package shape test 通过；
10. parameter matching test 通过；
11. 官方 CNO checkpoint load 通过；
12. scorer interface smoke 通过；
    13.真实官方 CNO forward smoke 通过。

因此：

> FF-01 implementation 已部分准备完成。

---

# 44. FF-01 OOM 问题

原计划希望复用历史：

```text
T1-ID-FE-N2-30M-S20260901
```

中的：

```text
batch = 18
```

但历史 N2 的训练对象主要是：

```text
lightweight residual-head
```

而 FF-01 当前需要：

```text
full CNO backprop
```

两者显存需求不同。

正式 G0：

```text
batch = 18
```

在当前 GPU 环境发生：

```text
CUDA OOM
```

OOM 发生在：

```text
full CNO backward
```

而不是：

* checkpoint load
* scorer
* Feature construction
* FiLM insertion

随后极短 preflight 表明：

```text
batch = 8
```

可以执行 full backprop。

但：

> batch protocol 尚未经过 ChatGPT review 正式重新冻结。

因此不能自行把：

```text
batch 18
```

修改为：

```text
batch 8
```

后直接继续正式实验。

---

# 45. FF-01 当前科学状态

目前：

```text
没有正式 G0 结果
没有正式 G1 结果
没有正式 G2 结果
```

因此不能得出：

```text
Gating 有效
```

也不能得出：

```text
Gating 无效
```

更不能根据此次 aborted run 判断：

```text
Temporal 无效
Spatial 无效
```

当前正确状态：

```text
FF-01
IMPLEMENTATION PARTIALLY READY

FORMAL EXPERIMENT NOT COMPLETED

NO SCIENTIFIC CONCLUSION
```

---

# 46. 初次 FF-01 暴露出的执行问题

初次 FF-01 任务设计过重。

原流程为：

```text
实现
↓
测试
↓
smoke
↓
G0 训练 30min
↓
等待
↓
G1 训练 30min
↓
等待
↓
判断 gate
↓
可能 G2
```

这导致 Codex：

* 长时间在线等待 GPU；
* 持续 polling；
* 重复报告运行状态；
* 消耗大量 token；
* 长流程中自行处理 protocol deviation；
* ChatGPT 无法在关键决策点及时重新 review。

其中典型错误是：

```text
while summary 不存在:
    sleep
    再检查
```

然后每隔一段时间报告：

```text
仍在运行
已过十分钟
已过半程
接近预算结束
```

这些信息几乎没有技术价值。

---

# 47. Feature Engineering 后续 Codex 执行原则

以后涉及 GPU 长任务时统一采用：

> **Short Action + Detached Job + GitHub State + Explicit Review**

不允许一个 Codex task 在线陪伴完整长训练过程。

---

# 48. Phase A：Design / Preflight

Codex 只负责：

* 阅读 protocol
* 实现代码
* unit test
* architecture smoke
* forward/backward smoke
* target micro-batch memory test
* peak memory
* 确认正式训练配置

这个阶段：

```text
不跑正式 30min / 60min 实验
```

完成后：

```text
REVIEW_REQUIRED
```

等待 ChatGPT。

---

# 49. Phase B：Launch

ChatGPT 明确批准正式候选后：

Codex：

1. 启动 **一个** 候选；
2. detached；
3. 记录 PID；
4. 记录 artifact path；
5. 更新 STATUS：
   `RUNNING`
6. commit / push 必要 provenance；
7. 立即结束当前 Codex task。

禁止：

```text
持续 polling
```

禁止：

```text
在线等待训练完成
```

---

# 50. Phase C：Result Read

训练预计完成后，

由用户重新触发 Codex。

Codex：

* 检查 process state 一次；
* 读取最终 summary；
* 读取最终 metrics；
* 验证 artifact；
* 更新 handoff；
* 更新 STATUS；
* STOP。

不自动启动下一个候选。

---

# 51. Phase D：Scientific Gate

ChatGPT / Sol review：

```text
GO
STOP
REVIEW_REQUIRED
```

然后决定是否：

```text
G0
→
G1
→
G2
```

下一候选必须经过新的明确授权。

---

# 52. 后续实验的 Protocol Deviation 原则

任何正式 protocol 如果执行中发现：

* OOM
* batch 不可行
* runtime 不可行
* checkpoint 不兼容
* architecture 插入点不成立
* scorer 不兼容

Codex 可以：

> 做最小诊断，定位问题。

但不能：

> 自行修改正式 protocol 并继续跑完整实验。

正确流程：

```text
发现问题
↓
最小诊断
↓
提出建议
↓
REVIEW_REQUIRED
↓
ChatGPT 冻结新 protocol
↓
再执行
```

---

# 53. 当前 FF-01 推荐下一步

下一步不要直接启动新的 30 分钟 G0。

应该先完成：

```text
FF-01 Preflight Closure
```

目标只有一个：

> 确定当前硬件上 full-CNO FiLM training 的安全训练 batch protocol。

需要研究：

```text
micro batch
gradient accumulation
effective batch
peak GPU memory
```

只允许：

```text
≤ 2 min memory / backward preflight
```

不产生正式 dev 结果。

不跑：

```text
G1
G2
```

---

# 54. Batch Protocol 的基本判断

不需要机械继承：

```text
batch = 18
```

因为历史 N2：

```text
residual-head training
```

与 FF-01：

```text
full CNO backprop
```

计算图完全不同。

真正需要保证的是 FF-01 内部：

```text
G0
G1
G2
```

严格共享：

* micro batch
* effective batch
* gradient accumulation
* optimizer
* learning rate
* seed
* training budget
* checkpoint selection
* train/dev split

这样：

```text
G1 vs G0
```

和：

```text
G2 vs G1
```

仍然是可解释的 controlled comparison。

---

# 55. 当前 Feature Engineering 核心结论

截至目前，Feature Engineering 方向已经形成以下稳定判断。

## 结论 1

Feature Discovery 已经足够。

当前不应该继续大规模寻找新 Feature。

---

## 结论 2

Temporal 是目前最有价值的信息族。

核心为：

```text
Mean
Std
Delta
```

即：

```text
mean state
fluctuation strength
recent dynamics
```

---

## 结论 3

Spatial Gradient 有较弱但独立的信息价值。

核心为：

```text
du_dx
du_dy
dv_dx
dv_dy
```

---

## 结论 4

以下 Feature 更适合作为物理摘要，而不是新的信息族：

```text
u'
v'
TKE_proxy
vorticity
```

---

## 结论 5

强 CNO 没有完全利用 Temporal / Spatial information。

Frozen CNO Probe 中仍存在：

```text
Rel-L2 / MVPE residual signal
```

---

## 结论 6

Feature information value 与 Fusion value 必须分开讨论。

当前统一判断：

```text
FEATURE_VALUE_POSITIVE_BUT_FUSION_HISTORY_NEGATIVE
```

---

## 结论 7

简单 output residual correction 容易：

```text
改善 point-wise error
+
破坏 temporal fluctuation
```

典型表现：

```text
Rel-L2 ↓
MVPE ↓
TKE ↑
```

---

## 结论 8

因此 Feature Fusion 的主要目标不是：

```text
让 Feature 更强地修 prediction
```

而是：

> **让 Temporal / Spatial prior 更安全地影响 representation，同时保护 TKE / fluctuation structure。**

---

# 56. 当前研究路线图

```text
Feature Engineering
        │
        ├──────────────────────┐
        │                      │
        ▼                      │
Feature Discovery              │
        │                      │
        ├── Raw                │
        │                      │
        ├── Temporal           │
        │     ├── Mean         │
        │     ├── Std          │
        │     └── Delta        │
        │                      │
        ├── Spatial            │
        │     └── 4 Gradients  │
        │                      │
        └── Derived            │
              ├── TKE Proxy    │
              ├── Vorticity    │
              └── u'/v'        │
        │                      │
        ▼                      │
Feature Value Probe            │
        │                      │
        ├── Data Diagnostic    │
        │                      │
        ├── PERSIST Probe      │
        │       Positive       │
        │                      │
        └── Frozen CNO Probe   │
                │              │
                ├── Rel signal │
                ├── MVPE signal│
                └── TKE tradeoff
        │
        ▼
Feature Discovery CLOSED
        │
        ▼
Feature Fusion
        │
        ├── FF-00 Protocol Freeze
        │       DONE
        │
        ├── FF-01 Gating
        │       Implementation partially ready
        │       Formal experiment pending
        │
        ├── FF-02 Multi-Branch
        │       NOT STARTED
        │
        └── FF-03 Feature-aware Objective
                NOT STARTED
```

---

# 57. 当前项目状态

```text
Feature Engineering
    ACTIVE

Feature Discovery
    CLOSED

Feature Value Study
    CLOSED

Feature Fusion
    ACTIVE
```

当前 Fusion 状态：

```text
FF-00
    COMPLETED

FF-01
    IMPLEMENTATION PARTIALLY READY
    FORMAL RESULT NOT AVAILABLE

FF-02
    NOT STARTED

FF-03
    NOT STARTED
```

当前没有可靠 FF-01 正式实验结论。

---

# 58. 下一步

Feature Engineering 当前推荐下一步：

```text
FF-01 Preflight Closure
```

目标：

```text
freeze full-CNO training memory protocol
```

之后：

```text
Launch G0 only
↓
STOP
↓
Read G0
↓
Review
↓
Launch G1 only
↓
STOP
↓
Review
↓
G2 only if explicitly authorized
```

不要重新采用：

```text
G0 → wait → G1 → wait → G2
```

的单会话长链执行方式。

---

# 59. 一句话总结

RealPDE Track 1 Feature Engineering 当前最核心的认识是：

> **值得研究的 Feature 已经基本找到，主要是 Temporal 的 mean/std/delta 和 Spatial 的四个 pixel gradients。真正未解决的问题已经不是“还能造什么 Feature”，而是“如何让模型利用这些 prior，同时不破坏 TKE 和 temporal fluctuation structure”。**

