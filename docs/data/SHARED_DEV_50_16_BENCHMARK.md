# Track 1 共享 50/16 Dev 基准与关键实验结果

> 用途：与并行参赛同事对齐同一个 Dev 数据集、评分口径和参考结果，方便直接比较实验。
>
> 状态：`FROZEN_SHARED_DEV_REFERENCE`  
> 更新时间：2026-09-16

## 0. 先看这里：实验代号与术语速查

> **重要：下面很多代号是本项目内部实验编号，不是论文或学术界通用缩写。** 以后新增实验时，也应在本节或实验表中同时写出“代号 + 中文含义”，不要只留编号。

| 写法 | 含义 | 一句话解释 |
|---|---|---|
| `Historical E0` | 历史 E0 基线 | `E0` 是项目内部 loss baseline 编号；这里特指官方 `sim_real_ft` warm-start、`MSE + 0.05*TKE` 的历史参考实验。 |
| `P0-A` | P0-A 特征扩展输入方案 | 项目内部特征工程版本号；把 **20 个可部署特征**直接扩展为 CNO 输入，而不是只用原始 `u/v/p`。 |
| `N2` | N2 多目标 Loss 配方 | 项目内部 loss 版本号；当前权重为 `MSE=1.0, TKE=0.05, Rel=0.027514, MVPE=0.009757`。 |
| `DW-01` | Dense-Window Experiment 01 | 稠密时间窗训练实验 01；核心变量是训练窗口采样方式。 |
| `Dense-All` | 全合法时间窗训练 | 对每条 Train trajectory 使用所有满足 Past20 + Future20 的合法起点，而不是固定 `stride=20` 的稀疏窗口。 |
| `MF-01` | Mean / Fluctuation Experiment 01 | 均值 / 脉动分解实验 01；把流场拆成 mean component 与 fluctuation component 建模。 |
| `MF-CNO` | Mean / Fluctuation CNO | 将 Mean / Fluctuation 分解嵌入 CNO 的模型版本。 |
| `Vorticity` | 涡量辅助监督 | 在训练目标中加入由 `u/v` 导出的 vorticity（涡量）约束，强化局部旋转/剪切结构学习。 |
| `Stage-B extra Rel` | 第二阶段追加 Rel-L2 优化 | SOTA-V2 在主训练之后用更低 LR 再训练一段，并额外加入 `Rel-L2` loss。 |
| `SOTA-V2` | 当前集成方案 Version 2 | 当前把 Dense-All、P0-A、MF-CNO、N2、Vorticity、Stage-B extra Rel 集成在一起的 competition-oriented 版本。 |
| `@32500` / `@30k` | checkpoint update 数 | `@32500` = optimizer update 32,500；`30k` = 30,000 updates，不是 epoch。 |
| `CLEAN` | 干净离线实验 | 初始化和本地 PIV 学习满足 Train/Dev 隔离，可用于方法有效性的离线判断。 |
| `OFFICIAL_WARM_START` | 官方 real-PIV warm-start | 使用官方 `sim_real_ft` 初始化，适合比赛冲榜，但不作为严格 clean causal evidence。 |
| `NO_GO` | 当前不继续 | 该独立方向没有达到预注册门槛，暂不继续投入。 |
| `PARK` | 有弱信号但暂存 | 有一定正向信号，但不足以单独进入昂贵的 SOTA merge，先保留。 |

其中 `E0 / P0-A / N2 / DW-01 / MF-01` 都应理解为**项目内部版本标签**。真正要记住的是右侧的机制含义，而不是字母本身。

## 1. 冻结数据划分

当前离线实验统一使用同一份 frozen manifest：

- 数据总池：82 条本地 PIV trajectories。
- Split：`50 Train / 16 Dev / 16 locked-final`。
- **按完整 trajectory 划分，不按 window 随机拆分。**
- Seed：`20260901`。
- Manifest：`artifacts/loss_optimization_v9_20260901_run1/evidence/manifests/id_seed20260901.json`。
- Manifest SHA-256：`42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`。
- 日常训练和选模只使用 Train / Dev；locked-final 不参与调参、early stop 或架构选择。

**协作者不要重新 random split。** 对齐的核心是复用相同 trajectory membership。

### 1.1 Dev 16 条 trajectory

```text
10125_0.h5
3750_0.h5
26700_0.h5
13950_5.h5
20325_5.h5
8850_5.h5
8850_10.h5
11400_10.h5
24150_10.h5
16500_10.h5
22875_15.h5
11400_15.h5
25425_15.h5
20325_20.h5
8850_20.h5
3750_20.h5
```

数据画像：`9 IN_DISTRIBUTION / 3 BOUNDARY / 4 OOD_LIKE`。该标签仅用于分布诊断，不用于重划分。

## 2. Dev 评估协议

统一使用：

- `T_in = 20`
- `T_out = 20`
- `stride = 20`
- `window_mode = fixed`
- `start = 0`
- 合法条件：`start + 40 <= frames`
- `sub_sample = 2`
- `include_pressure = false`；预测 pressure 固定为 0
- Dev 不 shuffle

一致性检查：

| Pool | Window 数 |
|---|---:|
| Train canonical fixed windows | 2052 |
| Train Dense-All windows | 40488 |
| **Dev fixed windows** | **659** |

如果 Dev 不是 **16 trajectories / 659 windows**，不要继续比较模型结果，先排查数据协议。

## 3. 评分口径

统一使用 Track 1 starting kit v9 官方 scorer：

- scorer SHA-256：`a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`
- 共享结果统一报告 Dev raw errors：`Rel-L2 / TKE / MVPE`
- 三项均为 **越低越好**。

不要与 Codabench 上 0–100 的 `rel_l2_score / tke_score / mvpe_score` 混用；leaderboard subscore 是越高越好。

## 4. 当前共同 Dev SOTA

当前 50/16 **SOTA-V2（集成方案 Version 2）** 配方：

`Dense-All + P0-A + MF-CNO + N2 + Vorticity + Stage-B extra Rel`

翻译成人话就是：

1. **Dense-All**：Train 使用全部合法 Past20→Future20 时间窗；
2. **P0-A**：20 个可部署特征直接扩展到 CNO 输入；
3. **MF-CNO**：使用 Mean / Fluctuation（均值 / 脉动）分解版本 CNO；
4. **N2**：使用 MSE、TKE、Rel-L2、MVPE 的多目标 loss；
5. **Vorticity**：增加涡量辅助监督；
6. **Stage-B extra Rel**：主训练结束后降低学习率，再追加一段强化 Rel-L2 的训练。

关键配置：

- warm-start：官方 `sim_real_ft/sim_real_cno.pth`
- warm-start SHA-256：`82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61`
- effective batch：8
- Stage A：update `1..30000`，LR `1e-5`
- Stage B：update `30001..35000`，LR `3e-6`，额外 `0.027514 * Rel-L2`
- N2：`MSE=1.0, TKE=0.05, Rel=0.027514, MVPE=0.009757`
- Vorticity weight：`15.5385751724`

### 4.1 SOTA-V2 Dev curve

| Update | Rel-L2 ↓ | TKE ↓ | MVPE ↓ |
|---:|---:|---:|---:|
| 7,500 | 0.125148 | 0.510103 | 0.108036 |
| 15,000 | 0.111479 | 0.484161 | 0.087232 |
| 25,000 | 0.106151 | 0.471133 | 0.082209 |
| 30,000 | 0.105989 | 0.474373 | 0.090337 |
| 31,000 | 0.100937 | 0.470932 | 0.078285 |
| **32,500** | **0.099935** | **0.469293** | **0.075778** |
| 35,000 | 0.099978 | 0.471441 | 0.075749 |

当前统一性能锚点建议使用：

> **SOTA-V2 @32,500：Rel-L2 = 0.099935 / TKE = 0.469293 / MVPE = 0.075778**

32.5k 是当前 full-data refit 训练强度映射使用的 primary Dev sweet spot。

## 5. 关键探索实验参考

以下实验都在同一 frozen 50/16 Dev 上评估。不同初始化、cleanliness 或训练预算的绝对值可作性能参考，但**不能跨 family 做单变量因果归因**。

| 实验 / checkpoint | Rel-L2 ↓ | TKE ↓ | MVPE ↓ | 结论 |
|---|---:|---:|---:|---|
| Historical E0 @7498（历史官方 warm-start 基线） | 0.168923 | 0.538475 | 0.136146 | `sim_real_ft` + `MSE + 0.05*TKE` 的历史 competition reference |
| P0-A + N2 @约4100（20 特征扩展输入 + 多目标 Loss） | 0.156106 | 0.553362 | 0.125436 | CLEAN 早期有效路线参考 |
| **DW-01 Dense-All @30k（全合法时间窗训练）** | **0.110092** | **0.479399** | **0.080831** | 强信号；2052 → 40488 train windows，后进入 SOTA-V2 |
| MF-01 @1500（Mean / Fluctuation 均值-脉动分解） | 0.188409 | 0.645156 | 0.160552 | 对 Direct：Rel/MVPE 小幅改善，TKE 恶化；standalone `NO_GO` |
| Vorticity @12000（涡量辅助监督） | 0.125012 | 0.510238 | 0.108193 | 对 matched control 三项均小幅改善，但 long-run gate 为 `PARK` |
| **SOTA-V2 @32500（当前集成方案 V2）** | **0.099935** | **0.469293** | **0.075778** | 当前共同 Dev benchmark |

几个值得共享的机制结论：

- **Dense-All（全合法时间窗训练）**：目前最明确的训练数据策略信号之一。Dev 始终固定 659 windows，训练窗口由 2052 扩展到 40488（`19.731×`）。
- **Vorticity supervision（涡量辅助监督）**：matched @12000 相对 control `0.130150/0.518528/0.110468`，改善约 `+3.95% / +1.60% / +2.06%`，但长期稳定性不足，standalone `PARK`。
- **Mean / Fluctuation decomposition（均值 / 脉动分解）**：早期 matched 实验 Rel/MVPE 有约 2%–3% 改善，但 TKE 变差，standalone `NO_GO`；后续仅作为 SOTA-V2 集成配方的一部分保留。
- **Structured Temporal Dynamics（显式时间动力学结构）**：Latent Temporal Conv 仅弱信号，Temporal Attention 基本无增益；当前不认为显式 Future20 temporal architecture 是主要瓶颈。
- **Random Phase（随机时间相位 / 起点增强）**：旧实验存在 shuffle confound，状态是 `INVALID_COMPARISON`，不能得出“Random Phase 无效”的结论。

## 6. 双方以后统一汇报格式

每个实验至少共享：manifest SHA、Dev 是否为 16 trajectories / 659 windows、初始化 checkpoint、recipe、训练预算，以及三项 raw error。

同时，**Experiment 字段不要只写代号**。建议写成：

```text
Experiment: DW-02 — Dense-All + Random Phase（全合法窗口 + 随机相位增强）
```

而不是只写：

```text
Experiment: DW-02
```

完整模板：

```text
Experiment: <代号> — <中文/英文机制说明>
Manifest SHA: 42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347
Dev: 16 trajectories / 659 windows
Init:
Recipe:
Budget:
Rel-L2:
TKE:
MVPE:
Matched baseline:
Delta vs baseline:
Notes:
```

判断某个技术因素是否有效时，优先使用相同 family 的 matched control；当前 SOTA-V2 主要用于性能锚定，不应被当成所有探索实验的因果 baseline。

## 7. 关键证据入口

- 数据划分：[DATASET_PROFILE.md](DATASET_PROFILE.md)
- 全局实验注册表：[../track1_experiment_registry.md](../track1_experiment_registry.md)
- Dense-All：[../experiments/dw01_dense_all_20260907/README_FOR_CHATGPT.md](../experiments/dw01_dense_all_20260907/README_FOR_CHATGPT.md)
- Structured Temporal / Vorticity：[../modeling/structured_temporal_dynamics_closeout.md](../modeling/structured_temporal_dynamics_closeout.md)
- Random Phase：[../experiments/rw00_rw01_random_phase_20260907/README_FOR_CHATGPT.md](../experiments/rw00_rw01_random_phase_20260907/README_FOR_CHATGPT.md)
- SOTA-V2 50/16：[../sota迭代/reviews/sota_v2_integrated_50_16_20260915/README.md](../sota迭代/reviews/sota_v2_integrated_50_16_20260915/README.md)

## 8. 一句话版本

> **共同 Dev = frozen manifest 的 16 trajectories / 659 fixed windows；当前性能锚点 = SOTA-V2@32.5k：Rel-L2 `0.099935` / TKE `0.469293` / MVPE `0.075778`。**
