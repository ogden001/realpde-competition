# REALPDE Clean Baseline V1 — 2026-09-24 分析与后续比较规则

Status: `FROZEN_RESEARCH_BASELINE / REVIEW_REQUIRED`

本文件总结 2026-09-24 完成的 `REALPDE_CLEAN_BASELINE_V1` 训练过程、结果解释，以及从本次实验开始统一采用的后续比较规则。

原始运行报告：

- `docs/clean_baseline_v1/results/20260924_run1/RUN_REPORT.md`
- 证据目录：`docs/clean_baseline_v1/results/20260924_run1/evidence/`

## 1. 为什么要建立这条新 baseline

过去数日的一部分实验沿用了 colleague all81/full-data 或包含相同 trajectory 的 Dev16 评估。那些实验仍可用于同 lineage 的局部机制 A/B、排查实现问题或产生新假设，但不能继续被当成 unseen-generalization 的 clean 证据。

本次 baseline 的目的不是再造一个 SOTA recipe，而是建立一把后续研究统一使用的干净尺子：

- Train / Seen Dev / Unseen AoA Holdout trajectory-level 隔离；
- 固定 stride=1 dense temporal sampling；
- 固定 Stage1 CNO + Stage2 frozen-backbone residual 主干；
- 只用 Seen Dev 的 point prediction 指标选 checkpoint；
- Holdout 只在 Stage1 + Stage2 完成后一次性诊断；
- 不让 SPS / runtime 参与科研模型选择。

从本次实验开始，所有新的 predictive modeling 想法，以及过去几天希望继续推进的实验方向，都必须明确回答：

> 相对 `REALPDE_CLEAN_BASELINE_V1`，在同一 clean split、sampling、checkpoint-selection 规则下，是否有增益？

旧协议上的结果只能作为“历史线索 / 假设来源”，不能直接继承为 clean baseline 上的有效结论。

## 2. 冻结数据与采样协议

81 条可用 released real-PIV trajectories：

- Train51：AoA 0/5/15/20；
- Seen-Dev12：AoA 0/5/15/20 各 3 条；
- Unseen-AoA10 Holdout18：全部且仅全部 AoA=10；
- `7575_0.h5` 排除；
- 三组 trajectory-level 完全不重叠。

训练协议：

- Past20 -> Future20；
- sub_sample=2 / P00；
- Train stride=1，枚举所有合法 temporal starts；
- global deterministic shuffle，seed=41；
- batch=8；
- Dev/Holdout stride=20，start=0，shuffle=False。

Train51 共 `41,317` 个合法 stride=1 windows。

Stage1 共消费 `69,784` samples，约等于 `1.69` 个完整 dense epoch。

Stage2 共消费 `307,200` samples，约等于 `7.44` 个完整 dense epoch；sampling audit 证明 `41,317 / 41,317` 个合法窗口均至少被看到一次。

## 3. Stage1：CNO 训练结果

Stage1 从官方 `sim_real_cno.pth` 初始化，保持 colleague-80 CNO architecture / loss 不变。

Seen-Dev12：

| Update | Rel-L2 | TKE | MVPE | point_score |
|---:|---:|---:|---:|---:|
| 0 | 0.448323 | 7.028707 | 0.494521 | 61.3386 |
| 2,000 | 0.138621 | 1.320505 | 0.101411 | 82.9747 |
| 4,000 | 0.114715 | 0.900639 | 0.090130 | 86.4045 |
| 6,000 | 0.103866 | 0.810984 | 0.081986 | 87.4249 |
| **8,000 best** | **0.099606** | **0.778031** | **0.080290** | **87.7966** |
| 8,723 final | 0.099932 | 0.792904 | 0.080300 | 87.6637 |

结论：

1. Stage1 真实 PIV fine-tune 有大幅有效学习，不是小修正；
2. 最优点已经出现在约 8k，8k -> 8.7k 出现轻微回落，尤其 TKE；
3. 对不改变 backbone / Stage1 loss / data regime 的后续实验，不应机械重跑 Stage1，可直接复用 Stage1 best；
4. Stage1 继续单纯加长训练目前没有证据支持。

Stage1 best checkpoint SHA256：

`6aec4edb5e42746080e19baaafa2d6b719bb57befb1ff9035ece7889ec280036`

## 4. Stage2：Residual 的真实贡献

Stage2 从 Stage1 best 开始，冻结 CNO；ResidualCorrector3D 最后一层 zero-init，因此 step0 与 Stage1 best 严格对齐。

Seen-Dev12：

| Checkpoint | Rel-L2 | TKE | MVPE | point_score |
|---|---:|---:|---:|---:|
| Stage2 step0 | 0.099606 | 0.778031 | 0.080290 | 87.7966 |
| **Stage2 best@37000** | **0.078363** | **0.507732** | **0.064408** | **90.9543** |
| Stage2 final@38400 | 0.078345 | 0.508196 | 0.064403 | 90.9498 |

Stage2 best 相对 step0：

- Rel-L2 raw error 约下降 21%；
- TKE raw error 约下降 35%；
- MVPE raw error 约下降 20%；
- point_score 提升约 +3.16。

因此 frozen-backbone residual correction 是本次 baseline 中已明确成立的大收益组件。

Stage2 best checkpoint SHA256：

`1536fe02e11fc0dc991a38cb1489d0be0752ec8149be7b9f1542a4cb81c2f975`

## 5. Unseen AoA10：Residual 的泛化边界

AoA10 Holdout18 从未参与训练、checkpoint selection 或 alpha tuning。

| Checkpoint | Rel-L2 | TKE | MVPE |
|---|---:|---:|---:|
| Stage2 step0 | 0.096635 | 0.714621 | 0.100787 |
| **Stage2 best** | **0.085567** | **0.523197** | 0.101422 |
| Stage2 final | 0.085548 | 0.523347 | 0.101495 |

Stage2 best 相对 step0：

- Rel-L2 改善约 11.45%；
- TKE 改善约 26.79%；
- MVPE 恶化约 0.63%。

解释：

- Residual 的 Rel/TKE correction 明显具有跨未见 AoA 的真实泛化；
- MVPE 的局部峰值修正没有表现出同样稳定的泛化，虽然恶化幅度很小；
- 后续任何 augmentation / backbone / residual 改动，不能只看 Seen Dev aggregate，需要同时关注 AoA10 上的 MVPE protection。

## 6. Late-horizon / “尾帧翘起”的新解释

Seen Dev Stage2 step0：

- F1 frame_rel_l2 = `0.08953`
- F20 = `0.11477`

Stage2 best：

- F1 = `0.03934`
- F20 = `0.09595`

AoA10 Holdout：

- step0 F1 / F20 = `0.07899 / 0.11649`
- Stage2 best F1 / F20 = `0.03833 / 0.10620`

因此 Residual 并没有把 F20 训练坏。真正现象是：

- 前期 horizon correction 极强，F1 可改善约 50%；
- 后期 horizon correction 明显弱，F20 仅改善约 9%（Holdout）到约 16%（Seen Dev）。

所以“尾部翘起”现在优先解释为：

> **Residual correction benefit 随预测 horizon 增长而显著衰减。**

这比“F18-F20 存在独立 bug”更符合 clean evidence。

后续 late-horizon 研究的目标应是提升 F15-F20 correction effectiveness，同时保护整体 Rel/TKE/MVPE，而不是继续无目的追查一个神秘尾帧异常。

## 7. Stage2 训练预算的效率结论

Seen-Dev point_score：

- 10k: 90.6065
- 20k: 90.7530
- 22k: 90.9369
- 30k: 90.9341
- 33k: 90.9517
- 35k: 90.9526
- 37k: 90.9543
- 38.4k: 90.9498

20k 以后已经进入明显的边际收益区；22k -> 37k 只增加约 `0.017` point_score。

后续建议训练预算：

| 用途 | Stage2 budget |
|---|---:|
| bounded screening | 10k |
| 正式机制验证 | 20k–22k |
| 明显有效候选复核 | 30k |
| 最终 merge / fully-converged candidate | 38.4k |

这不是新的科学变量，而是后续研发资源管理规则。任何正式 claim 仍需与同预算 matched baseline 比较。

## 8. 运行过程与工程结论

本次实际执行：

- GPU: RTX 3090 Ti 24 GB；
- Stage1: ~1h27m；
- Stage2: ~3h31m；
- Holdout: ~4m；
- total: ~5h03m；
- OOM: NO；
- environment recovery: NONE；
- 19 tests PASS。

本次执行 commit 为 `65e4f0f...`，发生在后续 RAM preload / persistent workers / b8-vs-b16 runtime profiling 代码之前。因此本次 baseline 本身没有使用这些后加的工程优化，也没有 runtime-profile 数据。

这不影响科研 baseline 的成立。后续可以使用新的工程 profile 提速，但必须保证科学 protocol 不变；若 batch/LR/update exposure 发生变化，则必须显式登记为不同 training profile。

## 9. 从现在开始的统一比较规则

### 9.1 新想法

所有新的 predictive idea 在进入高成本训练前必须登记：

- reference baseline = `REALPDE_CLEAN_BASELINE_V1`；
- 唯一计划变量；
- 是否修改 Stage1、Stage2、sampling、loss、augmentation 或 backbone；
- matched training budget；
- Seen-Dev12 对比；
- 是否需要在方案锁定后查看 AoA10 Holdout18。

默认不允许同时修改多个研究变量后再把收益归因给其中一个。

### 9.2 过去几天已做实验

过去几天的实验保留，不删除、不否定，但按以下方式解释：

| 历史方向 | 旧结果当前用途 | 后续要求 |
|---|---|---|
| late-horizon ramp / tail loss geometry | 局部机制线索 | 在 clean baseline 上 matched re-test 后才能进入结论 |
| random phase / temporal sampling | 候选 sampling 假设 | 以 frozen stride=1 baseline 为 control 重新对比 |
| AoA rotation / AoA mean-field augmentation | 泛化假设来源 | 必须用 Seen Dev + unseen AoA10 双重评价 |
| stronger backbone | 候选大台阶方向 | 从 clean split 重新训练/比较，不能引用 overlap Dev 的绝对优势 |
| Pareto TKE / residual loss variants | residual objective 假设 | 从 Stage1 clean best matched 启动，对比 baseline residual |
| end-to-end / joint fine-tune | 组合策略假设 | 必须以 clean Stage1+Stage2 为初始化/对照，禁止用旧 overlap Dev 直接 GO |
| all81/full82 residual experiments | competition/mechanism evidence | 不作为 clean generalization claim |

旧实验如果是在 all81/all82 或与 Dev trajectory overlap 条件下完成：

- 可用于说明“这个机制值得不值得重新测”；
- 可用于 debug、方向排序、实现复用；
- 不可直接写成“已经在 clean dev 上有效”；
- 不可直接触发最终 merge。

### 9.3 两套坐标系必须分开

以后项目同时维护两套 anchor：

1. **Research anchor**
   - `REALPDE_CLEAN_BASELINE_V1`
   - 用于 predictive modeling、机制判断、实验选择。

2. **Online submission anchor**
   - 当前 Codabench SOTA / 最新正式 submission；
   - 用于最终 merge、full-data refit、SPS、runtime 和线上收益判断。

Research baseline 不替代线上 SOTA；线上 SOTA 也不能替代 clean research control。

## 10. 当前优先研究问题

当前只保留少数结构性问题：

1. **Late-horizon correction**：解决 F1 改善约 50%，但 F20 仅改善约 10% 左右的问题；
2. **Unseen-AoA generalization**：保护已经很强的 Rel/TKE 泛化，同时解决 MVPE 不稳定；
3. **大台阶 backbone / data strategy**：只做有足够机制依据的 bounded clean comparison；
4. **最终 submission layer**：SPS、precision/runtime、full-data refit 独立于 clean research 选择。

默认停止：

- 无目的 loss-weight 微调；
- 围绕 0.x% Dev 增益反复长训；
- 在 overlap Dev 上继续扩展实验树；
- 因 GPU 空闲而机械启动 full-length experiment。

