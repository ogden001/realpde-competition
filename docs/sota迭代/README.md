# 线上 SOTA 迭代

## 1. 目标

本目录用于 Track 1 的**每日线上 SOTA 冲刺**。

核心目标不是独立研究某一个技术方向，而是利用每天有限的 Codabench 提交机会，把截至当天已经论证清楚、具备正向证据且能够稳定部署的优化项合并到当前最佳候选中，完成训练、SPS 优化、打包和线上提交，持续刷新正式分数。

各技术方向仍在自己的目录中完成研究和验证，例如 Modeling、Loss、Feature Engineering、Training、Inference。本目录只负责**跨方向收口与提交**。

### “merge SOTA” 的固定含义

当用户明确说 **“merge SOTA”**、**“合并 SOTA”** 或要求“把已验证有效策略合起来训练并提交”时，默认先进入 **SOTA Merge Worthiness Review**，而不是直接启动训练。

一次完整 SOTA merge 通常意味着约 **4～6 轮 ChatGPT/Codex 协作 + 4～6 GPU 小时**，还会消耗打包、提交和结果复盘成本。因此“存在正向证据”不等于“值得现在 merge”。Sol 必须先判断当前候选是否有足够明显的预期线上收益，只有结论为 **`MERGE_WORTHY`** 时，才进入 **SOTA Merge Execution Mode**。

固定判断流程：

```text
用户提出 merge SOTA
        ↓
列出当前线上 SOTA 已包含的策略变量
        ↓
列出本轮真正新增 / 替换的独立变量
        ↓
评估每个变量的证据强度、指标影响、兼容性和线上迁移风险
        ↓
估计组合后是否有“明显线上收益”的合理预期
        ↓
MERGE_WORTHY ?
   ├─ NO  → SKIP_MERGE，直接提醒用户不值得烧本轮资源
   └─ YES → 进入 SOTA Merge Execution Mode
```

### Merge Worthiness Gate

Sol 在启动任何完整 merge 前必须回答：

1. **当前线上 SOTA 的变量是什么？** 至少拆清 backbone / formulation / feature / loss / training / inference-SPS 等主要变量，避免把已经在 SOTA 中的组件误算成新收益。
2. **本轮真正新增或替换什么？** 必须指出独立策略变量，而不是笼统写“综合优化”。
3. **这些变量的证据强度如何？** 区分 ONLINE_KEEP、明确 matched positive、weak signal、conflicting signal、未经验证。
4. **它们预计影响哪些线上指标？** 说明 Rel-L2 / TKE / MVPE / Time / SPS 中哪几项可能获益、哪几项存在风险。
5. **为什么值得付出完整 merge 成本？** 必须给出组合后存在明显线上提点的合理依据，而不是“反正训练看看”。

默认决策：

- **`MERGE_WORTHY`**：存在一个大台阶变量，或多个证据较强、彼此兼容的增量组合，预期值得消耗一次完整 50/16 → full-data → package → Codabench 周期。
- **`SKIP_MERGE`**：只有孤立小优化、weak signal、预期 Final 仅微涨、线上迁移高度不确定，或当前组合相比继续 Exploration 没有足够高的资源回报率。此时即使用户刚说“merge SOTA”，Sol 也应主动提醒**这轮不值得做**。

特别规则：

- 离线 `+1%` 左右的小收益、单一弱信号、仅某一个次要指标改善，**默认不足以单独触发完整 SOTA merge**。
- 一个变量若离线收益很大，可以单独构成 `MERGE_WORTHY`，不要求机械凑够多个变量。
- 多个小变量只有在证据较强、机制兼容、预计能够形成有意义的组合收益时才值得合并。
- Merge Worthiness Review 是资源决策，不是新的研究 Campaign；只使用现有证据快速判断，不为证明“值得 merge”额外发明大量实验。

通过 Gate 后，固定目标为：

> 把此前已经验证有效、且经 Worthiness Review 判断值得本轮投入的策略直接编码到同一个 submission recipe 中。先在冻结的 50 Train / 16 Dev 上用同一 recipe 训练，默认约 2 小时，与历史 SOTA 的 50/16 结果直接比较；结果整体 OK 后立即启动全量 competition 训练。随后做最小 SPS / package smoke 并尽快 Codabench。线上失败或回退是可接受的实验结果，不应因为追求离线证据完美而长期阻塞提交。

执行顺序：

```text
MERGE_WORTHY
        ↓
最小代码合并
        ↓
固定 50/16 训练约 2 小时
        ↓
与历史 SOTA 50/16 直接比较
        ↓
结果 OK → 同 recipe 全量训练
        ↓
轻量 sanity check
(loss / NaN / Inf / 显存 / 速度 / 少量 checkpoint 指标)
        ↓
SPS / bounds
        ↓
最小 package smoke
        ↓
Codabench 提交
        ↓
根据线上结果 KEEP / ROLLBACK / 再研究
```

SOTA Merge Execution Mode 下的硬规则：

1. **50/16 是快速可比 Gate，全量训练是最终 submission 主路径。** 不跳过历史可比性，也不把 50/16 扩展成新的长期研究。
2. **不把 merge 重新变成 research。** 已验证策略直接合并；50/16 只用于回答“相对当前 SOTA 是否值得 full run”。
3. **优先时间效率。** 默认 50/16 训练约 2 小时；若指标趋势整体 OK，立即进入 full-data，不要求完整 ablation、trajectory Gate、机制诊断或大规模测试矩阵。
4. **提交失败可以接受。** Codabench 是真实实验的一部分；通过 Worthiness Gate 后，不以“必须确保线上提升”为前提才允许提交。
5. **只保留必要安全检查。** shape / finite / checkpoint 可加载 / prediction path / package clean-room 等直接影响提交正确性的检查必须做；不为一次 merge 临时建设通用框架、扩展测试矩阵或额外研究流水线。
6. **时间约束只能加速已通过 Worthiness Gate 的 merge。** “今晚 merge SOTA”不能绕过收益论证；若 `SKIP_MERGE`，应把 GPU 时间留给更高赔率的 Exploration 或其它候选。

## 2. 基本原则

1. **先判断值得不值得 merge，再判断怎么 merge。** 完整 merge 是高成本动作，不能把“有正向证据”直接等同于“马上提交”。
2. **只合并已经有证据的改动。** 不在 submission candidate 上临时加入未经验证的新结构、新 Loss 或新 Feature。
3. **当前线上最佳结果始终作为锚点。** 新版本必须明确说明相对上一版新增了什么。
4. **不为形成 submission 而形成 submission。** 当天没有足够高赔率的新增变量时，允许明确 `SKIP_MERGE`，继续 Exploration。
5. **SPS 是每次提交前的固定步骤。** SPS 优化与模型训练分开处理，不污染模型实验结论。
6. **避免无意义浪费提交次数，但不追求通过 Gate 后的证据完美。** 必要 local smoke 通过即可进入 submission review；线上失败或回退可以作为真实实验结果记录。
7. **线上结果用于确认整体效果。** 不把 Codabench 当作高频超参数搜索器，但允许在通过 Worthiness Gate 后用一次真实提交验证完整组合。

## 3. 每日流程

常规研究收口流程：

```text
各方向最新实验结论
        ↓
汇总当前 SOTA 变量与候选新增变量
        ↓
Merge Worthiness Review
        ↓
SKIP_MERGE ← 不值得
        │
        └→ MERGE_WORTHY
                ↓
确定单一 SOTA candidate recipe
                ↓
固定 50/16 约 2h 快速对比
                ↓
GO_FULL
                ↓
全量 competition refit / continuation
                ↓
SPS / bounds 优化
                ↓
最小 official/package smoke
                ↓
Codabench 提交
                ↓
记录线上结果并更新下一轮基线
```

## 4. 当前线上锚点

截至 2026-09-05：

- 当前最好正式结果：**`76.694784`**
- Candidate：`P0-A + N2 + CNO + full@43260 + v5 Adaptive Uncertainty Head@1400`
- Bounds：`half_width_uv = 0.0025 + sigma`，pressure half-width `0`
- Backbone checkpoint SHA256：`50b692e236d5df9285a5cee976a51e3457a7eeed0f87d55b6568745077645d71`
- Submission ZIP SHA256：`3285ad3a424988ab35061337ca836c23b5f7db04773246167da3a9f8eaa2178a`
- Codabench：Rel-L2 `93.434384` / TKE `77.588799` / MVPE `92.519563` / Time `87.066646` / SPS `29.519724`
- 相对 2026-09-04 SOTA `76.149726`：Final **`+0.545058`**，主要来自 SPS **`+1.974665`**；prediction 主体保持不变。

正式提交历史见：[`../submission_log.md`](../submission_log.md)。

## 5. 当前 SOTA 主线

当前优先路线：

**P0-A features + N2 loss + CNO + full-data late-training continuation + learned adaptive uncertainty bounds**。

当前 backbone 仍是 full@43260。2026-09-05 submission 只新增 v5 base Adaptive Uncertainty Head@1400，package prediction parity 为 `max_abs_diff=0.0`。线上 Rel-L2 / TKE 与上一版完全相同，MVPE 仅有舍入级变化，而 SPS 从 `27.545059` 提升到 `29.519724`，Final 从 `76.149726` 提升到 `76.694784`。因此 Adaptive Uncertainty 已从离线信号升级为**线上验证有效的 SOTA 组件**。

当前不作为 SOTA 主线自动合并：

- Residual Corrector：V5 canonical Gate 中 Rel-L2/MVPE 改善，但 TKE aggregate 恶化 `5.06%` 且 6/16 trajectories 超过保护阈值，当前 `NO-GO / PARKED_SIGNAL`；
- H1 / Point hybrid：Rel-L2、MVPE 有信号，但 trajectory-level TKE 保护不足；
- 新 Feature Fusion：底层信息有价值，但强 CNO 上稳定 fusion 收益尚未建立；
- 新 Loss：N2 仍是当前经过验证的主力 objective。

## 6. 记录规则

每天完成一次正式提交后，在本文件追加一个简短记录：

```text
日期：
Candidate：
相对上一版新增：
训练 checkpoint：
SPS / bounds：
Codabench：Final / Rel-L2 / TKE / MVPE / Time / SPS
结论：KEEP / ROLLBACK
下一轮主要问题：
```

如果某次用户提出 merge 但 Worthiness Review 结论为 `SKIP_MERGE`，不需要制造 submission 记录；只需在相关研究方向或 NEXT_ACTION 中保留为何不值得 merge 的简短结论。

详细实验数据继续记录在对应优化方向文档和 `track1_experiment_registry.md`，本目录不重复保存大段实验报告。

## 2026-09-04 Overnight full-data continuation

实验状态：`DONE` / `REVIEW_REQUIRED`。基于 `695cf27` 的 fixed P0-A + N2 CNO，从 full-data update `15300` 继续训练至 `43260`；固定 82 trajectories / 3383 windows、seed `20260901`、LR `1e-5`、micro-batch `4`、accumulate `2`。Preflight 与 2-update smoke 均通过，未访问 locked-final/private-test、SPS 或 Codabench。

- 远程主机/环境：RTX 3090，`realpde-pytorch-h5py:0831`
- Resume checkpoint：`/home/chyfuture/realpde_runs/p0a_n2_full_continuation_20260902/full_12481_to15300_deadline/model_last.pth`；update `15300`；SHA-256 `3ea4e4def03ae2f1d970975e4217358e1d762b88a69bdddfdf844d551baaa3e4`
- 输出：`/home/chyfuture/realpde_runs/sota_full_night_20260904_retry2/`
- launcher 日志：`/home/chyfuture/realpde_runs/sota_full_night_20260904_retry2_logs/launcher.log`
- 精确启动命令：`DATA_ROOT=/data KIT_ROOT=/kit RESUME_CHECKPOINT=/resume/model_last.pth OUT_ROOT=/out bash tools/run_sota_full_night_20260904.sh`
- 完成情况：elapsed `18351.24 s`；peak GPU allocation `5067694080` bytes；里程碑 `31100 / 36500 / 37850 / 40560 / 43260` 全部生成；最终 `model_last.pth` 已生成。
- Summary：`/home/chyfuture/realpde_runs/sota_full_night_20260904_retry2/full_night_summary.json`

## 2026-09-04 SPS bounds screen

实验状态：`DONE`。在冻结 50/16 validation dev、P0-A + N2 validation update `30900` 上，直接使用官方 v9 `scoring.py` 的 SPS 实现扫描小范围 `abs + rel * |prediction|` bounds；未训练、未访问 locked-final/private-test、未提交 Codabench。

Validation raw errors：Rel-L2 `0.11284460`，TKE `0.50010282`，MVPE `0.08728255`。

| Bounds | SPS score | Coverage |
|---|---:|---:|
| fallback | 16.295627 | 0.303976 |
| abs=0.0050, rel=0 | 33.240237 | 0.538482 |
| abs=0.0075, rel=0 | 37.788062 | 0.669603 |
| abs=0.0100, rel=0 | 38.735065 | 0.750755 |
| abs=0.0125, rel=0 | 37.917093 | 0.803713 |
| abs=0.0075, rel=0.01 | 38.838701 | 0.708041 |
| **abs=0.0075, rel=0.02** | **39.112385** | **0.735240** |

结论：`abs=0.0075, rel=0.02` 为本地 SPS 首选，相对 fallback 提升 `+22.816758`。

## 2026-09-04 Codabench new SOTA

日期：`2026-09-04`

Candidate：`P0-A + N2 + CNO + full@43260 + abs0075_rel002`

相对上一版新增：full-data late-training 从 `15300` 深化到 `43260`，并显式输出校准后的 SPS bounds。

训练 checkpoint：update `43260`，SHA256 `50b692e236d5df9285a5cee976a51e3457a7eeed0f87d55b6568745077645d71`。

SPS / bounds：`half_width = 0.0075 + 0.02 * abs(prediction)`。

Codabench：Final `76.149726` / Rel-L2 `93.434384` / TKE `77.588799` / MVPE `92.519561` / Time `86.998134` / SPS `27.545059`。

结论：`KEEP / PREVIOUS SOTA`。

## 2026-09-05 Codabench adaptive uncertainty new SOTA

日期：`2026-09-05`

Candidate：`P0-A + N2 + CNO + full@43260 + v5 base Adaptive Uncertainty Head@1400`

相对上一版新增：仅将 static bounds 替换为 learned adaptive uncertainty；backbone prediction parity `max_abs_diff=0.0`。

训练 checkpoint：仍为 full@43260，SHA256 `50b692e236d5df9285a5cee976a51e3457a7eeed0f87d55b6568745077645d71`。

SPS / bounds：`half_width_uv = 0.0025 + sigma`，pressure half-width `0`。

Submission ZIP SHA256：`3285ad3a424988ab35061337ca836c23b5f7db04773246167da3a9f8eaa2178a`。

Codabench：Final **`76.694784`** / Rel-L2 `93.434384` / TKE `77.588799` / MVPE `92.519563` / Time `87.066646` / SPS **`29.519724`**。

结论：**KEEP / NEW SOTA**。相对 2026-09-04 版本 Final `+0.545058`、SPS `+1.974665`、Time `+0.068512`，物理 prediction scores 基本不变。Adaptive Uncertainty 在线验证成功。

下一轮主要问题：在保留 adaptive uncertainty 的前提下，继续积累具有足够证据和预期收益的模型 / formulation / objective 增量；只有通过 Merge Worthiness Gate 后，才进入 `50/16 → GO_FULL → package → submit`。