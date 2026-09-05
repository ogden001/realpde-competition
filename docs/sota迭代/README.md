# 线上 SOTA 迭代

## 1. 目标

本目录用于 Track 1 的**每日线上 SOTA 冲刺**。

核心目标不是独立研究某一个技术方向，而是利用每天有限的 Codabench 提交机会，把截至当天已经论证清楚、具备正向证据且能够稳定部署的优化项合并到当前最佳候选中，完成训练、SPS 优化、打包和线上提交，持续刷新正式分数。

各技术方向仍在自己的目录中完成研究和验证，例如 Modeling、Loss、Feature Engineering、Training、Inference。本目录只负责**跨方向收口与提交**。

### “merge SOTA” 的固定含义

当用户明确说 **“merge SOTA”**、**“合并 SOTA”** 或要求“把已验证有效策略合起来训练并提交”时，默认进入 **SOTA Merge Execution Mode**，这是一条执行指令，不是新的研究任务。

固定目标：

> 把此前已经验证有效、用户希望合并的策略直接编码到同一个 submission recipe 中。先在冻结的 50 Train / 16 Dev 上用同一 recipe 训练，默认约 2 小时，与历史 SOTA 的 50/16 结果直接比较；结果整体 OK 后立即启动全量 competition 训练。随后做最小 SPS / package smoke 并尽快 Codabench。线上失败或回退是可接受的实验结果，不应因为追求离线证据完美而长期阻塞提交。

默认执行顺序：

```text
已验证有效策略
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

SOTA Merge Mode 下的硬规则：

1. **50/16 是快速可比 Gate，全量训练是最终 submission 主路径。** 不跳过历史可比性，也不把 50/16 扩展成新的长期研究。
2. **不把 merge 重新变成 research。** 已验证策略直接合并；50/16 只用于回答“相对当前 SOTA 是否值得 full run”。
3. **优先时间效率。** 默认 50/16 训练约 2 小时；若指标趋势整体 OK，立即进入 full-data，不要求完整 ablation、trajectory Gate、机制诊断或大规模测试矩阵。
4. **提交失败可以接受。** Codabench 是真实实验的一部分；不以“必须确保线上提升”为前提才允许提交。
5. **只保留必要安全检查。** shape / finite / checkpoint 可加载 / prediction path / package clean-room 等直接影响提交正确性的检查必须做；不为一次 merge 临时建设通用框架、扩展测试矩阵或额外研究流水线。
6. **如果用户说“今晚 merge SOTA”或给出类似时间约束，速度优先级高于研究完备性。** 除非遇到会让训练或 submission 明显无效的硬错误，否则应持续向“50/16 → full-data → package → submit”推进。

## 2. 基本原则

1. **只合并已经有证据的改动。** 不在 submission candidate 上临时加入未经验证的新结构、新 Loss 或新 Feature。
2. **当前线上最佳结果始终作为锚点。** 新版本必须明确说明相对上一版新增了什么。
3. **每天尽量形成一个可提交版本。** 当天没有足够可靠的新模型改动时，也可以只做 SPS、推理或打包优化。
4. **SPS 是每次提交前的固定步骤。** SPS 优化与模型训练分开处理，不污染模型实验结论。
5. **避免无意义浪费提交次数，但不追求提交前证据完美。** 必要 local smoke 通过即可进入 submission review；线上失败或回退可以作为真实实验结果记录。
6. **线上结果用于确认整体效果。** 不把 Codabench 当作高频超参数搜索器，但允许在 SOTA Merge Mode 中用一次真实提交验证完整组合。

## 3. 每日流程

常规研究收口流程：

```text
各方向最新实验结论
        ↓
筛选当天可合并的 KEEP / GO 项
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

下一轮主要问题：在保留 adaptive uncertainty 的前提下，把其它已经验证有正向证据的策略按 `50/16 约 2h → GO_FULL → package → submit` 的 merge 流程快速合并，不再把 SOTA merge 变成新的研究 Campaign。
