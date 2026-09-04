# 线上 SOTA 迭代

## 1. 目标

本目录用于 Track 1 的**每日线上 SOTA 冲刺**。

核心目标不是独立研究某一个技术方向，而是利用每天有限的 Codabench 提交机会，把截至当天已经论证清楚、具备正向证据且能够稳定部署的优化项合并到当前最佳候选中，完成训练、SPS 优化、打包和线上提交，持续刷新正式分数。

各技术方向仍在自己的目录中完成研究和验证，例如 Modeling、Loss、Feature Engineering、Training、Inference。本目录只负责**跨方向收口与提交**。

## 2. 基本原则

1. **只合并已经有证据的改动。** 不在 submission candidate 上临时加入未经验证的新结构、新 Loss 或新 Feature。
2. **当前线上最佳结果始终作为锚点。** 新版本必须明确说明相对上一版新增了什么。
3. **每天尽量形成一个可提交版本。** 当天没有足够可靠的新模型改动时，也可以只做 SPS、推理或打包优化。
4. **SPS 是每次提交前的固定步骤。** SPS 优化与模型训练分开处理，不污染模型实验结论。
5. **避免浪费提交次数。** 本地 official scorer、SPS/interval 检查、runtime smoke 和 clean-package smoke 通过后再提交。
6. **线上结果只用于确认整体效果。** 不把 Codabench 当作高频超参数搜索器。

## 3. 每日流程

```text
各方向最新实验结论
        ↓
筛选当天可合并的 KEEP / GO 项
        ↓
确定单一 SOTA candidate recipe
        ↓
必要的小规模 A/B 或收尾实验
        ↓
全量 competition refit / continuation
        ↓
SPS / bounds 优化
        ↓
official smoke + runtime + package check
        ↓
Codabench 提交
        ↓
记录线上结果并更新下一轮基线
```

## 4. 当前线上锚点

截至 2026-09-04：

- 当前最好正式结果：**`76.149726`**
- Candidate：`P0-A + N2 + CNO + full@43260 + explicit SPS bounds`
- Bounds：`half_width = 0.0075 + 0.02 * abs(prediction)`
- Checkpoint SHA256：`50b692e236d5df9285a5cee976a51e3457a7eeed0f87d55b6568745077645d71`
- Submission ZIP SHA256：`f8a79ec1114b4e7f05edc9bc95c6810c5250ca27b5044ca387044c0671d9fc98`
- Codabench：Rel-L2 `93.434384` / TKE `77.588799` / MVPE `92.519561` / Time `86.998134` / SPS `27.545059`
- 相对旧线上最佳 `75.584550`：Final `+0.565176`

正式提交历史见：[`../submission_log.md`](../submission_log.md)。

## 5. 当前 SOTA 主线

当前优先路线：

**P0-A features + N2 loss + CNO + full-data late-training continuation + calibrated SPS bounds**。

这条路线已获得线上验证：相比旧历史最佳 CNO，当前 candidate 的 TKE 明显更高，同时 Rel-L2/MVPE 基本保持强势；显式 bounds 将上一次 P0-A full@15300 的 SPS `11.431650` 恢复到 `27.545059`，最终分数刷新为 `76.149726`。

当前不作为 SOTA 主线自动合并：

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

结论：`abs=0.0075, rel=0.02` 为本地 SPS 首选，相对 fallback 提升 `+22.816758`；候选间已出现宽度继续增加后的收益饱和，因此没有扩 SPS 网格。

## 2026-09-04 Codabench new SOTA

日期：`2026-09-04`

Candidate：`P0-A + N2 + CNO + full@43260 + abs0075_rel002`

相对上一版新增：full-data late-training 从 `15300` 深化到 `43260`，并显式输出校准后的 SPS bounds。

训练 checkpoint：update `43260`，SHA256 `50b692e236d5df9285a5cee976a51e3457a7eeed0f87d55b6568745077645d71`。

SPS / bounds：`half_width = 0.0075 + 0.02 * abs(prediction)`。

Codabench：Final **`76.149726`** / Rel-L2 `93.434384` / TKE `77.588799` / MVPE `92.519561` / Time `86.998134` / SPS `27.545059`。

结论：**KEEP / NEW SOTA**。相对旧线上最佳 `75.584550` 提升 `+0.565176`。SPS 已从用户上一版 full@15300 的 `11.431650` 恢复到 `27.545059`；TKE 仍保持明显高于旧历史最佳的水平。

下一轮主要问题：在保持 Rel-L2 / MVPE / SPS 不回退的前提下，优先寻找进一步提升 TKE 或模型组合收益的方案；不浪费提交机会在无新增证据的 full@40560 backup 上。
