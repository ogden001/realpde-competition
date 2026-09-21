# EXPERIMENT_HISTORY

只记录形成 80 分方案的关键节点。更早的 UNet / FNO 基线对照不进入本表。

## 1. 线上提交时间线

| Version | Main Change | Dev Rel | Dev TKE | Dev MVPE | Dev SPS | Online Final |
|---|---|---:|---:|---:|---:|---:|
| 9-1 champion | CNO E3 + residual h64/b2/u2400 | N/A | N/A | N/A | N/A | 78.077620 |
| A probe | + adaptive SPS interval (h32 head) | N/A | N/A | N/A | N/A | 78.685445 |
| fp16-base | fp16 inference | N/A | N/A | N/A | N/A | 78.806628 |
| all81 + stride1 + probe | all81 CNO, stride=1, probe head | N/A | N/A | N/A | N/A | 79.618801 |
| contdense residual | continued residual head | N/A | N/A | N/A | N/A | 79.773638 |
| E | contdense2 + probe | N/A | N/A | N/A | N/A | 79.876809 |
| **H (FINAL)** | h96x8 long residual + h64 logmae head | 0.080230 | 0.450854 | 0.070934 | 51.6332 | **80.078849** |

线上 SPS 对照（有记录的几次）：

```text
9-1 champion   sps 33.099
A probe        sps 35.596
fp16-base      sps 35.584
all81+stride1  sps 38.443
contdense      sps 38.952
E              sps 39.467
H (FINAL)      sps 40.209
```

H 的真实子分：

```text
rel_l2_score = 94.739714
tke_score    = 76.892538
mvpe_score   = 93.845797
time_score   = 86.542490
sps_score    = 40.209117
final_score  = 80.078849
```

## 2. 方案形成路径

```text
官方 CNO baseline
  -> all81 fine-tune (stride=1)
  -> residual corrector (h64 -> h96, u2400 -> u38400)
  -> SPS adaptive interval (probe head + grid scan)
  -> H: h96x8 residual + h64 logmae head + scanned interval
  -> 80.078849 online
```

## 3. CONFIRMED EFFECTIVE

有明确实验或线上提交支持，进入最终方案：

| Strategy | Evidence |
|---|---|
| all81 全量训练（含 dev 轨迹） | all81+stride1 线上 79.619，明显高于只用 65 条 |
| CNO stride=1 微调 | 65926 windows，Stage 1 输出 cno_final_all81.pt |
| Residual corrector（零初始化、max_delta=0.04、alpha=1.0） | 多次线上提升 |
| 残差头容量 × 步数匹配：h96 + 2400/4800/9600/19200/38400 | dev best 从 80.461 单调升到 81.240 |
| h64 + logmae uncertainty head | dev SPS 50.2 -> 51.60 |
| SPS 区间网格扫描（floor/mult/rel） | dev SPS 51.60 -> 51.63，线上 sps 40.209 |
| 冻结 backbone 的缓存训练（stride=5） | head 训练只需约 18 分钟/6000 步 |

## 4. TESTED BUT NOT USED

已实验但未进入最终方案（不要重复试）：

| Strategy | Result | Decision |
|---|---|---|
| SPS 参数继续微调 | 最优区域平坦，第 1/2 名差 0.010 | 榨干，停止 |
| 逐 horizon SPS 校准 | +0.001 | 冗余（sigma 已含时间维） |
| pinball / sps surrogate / reward-weighted loss | 全无效；reward-weighted 与基线逐位相同 | 放弃 |
| history-context probe head | +0.05，噪声级 | 放弃 |
| 相似模型集成 | 输出差异仅 1.3% | 放弃 |
| stride=1 从零训练残差头 | -2.9% | 放弃（只有续训形式有效） |
| 低 LR 收口（contdense3, lr 1e-5） | -0.76 | 放弃 |
| residual blocks=3（h96b3） | -0.70 | 放弃 |
| 探针头 h96 vs h64 | 仅 +0.05 | 选 h64（省推理） |
| AMP bf16 | 会改变数值精度 | 按用户约束放弃 |
| UNet / FNO 基线替换 | 均低于 CNO（历史实验） | 保留 CNO |

## 5. UNKNOWN

```text
1. 4 相位下采样增强（P00/P01/P10/P11）是否有效
   - 敏感性实验已完成：P00/P01 接近，P10/P11 的 mvpe 差约 6%
   - h64 baseline vs phaseaug 对照实验进行中（2026-09-21）
   - 未进入 80 分包

2. h96x16（76800 updates）是否继续提升
   - 训练进行中，best_iter 尚未确定
   - 预期边际收益递减（h96x4 -> h96x8 本地 +0.191）

3. AoA / 攻角旋转增强
   - 完全未实现、未验证

4. 同事方案的 P0-A（20 通道输入）+ N2 多目标损失 + 涡量监督
   - 不在本仓库，未复现
   - 同事 TKE 79.164 vs 我们 76.893，是唯一明显领先项

5. Stage 1 CNO 训练未固定随机种子（final_all81.py 无 manual_seed）
   - 复现时数据顺序可能不同，需要固定种子后重跑或接受相似但不完全相同的权重
```

## 6. FUTURE_RESEARCH_NOTES

以下内容**没有在 handoff 任务中执行**，只是记录：

```text
- h96x16 / h128 长训或更大 corrector 容量
- 4 相位增强若对照为正，可上 h96 长训
- P0-A 20 通道输入重训 CNO（同事路线，TKE 潜力）
- 涡量 / TKE 多目标监督
- 最后几帧误差专项优化（逐 horizon SPS 已验证为参数层面无解）
```
