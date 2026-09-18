# 线上 SOTA 迭代

## 1. 目标

本目录用于 RealPDE Track 1 的**线上 SOTA 收口与提交**。

这里不负责把每个技术方向研究到极致，而是把已有证据中值得合并的变量收口为单一 candidate，经过固定 50/16 Gate、full-data refit、SPS calibration、package smoke 和 Codabench，持续刷新真实线上结果。

详细历史实验保存在：

- `docs/sota迭代/reviews/`
- `docs/submission_log.md`
- `docs/track1_experiment_registry.md`
- `docs/coordination/`

本 README 只维护**当前线上 anchor、SOTA recipe、Merge Worthiness 规则和最新提交结论**。

---

## 2. 当前线上 SOTA

截至 **2026-09-18**，当前线上最好仍是 2026-09-17 teammate35 full-specific SPS 包：

- Final：**`77.732796`**
- Rel-L2：`93.816645`
- TKE：`79.164203`
- MVPE：`93.411176`
- Time：`86.699342`
- SPS：**`31.961724`**

当前 recipe：

```text
Dense-All
+ P0-A 20-channel features
+ MF-CNO
+ N2 loss
+ vorticity supervision
+ Stage-B low-LR extra Rel
+ all-82 full-data refit @53582
+ teammate35 full-specific uncertainty head @1600
+ half_width_uv = 0.0025 + sigma
+ pressure half-width = 0
```

Full backbone：

- update：`53582`
- SHA256：`f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce`

Current SOTA package：

- path：`/home/chyfuture/realpde_runs/sps_teammate_final_20260917/package_clean/submission.zip`
- SHA256：`cc236a4926d36568ea9eb2d4c770f08b0a1c058ec3e85f82256d70d3df8db685`
- bytes：`30260970`
- package prediction parity：`0.0`

2026-09-18 的 exact teammate uncertainty-recipe submission 得到 Final `77.728857` / SPS `31.899537`，没有超过 2026-09-17 SOTA。

最新 review：

- `docs/sota迭代/reviews/sps_teammate_final_20260917/README.md`
- `docs/sota迭代/reviews/sps_teammate_exact_submit_20260918/README.md`

---

## 3. 当前 SOTA 的线上增益

相对 2026-09-16 SOTA-V2 adaptive 包，2026-09-17 teammate35 full-specific SPS 包只替换 uncertainty/SPS 路径，point predictor 保持完全一致：

| Metric | 2026-09-16 | Current SOTA | Delta |
|---|---:|---:|---:|
| Final | `77.314446` | **`77.732796`** | **`+0.418350`** |
| Rel-L2 | `93.816645` | **`93.816645`** | `0.000000` |
| TKE | `79.164203` | **`79.164203`** | `0.000000` |
| MVPE | `93.411176` | **`93.411176`** | `0.000000` |
| Time | `86.898836` | `86.699342` | `-0.199494` |
| SPS | `30.319572` | **`31.961724`** | **`+1.642152`** |

结论：**`ONLINE_KEEP / CURRENT_ONLINE_SOTA`**。

2026-09-18 exact teammate uncertainty-recipe submission 保持三个 point scores 完全不变，但 SPS 为 `31.899537`、Final 为 `77.728857`；相对 current SOTA，SPS `-0.062187`、Final `-0.003939`。更精确的 recipe 对齐没有带来线上增益。

---

## 4. 50/16 → Full-data 证据链

### 4.1 50/16 predictive Gate

固定 50 Train / 16 Dev，SOTA-V2 @32500：

- Rel-L2 `0.0999346152`
- TKE `0.4692927301`
- MVPE `0.0757779852`

Historical balanced anchor：

- Rel-L2 `0.112925`
- TKE `0.494840`
- MVPE `0.084671`

对应 raw error 下降约：

- Rel-L2：`11.50%`
- TKE：`5.16%`
- MVPE：`10.50%`

### 4.2 Full-data refit

- released PIV trajectories：`82`
- canonical windows：`3383`
- Dense-All windows：`66755`
- Stage A end：`49461`
- final update：`53582`
- effective batch：`8`

`53582` 是在 full refit 前由 50/16 `@32500` sweet spot 按 Dense epoch exposure 映射后冻结的终点，不是依据 full-data training loss 事后选模。

### 4.3 Adaptive uncertainty

Validation backbone：SOTA-V2 50/16 `@32500`。

Head training：

- 50 Train only
- canonical windows：`2052`
- updates：`1400`
- architecture：`15 → 32`, 2 residual blocks

Calibration：

- 16 Dev
- canonical windows：`659`
- fixed 28-row floor × multiplier grid

结果：

- same-backbone static SPS：`42.1248919471`
- adaptive best SPS：`45.0700816004`
- bounds：`floor=0.0025, mult=1.0`

最终把同一个 head + frozen bounds 挂到 full@53582，不再用 full-data 重新训练 uncertainty head。

---

## 5. Merge Worthiness Review

当用户提出“merge SOTA / 合并 SOTA / 做一次提交版本”时，默认**先判断值不值得 merge**，不是直接启动训练。

完整 merge 会消耗多轮 ChatGPT/Codex 协作、GPU 长训、package 和提交机会。因此必须回答：

1. 当前线上 SOTA 已经包含哪些变量？
2. 本轮真正新增 / 替换哪些独立变量？
3. 每个变量的证据强度如何？
4. 预计影响 Rel-L2 / TKE / MVPE / Time / SPS 哪几项？
5. 相对当前 Final `77.732796`，是否存在足够明显的线上提点预期？

决策：

- `MERGE_WORTHY`：存在大台阶变量，或多个证据强且兼容的增量；
- `SKIP_MERGE`：只有孤立小收益、weak signal、单指标 trade-off 或线上迁移风险高。

离线 `+1%` 左右的单一弱收益默认不足以单独触发完整 merge。

---

## 6. 固定 SOTA Merge 流程

```text
各方向最新证据
        ↓
Merge Worthiness Review
        ↓
MERGE_WORTHY ?
  ├─ NO → SKIP_MERGE
  └─ YES
        ↓
固定 50/16 matched candidate
        ↓
与当前 SOTA comparable anchor 比较
        ↓
GO_FULL
        ↓
all-released full-data refit
        ↓
SPS / uncertainty calibration
        ↓
clean package build
        ↓
real-fixture A/B parity smoke
        ↓
Codabench
        ↓
KEEP / ROLLBACK
```

硬规则：

- 不把 merge 重新变成长期 research campaign；
- 不因为 GPU 空闲就机械 full train；
- 不使用 locked-final/private test 做调参；
- SPS 与 predictive modeling 分开评估；
- uncertainty 不能改变 prediction；
- package 使用白名单、clean rebuild、parity smoke；
- 新线上结果必须写入 `docs/submission_log.md`。

---

## 7. 当前主线与 parked 方向

当前 SOTA core：

- Dense-All：`ONLINE_KEEP`
- P0-A：`ONLINE_KEEP`
- MF：`INTEGRATED_KEEP / attribution unresolved`
- Vorticity：`INTEGRATED_KEEP / attribution unresolved`
- N2 + Stage-B：`ONLINE_KEEP`
- Adaptive Uncertainty：`ONLINE_KEEP`

其中 MF / Vorticity 的 isolated long-run 证据曾 mixed，因此**不能把线上总增益单独归因给它们**；但在完整 SOTA-V2 integrated recipe 中它们已经通过整体线上验证，不需要为了提交主线重新做 ablation。

当前不自动合并：

- Residual Corrector / Local / Point：作为纯 point 增益路线曾因 TKE 保护不足而 PARKED；但 teammate SPS 证据表明其 `base → residual corrector → final` 结构可能与 uncertainty 泛化耦合。仅允许在下一次完整 SOTA merge 中作为 residual+SPS 联合结构重新评估，不自动单独推进；
- coarse+fine Multi-scale：NO_GO；
- Structured Temporal Transformer / Attention / ΔUV：CLOSED / WEAK_SIGNAL；
- raw CFD / 复杂 Sim2Real：PARKED；
- Feature 堆叠扫描：PARKED。

---

## 8. 下一轮重点

2026-09-17/18 两次 teammate SPS 提交已经把纯 uncertainty-head recipe 调整的收益空间基本跑清。下一轮不再继续 SPS-only 微调，重点转为完整 SOTA merge：

1. **Residual + SPS 联合结构**：研究 teammate 的 `base → residual corrector → final` 与 `uncertainty(x, base) → final-error scale` 联合机制，避免继续把 uncertainty head 当作孤立后处理器。
2. **已验证主链路增量合并**：AoA/数据增强、Stage-B、残差修正等只按已有证据进入完整 merge，不重新铺大规模消融。
3. **TKE protection**：Residual corrector 若进入 merge，必须继续保护 TKE，不能只追 Rel-L2/MVPE。
4. **Late horizon / stronger backbone**：保留为后续大台阶方向，不为本轮 SPS 问题额外扩散实验。

SPS-only 关闭项：loss 选择、seed、1600/1800/2000 steps、h32/h64、50/82 uncertainty-head scope、floor/mult 微调。除非出现新的结构级证据，否则不再为这些变量消耗提交机会。

## 9. 2026-09-17 SPS stride=1 replica

实验状态：`COMPLETED / SPS_REPLICA_NO_GO / REVIEW_REQUIRED`。在冻结 50 Train / 16 Dev、SOTA-V2 validation backbone@32500 上，按预注册协议训练 `h32/b2` Gaussian-NLL uncertainty head `1400` updates；唯一变量是训练窗口从 fixed stride20 改为全部合法 `dense_all` stride1 窗口（`40488` train windows）。

- baseline Dev SPS：`45.07008160038756`
- stride1 candidate：`45.0605672284722`，delta `-0.00951437191535831`
- mean UV width：`0.02358330972492695 → 0.023670747876167297`，ratio `1.0037076284991469`，width guard 通过
- Gate：`SPS_REPLICA_NO_GO`，未达到预注册 `+1.5` SPS 门槛
- Phase 2、full-specific head、package 和 clean-room smoke：按 Stop 规则全部跳过

没有访问 locked-final/private Future20，也没有 Codabench。完整轻量证据见 [`reviews/sps_stride1_replica_20260916/README.md`](reviews/sps_stride1_replica_20260916/README.md)。

---

## 10. 2026-09-17 SPS-A2 backbone/head mismatch audit

实验状态：`COMPLETED / REVIEW_REQUIRED`。固定 50/16 Dev、同一 validation run 的 Backbone A=`@30000` 与 B=`@32500`，只训练 A-matched Head-A，并复用现有 frozen B Head-B；四组合均使用同一 28-row calibration grid。

- B + Head-B + Bcal：SPS `45.07008160038756`，严格复现当前 baseline
- B + Head-A + Acal：SPS `45.11577471806446`
- B + Head-A + Bcal：SPS `45.11577471806446`
- `mismatch_total = -0.04569311767689754`
- `mismatch_after_recalibration = -0.04569311767689754`
- 结论：`BACKBONE_HEAD_MISMATCH_NOT_SUPPORTED`

两次 calibration 都选择 `(floor=0.0025, mult=1.0)`，因此 recalibration 没有改变 cross-backbone 结果。未访问 locked-final/private/Codabench，也未进入 full head、package 或 smoke。完整证据见 [`reviews/sps_backbone_head_mismatch_20260917/README.md`](reviews/sps_backbone_head_mismatch_20260917/README.md)。

---

## 11. 2026-09-18 exact teammate SPS online submission

状态：`COMPLETED / ONLINE_NO_GAIN / CLOSE_SPS_ONLY_RECIPE_TUNING`。

在 frozen full SOTA-V2 @53582 point predictor 上执行 teammate uncertainty-head recipe：35-channel features、h32/b2/drop0、masked Gaussian NLL、seed41、2000 updates、每200步 Dev SPS 评估，并对每个 checkpoint 扫描固定 28-row floor × multiplier grid。项目固定 50 Train / 16 Dev 保持不变。

离线：

- selected update：`1800`
- floor / mult：`0.0025 / 1.0`
- Dev SPS：`44.48550611699792`
- coverage：`0.8642850171482572`
- mean UV width：`0.025083480402827263`
- point prediction parity：`0.0`

线上：

- Rel-L2：`93.816645`
- TKE：`79.164203`
- MVPE：`93.411176`
- Time：`86.828909`
- SPS：`31.899537`
- Final：`77.728857`

相对 2026-09-17 current SOTA，SPS `-0.062187`、Final `-0.003939`，point scores 完全相同。结论：更精确复制 masked NLL、seed、checkpoint search 与 calibration recipe 没有带来线上收益，不能解释 teammate SPS `38.442870` 的差距。

重要边界：这仍不是 teammate 整套端到端结构的完全复刻。teammate uncertainty head 观察 residual correction 前的 base，而区间中心是 residual-corrected final prediction；我们的当前 SOTA 没有同样的 pre-correction base / residual-corrector / final 三段结构。因此后续若继续追 SPS，只在完整 SOTA merge 中联合评估 residual + uncertainty，不再做 uncertainty-head-only 微调。

完整证据：`docs/sota迭代/reviews/sps_teammate_exact_submit_20260918/README.md`。

---

## 12. 关键文档

- 当前战略：`docs/realpde整体优化概要.md`
- Submission log：`docs/submission_log.md`
- 最新 online review：`docs/sota迭代/reviews/sota_v2_adaptive_20260916/README.md`
- 最新 handoff：`docs/coordination/CHATGPT_HANDOFF_SOTA_V2_ADAPTIVE_20260916.md`
- Inference：`docs/inference/inference概要.md`
- Experiment registry：`docs/track1_experiment_registry.md`
- 当前任务：`docs/sota迭代/NEXT_ACTION.md`


---

## 13. 2026-09-18 SOTA-V3 Fast Joint 验证

状态：`READY_FOR_EXECUTION / REVIEW_REQUIRED`。

原先的 8-stage SOTA-V3 long-run（Dev backbone 35k → Dev residual 30k → full backbone → full residual ...）因单卡训练成本过高，已在用户决策后**停止并被本方案替代**。旧 runner 保留作历史代码，不是当前施工入口。

当前目标只是在固定 50 Train / 16 Dev 上回答一个问题：

> 从已经验证的 SOTA-V2 @32500 + mature residual @30000 出发，加入 AoA 后做短程端到端 joint fine-tune，能否超过历史最强的 residual+TKE-projection 组合？

冻结初始化：

- backbone @32500 SHA256 `6926722895611c38dee79cf16ce43b87c7de3d4a1305e9db431dadd4c1a7bf47`
- residual @30000 SHA256 `1f72b329d9a160b206a6e9f4fce5e780022b3bac7a9ce4bdf5b8c9efa30b4506`
- comparator：`@32500 + residual@30000 + alpha=1 + spatial_tke_map`
  - Rel-L2 `0.0889103040`
  - TKE `0.4692927301`
  - MVPE `0.0711286217`

Fast joint：

- Train50 Dense-All；Dev16 固定；
- 5,000 updates；每 1,000 updates 评估；
- AoA `±2° / p=0.5`，Past/Future 同角度；
- backbone LR `1e-6`，residual LR `1e-5`；
- forward 固定 `base -> residual -> spatial_tke_map -> final`；
- backbone 与 residual 同时反向传播；
- Loss 使用现有 Stage-B physical loss 直接监督 final；
- 不增加新的 auxiliary loss，不做 sweep。

GO_FULL Gate：

1. normalized 三指标平均 error 相对 historical comparator 至少下降 1%；
2. 任一单指标恶化不超过 2%；
3. 至少两项指标改善。

若 point Gate=GO，只允许继续在同一 50/16 上跑 Dev SPS，随后停止等待人工 review。**禁止自动 full train**。

当前代码与任务：

- `tools/realpde_sota_v3_fast_joint.py`
- `tests/test_sota_v3_fast_joint.py`
- `docs/sota迭代/NEXT_ACTION.md`

只有用户审阅 50/16 结果并明确确认后，才另开 full-data train + SPS + package 提交任务。
