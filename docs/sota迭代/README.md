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

截至 **2026-09-16**：

- Final：**`77.314446`**
- Rel-L2：`93.816645`
- TKE：`79.164203`
- MVPE：`93.411176`
- Time：`86.898836`
- SPS：`30.319572`

当前 recipe：

```text
Dense-All
+ P0-A 20-channel features
+ MF-CNO
+ N2 loss
+ vorticity supervision
+ Stage-B low-LR extra Rel
+ all-82 full-data refit @53582
+ fresh Adaptive Uncertainty Head @1400
+ half_width_uv = 0.0025 + sigma
+ pressure half-width = 0
```

Full backbone：

- update：`53582`
- SHA256：`f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce`

Final clean package：

- path：`/home/chyfuture/realpde_runs/sota_v2_adaptive_20260916/package_clean/submission.zip`
- SHA256：`9cfc055c6232d2b0aef9f88f1cb3de2aae883b7cff7f02659ed8b9cf85b3ed55`
- bytes：`30191330`
- package prediction parity A/B：`0.0 / 0.0`

最新完整 review：

`docs/sota迭代/reviews/sota_v2_adaptive_20260916/README.md`

---

## 3. 相对上一线上 SOTA

上一版：

```text
P0-A + N2 + CNO + full@43260
+ v5 Adaptive Uncertainty Head@1400
Final = 76.694784
```

SOTA-V2 相对增益：

| Metric | Previous SOTA | SOTA-V2 | Delta |
|---|---:|---:|---:|
| Final | `76.694784` | **`77.314446`** | **`+0.619662`** |
| Rel-L2 | `93.434384` | **`93.816645`** | `+0.382261` |
| TKE | `77.588799` | **`79.164203`** | `+1.575404` |
| MVPE | `92.519563` | **`93.411176`** | `+0.891613` |
| Time | `87.066646` | `86.898836` | `-0.167810` |
| SPS | `29.519724` | **`30.319572`** | `+0.799848` |

结论：**`ONLINE_KEEP / NEW_ONLINE_SOTA`**。

这次提升不是 SPS 单点收益。Rel-L2、TKE、MVPE 三个 predictive subscore 与 SPS 同时上涨，说明 SOTA-V2 integrated backbone 的 50/16 正收益成功迁移到线上 hidden evaluation。

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
5. 相对当前 Final `77.314446`，是否存在足够明显的线上提点预期？

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

- Residual Corrector / Local / Point：TKE 保护不足，PARKED；
- coarse+fine Multi-scale：NO_GO；
- Structured Temporal Transformer / Attention / ΔUV：CLOSED / WEAK_SIGNAL；
- raw CFD / 复杂 Sim2Real：PARKED；
- Feature 堆叠扫描：PARKED。

---

## 8. 下一轮重点

当前先做线上结果复盘，不立即启动新 full-data run。

值得继续关注：

1. **TKE amplitude calibration**：当前 TKE 仍是主要物理短板；已有诊断显示趋势相关性高，但能量幅值存在偏差。
2. **SPS calibration generalization**：Dev adaptive SPS `45.07`，线上 `30.32`，说明 calibration 泛化仍有空间。
3. **Late horizon**：h19/h20 仍占较高 squared-error fraction。
4. **更强 backbone family / modal modeling**：只有出现明显大台阶证据才进入下一次 merge。

## 9. 2026-09-17 SPS stride=1 replica

实验状态：`COMPLETED / SPS_REPLICA_NO_GO / REVIEW_REQUIRED`。在冻结 50 Train / 16 Dev、SOTA-V2 validation backbone@32500 上，按预注册协议训练 `h32/b2` Gaussian-NLL uncertainty head `1400` updates；唯一变量是训练窗口从 fixed stride20 改为全部合法 `dense_all` stride1 窗口（`40488` train windows）。

- baseline Dev SPS：`45.07008160038756`
- stride1 candidate：`45.0605672284722`，delta `-0.00951437191535831`
- mean UV width：`0.02358330972492695 → 0.023670747876167297`，ratio `1.0037076284991469`，width guard 通过
- Gate：`SPS_REPLICA_NO_GO`，未达到预注册 `+1.5` SPS 门槛
- Phase 2、full-specific head、package 和 clean-room smoke：按 Stop 规则全部跳过

没有访问 locked-final/private Future20，也没有 Codabench。完整轻量证据见 [`reviews/sps_stride1_replica_20260916/README.md`](reviews/sps_stride1_replica_20260916/README.md)。

---

## 10. 关键文档

- 当前战略：`docs/realpde整体优化概要.md`
- Submission log：`docs/submission_log.md`
- 最新 online review：`docs/sota迭代/reviews/sota_v2_adaptive_20260916/README.md`
- 最新 handoff：`docs/coordination/CHATGPT_HANDOFF_SOTA_V2_ADAPTIVE_20260916.md`
- Inference：`docs/inference/inference概要.md`
- Experiment registry：`docs/track1_experiment_registry.md`
- 当前任务：`docs/sota迭代/NEXT_ACTION.md`
