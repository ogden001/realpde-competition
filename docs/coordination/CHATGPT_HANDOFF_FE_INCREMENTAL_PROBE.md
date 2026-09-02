# ChatGPT / Sol Review Handoff — `T1-ID-FE-INCR-PERSIST-RIDGE-S20260902`

请从已连接的 GitHub 仓库读取本文件，并按 Track 1 V3 协议 review。任务已停止在
`REVIEW_REQUIRED`，没有授权下一轮模型训练、locked-final/private-test 审计或
Codabench 提交。

## 结论摘要

这是一次最小 supervised incremental-value probe，不是神经网络训练，也不是
“Feature 对所有模型都有效”的证明。由于本地没有可追溯的 CNO validation
prediction array，本次复用了已有 registry 中的 frozen `PERSIST` prediction：
每个完整 20→20 runtime window 用最后一个观测 `u/v` frame 重复 20 个未来 frame。
因此 `baseline_checkpoint=null` 是有意的，避免把新推理冒充既有 baseline prediction。

在相同 baseline、样本和 residual target 下，使用 train-only 标准化与闭式 ridge
residual probe 得到：

| Feature package | dim | dev Rel-L2 | dev TKE | dev MVPE | Δ Rel vs Raw-Control | Δ TKE | Δ MVPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw-Control | 2 | 0.131353 | 0.987575 | 0.130701 | 0 | 0 | 0 |
| Raw+Temporal(mean/std/delta) | 8 | 0.118340 | 0.940578 | 0.109454 | -0.013013 | -0.046997 | -0.021247 |
| Raw+Spatial(4 primitive) | 6 | 0.130241 | 0.973685 | 0.129355 | -0.001112 | -0.013889 | -0.001346 |
| Raw+Temporal+Spatial | 12 | 0.116346 | 0.936060 | 0.107228 | -0.015007 | -0.051515 | -0.023473 |

这里的 delta 是 corrected-field raw error 的候选减 control，负值更好。已有
registered PERSIST（不带 residual probe）的 dev raw errors 为
`0.135578 / 1.000000 / 0.132707`，所以 Raw-Control 本身也改善了该轻量 residual
control。Trajectory-macro win rate 对 Raw-Control：Temporal 为 Rel/TKE/MVPE
`0.875 / 1.000 / 0.938`，Spatial 为 `0.812 / 1.000 / 0.750`，联合组为
`0.938 / 1.000 / 1.000`。

## 预注册口径与证据

- Manifest SHA-256：`42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`
- Split：冻结 50 train / 16 dev；完整未来 residual 窗口为 2052 train / 659 dev；未读 final。
- Runtime 输入：20 frames，stride 20，空间 H×W=`32×64`（官方 scorer 的 2× spatial subsample）。
- Baseline 来源：已登记 `T1-ID-POINT-V0-7500-S20260901` 的 deterministic `PERSIST`；无 checkpoint inference。
- Residual target：`target[20:40,:,:,:2] - repeat(input[19,:,:,:2], 20)`；没有使用 target-derived feature。
- Raw-Control：最后观测 frame 的 `(u,v)`，2 维/像素。
- Temporal：增加 `(u_mean,v_mean,u_std,v_std,u_delta,v_delta)`，其中 delta 为最后两帧差，8 维/像素总输入。
- Spatial：增加最后观测 frame 的 `du_dx_pixel, du_dy_pixel, dv_dx_pixel, dv_dy_pixel`，6 维/像素总输入。
- Joint：12 维/像素。TKE 和 vorticity 没有作为独立输入。
- Spatial edge rule：spacing=1；interior centered difference；outer index 使用 first-order forward/backward difference。
- Ridge：每组独立闭式解；feature mean/std 只由 train rows (`4,202,496` rows) 计算；`(XᵀX + alpha·n·I)⁻¹XᵀY`，alpha=`1e-2`；没有 dev 调参。
- Scorer SHA-256：`a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`
- 实际命令（从 workspace root）：
  `python code/tools/realpde_incremental_value_probe.py --data-root artifacts/sim2real_gap_full/_full_cache/real --manifest artifacts/loss_optimization_v9_20260901_run1/evidence/manifests/id_seed20260901.json --kit-root <temporary extracted starting-kit> --out-dir artifacts/fe_incremental_probe_s20260902`
- Implementation SHA-256：`6e37aa883c6080df3ce60b15173f1d451002a50694c55aaff47605ca4eea3567`
- Run wall time：约 `251 s`（CPU，约 4.2 min；预算 ≤30 min）。Run 时 repository HEAD=`442584e4e7ada3ad6472a055295a3dfbb917da22`；提交本次脚本与文档前工作树已有无关 dirty/untracked 改动。
- 代码 commit：见本 handoff 所在提交；生成的数值产物不进 Git。

## 产物

本地 repository-relative artifact directory：`../artifacts/fe_incremental_probe_s20260902/`

- `summary.json`：全部 metadata、四组指标、window/trajectory win rates
- `window_metrics.csv`：659 个 dev windows 的 baseline 与四组 corrected-field 指标
- `trajectory_metrics.csv`：16 条 dev trajectory 的 macro 指标
- `ridge_raw_control.npz`, `ridge_raw_temporal.npz`, `ridge_raw_spatial.npz`, `ridge_raw_temporal_spatial.npz`：小型 ridge 系数与 train-only normalizer
- `report.md`：同口径结果表

## Shortlist 与停止状态

- **Raw+Temporal：`KEEP_FOR_MODEL_PROBE`** — 三个指标均有明显、同方向且 trajectory-level 稳定增量。
- **Raw+Spatial：`KEEP_FOR_MODEL_PROBE`** — 增量较小，但三指标同方向，未出现 trade-off；仅凭本线性 probe 不扩大结论。
- **Raw+Temporal+Spatial：`KEEP_FOR_MODEL_PROBE`** — 本 probe 数值最佳，但不能据此跳过 ChatGPT review。
- 未列入的 `TKE`、`vorticity` 不构成独立输入；vorticity 是确定性 derived summary `dv_dx_pixel - du_dy_pixel`，前一 Spatial diagnostic 的 Pearson/Spearman 均为 `1.0/1.0`。

Protocol conclusion：`GO`（仅表示三组进入 ChatGPT review 的 shortlist）+ final state
`REVIEW_REQUIRED`。请 review 后再给出唯一、范围受限的 `NEXT_ACTION`。
