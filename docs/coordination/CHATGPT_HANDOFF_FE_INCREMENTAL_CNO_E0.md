# ChatGPT / Sol Review Handoff — `T1-ID-FE-INCR-FROZEN-CNO-E0-RIDGE-S20260902`

请从已连接的 GitHub 仓库读取本文件，并按 Track 1 V3 协议 review。本任务已经停止在
`REVIEW_REQUIRED`；不要自行启动 fusion training、locked-final/private-test 审计或
Codabench 提交。

## 1. Frozen CNO baseline provenance

- Baseline experiment/artifact ID: `T1-ID-LOSS-E0-90M-S20260901`（registry 中的
  historical `OFFICIAL_WARM_START` E0 reference）。
- Checkpoint artifact: `loss_optimization_v9_20260901_run1/raw/long_E0_s20260901/model_best.pth`
  （远程 artifact；绝对主机路径不写入 Git）。
- Checkpoint SHA-256: `5d02c8da5bcbcbcc47917b0021b1007b2036931a3a51483772f7607deeb4aff6`。
- Official starting-kit scorer: v9，SHA-256
  `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`。
- Model protocol: vendored `CNO3d(in_dim=3, out_dim=3, out_dim_mult=1,
  in_size=64, N_layers=3)`；input/output `[B,20,32,64,3]`；真实输入为 raw `u/v`
  加 zero pressure channel，输出恢复为 `u/v/p`，scorer 只按 target 的 measured channels
  计算 Rel-L2/TKE/MVPE。
- Inference source: implementation SHA-256
  `73724987b0f471d74c803f07ce866273b76a1faedec2c1c6b31e75aa1db588b1`；运行时 source
  commit 参数为 `c51b2fbf6d8656c455872d68bb394106f2de18a1`，该 commit 上已有与本任务无关
  dirty/untracked work；脚本随后作为本次提交的一部分纳入 Git。

## 2. Frozen protocol and commands

- Manifest SHA-256: `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`。
- Split: exactly 50 train / 16 dev trajectories；final/private-test 未读取。
- Window: complete `20 input → 20 target`，stride `20`，spatial subsample `2`，实际
  runtime H×W=`32×64`；2052 train / 659 dev windows。
- Smoke command (GPU container, one window per manifest trajectory):
  `docker run --rm --gpus all -v /tmp/realpde_frozen_cno_incremental_probe.py:/probe.py:ro -v /tmp/id_seed20260901.json:/manifest.json:ro -v /tmp/realpde_t1_kit_20260902/realpde_t1_starting_kit_v9:/kit:ro -v /home/chyfuture/RealPDE_data:/data:ro -v /home/chyfuture/realpde_runs:/runs:ro -v /tmp/realpde_cno_probe_smoke_e0:/out realpde-pytorch-h5py:0831 python /probe.py --data-root /data/p0ab_real_h5_20260830 --manifest /manifest.json --kit-root /kit --checkpoint /runs/loss_opt_v9_20260901_run1/long_E0_s20260901/model_best.pth --out-dir /out --batch-size 8 --max-windows-per-trajectory 1 --experiment-id T1-ID-FE-INCR-FROZEN-CNO-E0-RIDGE-S20260902`。
  Smoke confirmed `cuda:0`, shape `[20,32,64,3]`, checkpoint SHA above, and scorer import.
- Formal inference + ridge command (the script performs frozen-CNO inference and CPU ridge in one run):
  `docker run --rm --gpus all -v /tmp/realpde_frozen_cno_incremental_probe.py:/probe.py:ro -v /tmp/id_seed20260901.json:/manifest.json:ro -v /tmp/realpde_t1_kit_20260902/realpde_t1_starting_kit_v9:/kit:ro -v /home/chyfuture/RealPDE_data:/data:ro -v /home/chyfuture/realpde_runs:/runs:ro -v <remote-artifact-dir>:/out realpde-pytorch-h5py:0831 python /probe.py --data-root /data/p0ab_real_h5_20260830 --manifest /manifest.json --kit-root /kit --checkpoint /runs/loss_opt_v9_20260901_run1/long_E0_s20260901/model_best.pth --out-dir /out --batch-size 32 --code-commit c51b2fbf6d8656c455872d68bb394106f2de18a1 --experiment-id T1-ID-FE-INCR-FROZEN-CNO-E0-RIDGE-S20260902 --baseline-family "registered T1-ID-LOSS-E0-90M model_best; frozen weights"`。
- Ridge definition: per-pixel residual target `target - frozen CNO prediction`; four fixed
  groups; feature mean/std fitted on train rows only (`4,202,496` rows); centered standardized
  closed form `(XᵀX + 1e-2·n·I)⁻¹XᵀY`; no dev alpha/threshold selection. TKE-proxy and vorticity
  are not independent inputs.
- Total formal runtime: `834.0 s` (13.9 min), GPU inference plus CPU probe; within 45-min budget.

## 3. Baseline and probe results

Frozen CNO raw dev (before ridge): Rel-L2 `0.168923`, TKE `0.538475`, MVPE `0.136146`.
This exactly matches the registered E0 reference to reported precision.

| package | input dim | dev Rel-L2 | dev TKE | dev MVPE | Δ vs Raw-Control (Rel/TKE/MVPE) |
|---|---:|---:|---:|---:|---:|
| Raw-Control | 2 | 0.168162 | 0.594538 | 0.135999 | 0 / 0 / 0 |
| Raw+Temporal (mean/std/delta) | 8 | 0.161318 | 0.608969 | 0.135610 | -0.006843 / +0.014432 / -0.000390 |
| Raw+Spatial (4 primitive gradients) | 6 | 0.166255 | 0.624081 | 0.135474 | -0.001907 / +0.029543 / -0.000525 |
| Raw+Temporal+Spatial | 12 | 0.159876 | 0.631270 | 0.135046 | -0.008286 / +0.036732 / -0.000954 |

Delta is candidate minus Raw-Control; negative is lower error. Trajectory-macro means and
trajectory-level win rates versus Raw-Control:

| package | macro Rel/TKE/MVPE | win rate Rel/TKE/MVPE |
|---|---|---|
| Raw+Temporal | `0.160278 / 0.608021 / 0.134614` | `1.000 / 0.313 / 0.563` |
| Raw+Spatial | `0.165185 / 0.624221 / 0.134475` | `1.000 / 0.125 / 0.750` |
| Raw+Temporal+Spatial | `0.158859 / 0.630452 / 0.134033` | `1.000 / 0.250 / 0.625` |

Joint versus Temporal (tests Spatial beyond Temporal):

- Window-micro delta Rel/TKE/MVPE: `-0.001442 / +0.022301 / -0.000564`.
- Trajectory-level win rate Rel/TKE/MVPE: `0.938 / 0.000 / 0.688`.
- Thus Spatial retains independent Rel-L2 and MVPE signal beyond Temporal, but the added
  signal is consistently harmful on TKE under this frozen ridge probe.

## 4. Comparison and interpretation

| source | Temporal | Spatial | interpretation |
|---|---|---|---|
| PERSIST ridge probe (`T1-ID-FE-INCR-PERSIST-RIDGE-S20260902`) | all three metrics improved; wins Rel/TKE/MVPE `0.875/1.000/0.938` | all three improved modestly; wins `0.812/1.000/0.750` | feature information looked uniformly positive against weak persistence |
| Frozen strong CNO E0 probe (this task) | Rel/MVPE improve, TKE worsens; wins `1.000/0.313/0.563` | Rel/MVPE improve, TKE worsens; wins `1.000/0.125/0.750` | residual information remains, but not stable across all protected metrics |
| Historical FE-01 / FE-02 fusion | FE-01 did not beat Raw-Control; FE-02 had Rel/TKE signal with MVPE trade-off and was NO-GO | historical fusion implementation effect only | do not equate fusion failure with no feature information |

This is the required conflict label: **`FEATURE_VALUE_POSITIVE_BUT_FUSION_HISTORY_NEGATIVE`**
for Rel-L2/MVPE directions, with an explicit TKE trade-off. The CNO probe does not prove
universal nonlinear fusion value and does not authorize training.

## 5. Shortlist and stop state

Using a conservative all-three-metric protection rule:

- Temporal: **`LOW_INCREMENTAL_VALUE`** as a uniformly safe package; retain only as a
  review-only fusion candidate because Rel-L2 is stable and MVPE slightly improves.
- Spatial: **`LOW_INCREMENTAL_VALUE`** as a uniformly safe package; it has independent
  Rel-L2/MVPE value beyond Temporal but a clear TKE penalty.
- Temporal+Spatial: **`LOW_INCREMENTAL_VALUE`** under the same rule; best Rel-L2/MVPE but
  largest TKE degradation.

Protocol conclusion: `STOP` for automatic fusion training; final execution state:
`REVIEW_REQUIRED`. ChatGPT/Sol should return one bounded next action after review.

## 6. Artifacts

Local repository-relative artifact directory:
`../artifacts/fe_incremental_probe_cno_e0_s20260902/`

- `summary.json`: provenance, counts, metrics, macro means, win rates, Joint-vs-Temporal.
- `window_metrics.csv`: 659 dev windows with frozen baseline and all corrected fields.
- `trajectory_metrics.csv`: 16 dev trajectory macro rows.
- `ridge_raw_control.npz`, `ridge_raw_temporal.npz`, `ridge_raw_spatial.npz`,
  `ridge_raw_temporal_spatial.npz`: coefficients and train-only normalizers.
- `report.md`: compact result table.

Generated outputs remain outside Git. STATUS and this handoff are the only pushed coordination
artifacts needed for review.
