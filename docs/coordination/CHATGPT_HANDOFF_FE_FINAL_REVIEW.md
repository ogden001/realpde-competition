# Track 1 Feature Engineering — Final Review Handoff

**Final state: `CLOSED / REVIEW_REQUIRED`**
**Feature Discovery: `CLOSED`**
**Protocol conclusion: `STOP` (no new Feature or Fusion execution) / `REVIEW_REQUIRED`**

请从已连接的 GitHub 仓库读取本文件，并按 Track 1 V3 协议完成 review。本文是
FE-DATA-01、Spatial diagnostic、PERSIST incremental probe、Frozen-CNO incremental
probe 以及历史 FE-01/FE-02 的最终收口记录。所有数值来自已经登记的 handoff、registry
和既有本地产物；本次收口没有重跑 inference、ridge、训练或 GPU job，也没有访问
locked-final/private-test 或 Codabench。

## 1. Frozen scope and provenance

- Manifest SHA-256: `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`。
- Split: 50 train / 16 dev / 16 locked-final trajectories. 本项目只读取 train/dev。
- Runtime windows: complete 20 input → 20 target frames, stride 20. FE-DATA/Spatial
  diagnostic used raw `H×W=64×128` and 2,102/675 train/dev input windows;
  incremental probes used scorer/runtime `H×W=32×64` and 2,052/659 complete windows。
- Official starting-kit v9 scorer SHA-256:
  `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`。
- The local registry contains sensitive absolute paths and unrelated dirty work and is
  intentionally not pushed. This handoff contains the review facts needed without local
  artifacts.

### Key experiment IDs and implementation provenance

| Experiment | Purpose / state | Code or source provenance |
|---|---|---|
| `T1-ID-FE-DATA01-B1-S20260902` | Basic feature distribution; `DONE / REVIEW_REQUIRED` | Existing handoff `CHATGPT_HANDOFF_FE_DATA01.md`; report/CSV local only |
| `T1-ID-FE-SPATIAL-DATA01-S20260902` | Four pixel gradients + derived vorticity; `DONE / REVIEW_REQUIRED` | `tools/realpde_spatial_diagnostic_batch1.py`, SHA-256 `2fa8cd5b2b33532c365673f04cd621e6d14f3b1188b6538ca56865306ab267ad` |
| `T1-ID-FE-INCR-PERSIST-RIDGE-S20260902` | Minimal supervised residual probe; `DONE / REVIEW_REQUIRED` | `tools/realpde_incremental_value_probe.py`, SHA-256 `6e37aa883c6080df3ce60b15173f1d451002a50694c55aaff47605ca4eea3567`; run HEAD `442584e4e7ada3ad6472a055295a3dfbb917da22` with unrelated dirty/untracked work |
| `T1-ID-FE-INCR-FROZEN-CNO-E0-RIDGE-S20260902` | Frozen strong-CNO residual probe; `DONE / REVIEW_REQUIRED` | implementation SHA-256 `73724987b0f471d74c803f07ce866273b76a1faedec2c1c6b31e75aa1db588b1`; run source commit `c51b2fbf6d8656c455872d68bb394106f2de18a1` had unrelated dirty/untracked work; handoff code was committed at `400757b` |

Frozen-CNO baseline provenance: registered `T1-ID-LOSS-E0-90M-S20260901`, historical
`OFFICIAL_WARM_START` E0 `model_best.pth`; artifact-relative identifier
`loss_optimization_v9_20260901_run1/raw/long_E0_s20260901/model_best.pth`; checkpoint
SHA-256 `5d02c8da5bcbcbcc47917b0021b1007b2036931a3a51483772f7607deeb4aff6`.
Protocol is official CNO3d `3→3`, input/output `[B,20,32,64,3]`, raw `u/v` plus zero
pressure input and `u/v/p` output. Formal frozen-CNO inference plus CPU ridge took
`834.0 s` (13.9 min). The exact smoke and formal commands are recorded below and in
`docs/coordination/CHATGPT_HANDOFF_FE_INCREMENTAL_CNO_E0.md`.

The PERSIST probe ran for approximately `251 s` (4.2 min) on CPU. The Spatial diagnostic
used the existing completed run and its implementation SHA above; no diagnostic was
re-run for this closure.

## 2. Feature catalog frozen at closure

### Primitive information families

1. **Raw:** `u`, `v`.
2. **Temporal:** per-channel `mean`, `std`, and recent `delta` (last frame minus the
   previous frame).
3. **Spatial:** `du_dx_pixel`, `du_dy_pixel`, `dv_dx_pixel`, `dv_dy_pixel`.

Spatial Definition is frozen as pixel-space finite difference with spacing 1: centered
difference at interior pixels; first-order forward difference on index 0 and first-order
backward difference on the last index, independently for rows and columns. No smoothing,
clipping, normalization, coordinates, mask, target, CFD, Re/AoA or private data was added.

### Derived / re-expression families

- **speed:** near-re-expression of the raw channels; Pearson(speed, `abs(u)`) was
  `0.99991` train and `0.99992` dev.
- **u²/v² fluctuation energy:** strictly redundant with `std²` to float32 rounding
  (`std_u_20² = u2_prime_mean`, and the corresponding `v` identity).
- **TKE_proxy:** derived physical summary, not an independent primitive input.
- **vorticity_pixel:** derived spatial summary
  `dv_dx_pixel - du_dy_pixel`, not an extra primitive input. Its Pearson/Spearman
  correlation with that exact expression is `1.0/1.0`.

No new feature family is authorized by this review. In particular, TKE and vorticity were
not supplied as additional independent dimensions in either residual probe.

## 3. Data-side evidence (FE-DATA-01 + Spatial)

The FE-DATA-01 handoff was reconciled with the current local report/CSV. Correct train/dev
p95 values are `std_u_20 = 0.03260 / 0.03767` and
`TKE_proxy_input = 0.0007264 / 0.0009037`. Earlier handoff values came from an early
late-window-biased slot-replacement reservoir; the corrected report uses the bounded
priority reservoir while moments/counts use all values. No data was regenerated for this
closure.

Spatial value-level mean/p95 (train / dev) was:

| Feature | Train mean / p95 | Dev mean / p95 |
|---|---:|---:|
| `du_dx_pixel` | `0.0007757 / 0.009284` | `0.0008058 / 0.01028` |
| `du_dy_pixel` | `-0.0004949 / 0.01444` | `-0.0005389 / 0.01526` |
| `dv_dx_pixel` | `-0.00007716 / 0.003081` | `-0.00007935 / 0.002974` |
| `dv_dy_pixel` | `0.0001030 / 0.003894` | `0.0001059 / 0.003779` |
| `vorticity_pixel` | `0.0004177 / 0.01417` | `0.0004595 / 0.01506` |

Outer-image-edge/interior `abs_mean` ratios (train / dev) were `du_dx 0.800/0.822`,
`du_dy 0.615/0.633`, `dv_dx 0.572/0.573`, and `dv_dy 0.768/0.749`; all corresponding
`abs_p95` ratios were below `0.82`. Therefore the gradients are not primarily an outer
image-edge artifact under the frozen edge rule. Train/dev scale changes are descriptive
and mild; no significance gate was applied.

## 4. Existing supervised probe evidence

Both probes used the same four fixed packages, train-only feature normalization and
closed-form per-pixel ridge residual correction. Dimensions are per pixel and include Raw
Control: 2, Raw+Temporal: 8, Raw+Spatial: 6, Joint: 12. No dev tuning of alpha was done.

### PERSIST baseline (`T1-ID-FE-INCR-PERSIST-RIDGE-S20260902`)

The deterministic PERSIST prediction repeats the last observed `u/v` frame over the 20
future frames. Dev window-micro errors (Rel-L2 / TKE / MVPE):

| Package | Dim | Rel-L2 | TKE | MVPE | Δ vs Raw-Control (Rel / TKE / MVPE) |
|---|---:|---:|---:|---:|---:|
| Raw-Control | 2 | 0.131353 | 0.987575 | 0.130701 | 0 / 0 / 0 |
| Raw+Temporal | 8 | 0.118340 | 0.940578 | 0.109454 | -0.013013 / -0.046997 / -0.021247 |
| Raw+Spatial | 6 | 0.130241 | 0.973685 | 0.129355 | -0.001112 / -0.013889 / -0.001346 |
| Raw+Temporal+Spatial | 12 | 0.116346 | 0.936060 | 0.107228 | -0.015007 / -0.051515 / -0.023473 |

Trajectory-level win rates versus Raw-Control (Rel / TKE / MVPE) were Temporal
`0.875 / 1.000 / 0.938`, Spatial `0.812 / 1.000 / 0.750`, and Joint
`0.938 / 1.000 / 1.000`. Thus PERSIST shows three-metric, same-direction improvement.

### Frozen strong CNO (`T1-ID-FE-INCR-FROZEN-CNO-E0-RIDGE-S20260902`)

Frozen CNO raw dev is Rel-L2 / TKE / MVPE `0.168923 / 0.538475 / 0.136146`, matching
the registered E0 reference. The corrected-field results are:

| Package | Dim | Rel-L2 | TKE | MVPE | Δ vs Raw-Control (Rel / TKE / MVPE) |
|---|---:|---:|---:|---:|---:|
| Raw-Control | 2 | 0.168162 | 0.594538 | 0.135999 | 0 / 0 / 0 |
| Raw+Temporal | 8 | 0.161318 | 0.608969 | 0.135610 | -0.006843 / +0.014432 / -0.000390 |
| Raw+Spatial | 6 | 0.166255 | 0.624081 | 0.135474 | -0.001907 / +0.029543 / -0.000525 |
| Raw+Temporal+Spatial | 12 | 0.159876 | 0.631270 | 0.135046 | -0.008286 / +0.036732 / -0.000954 |

Trajectory-macro means (Rel / TKE / MVPE) were Temporal `0.160278 / 0.608021 / 0.134614`,
Spatial `0.165185 / 0.624221 / 0.134475`, and Joint `0.158859 / 0.630452 / 0.134033`.
Trajectory win rates versus Raw-Control were Temporal `1.000 / 0.313 / 0.563`, Spatial
`1.000 / 0.125 / 0.750`, and Joint `1.000 / 0.250 / 0.625`.

Joint versus Temporal (the independent Spatial increment) was Rel/TKE/MVPE
`-0.001442 / +0.022301 / -0.000564`, with trajectory win rates `0.938 / 0.000 / 0.688`.
Spatial therefore retains Rel/MVPE residual signal beyond Temporal, but adds a consistent
TKE penalty in this frozen ridge probe.

The frozen ridge correction itself worsens TKE even for Raw-Control (`0.538475` frozen
raw → `0.594538` Raw-Control ridge). Consequently, the TKE trade-off cannot be attributed
simply to Temporal/Spatial having no information; it is a limitation of this linear
correction/protection objective.

## 5. Historical fusion evidence and final interpretation

Historical FE-01/FE-02 were neural fusion implementations, not the same residual probe.
Their recorded reference values were FE-00R Raw-Control `0.183850 / 0.636973 / 0.133649`,
FE-01 `0.189399 / 0.641316 / 0.140459`, and FE-02 `0.177478 / 0.640643 / 0.140967`
(Rel-L2 / TKE / MVPE). FE-01 did not beat Raw-Control; FE-02 showed Rel/TKE signal with
an MVPE trade-off and was NO-GO. These are **fusion implementation evidence**, not proof
that the underlying features contain no information.

Required conflict label: **`FEATURE_VALUE_POSITIVE_BUT_FUSION_HISTORY_NEGATIVE`**.
The PERSIST probe is uniformly positive; the frozen strong CNO probe preserves Rel/MVPE
residual signal but damages TKE under every ridge package. This distinguishes feature
information value from the historical fusion implementation outcome and does not imply
universal nonlinear-model value.

## 6. Frozen shortlist and closure decision

| Package / family | Final value label | Frozen shortlist status |
|---|---|---|
| Temporal (`mean/std/delta`) | `SIGNAL_POSITIVE_BUT_FUSION_NOT_JUSTIFIED` | Retain as review-only candidate for a separately authorized fusion project; no promotion now |
| Spatial (four primitive gradients) | `WEAK_SIGNAL_POSITIVE_BUT_FUSION_NOT_JUSTIFIED` | Retain as review-only candidate; independent Rel/MVPE signal is weak and TKE-sensitive |
| Temporal + Spatial | Best Rel/MVPE residual signal, but TKE trade-off prevents automatic promotion | Retain only as a bounded review candidate; no automatic fusion/training |
| TKE proxy / vorticity | Derived summaries, not new primitive information | Not independent inputs; do not expand catalog |

**Final decision:** `STOP` all new Feature experiments, catalog expansion, ridge tuning,
model training, locked-final access and Codabench work. Feature Discovery is now
**`CLOSED`**. A future reopening must be a new independent Feature Fusion sub-project with
the explicit objective: **“利用 Temporal/Spatial residual signal，同时保护 TKE/fluctuation structure”**.
This closure does not authorize that project.

## 7. Commands and artifact paths (for auditability)

Existing commands (not re-run during closure):

- Spatial diagnostic, from workspace root:
  `python code/tools/realpde_spatial_diagnostic_batch1.py --data-archive data/train_real.tar.gz --manifest artifacts/loss_optimization_v9_20260901_run1/evidence/manifests/id_seed20260901.json --out-dir artifacts/spatial_diagnostic_batch1`
- PERSIST probe, from workspace root:
  `python code/tools/realpde_incremental_value_probe.py --data-root artifacts/sim2real_gap_full/_full_cache/real --manifest artifacts/loss_optimization_v9_20260901_run1/evidence/manifests/id_seed20260901.json --kit-root <temporary extracted starting-kit> --out-dir artifacts/fe_incremental_probe_s20260902`
- Frozen-CNO smoke command (one window per trajectory):
  `docker run --rm --gpus all -v /tmp/realpde_frozen_cno_incremental_probe.py:/probe.py:ro -v /tmp/id_seed20260901.json:/manifest.json:ro -v /tmp/realpde_t1_kit_20260902/realpde_t1_starting_kit_v9:/kit:ro -v /home/chyfuture/RealPDE_data:/data:ro -v /home/chyfuture/realpde_runs:/runs:ro -v /tmp/realpde_cno_probe_smoke_e0:/out realpde-pytorch-h5py:0831 python /probe.py --data-root /data/p0ab_real_h5_20260830 --manifest /manifest.json --kit-root /kit --checkpoint /runs/loss_opt_v9_20260901_run1/long_E0_s20260901/model_best.pth --out-dir /out --batch-size 8 --max-windows-per-trajectory 1 --experiment-id T1-ID-FE-INCR-FROZEN-CNO-E0-RIDGE-S20260902`
- Frozen-CNO formal command (the `<remote-artifact-dir>` mount is intentionally
  redacted; all other arguments are exact):
  `docker run --rm --gpus all -v /tmp/realpde_frozen_cno_incremental_probe.py:/probe.py:ro -v /tmp/id_seed20260901.json:/manifest.json:ro -v /tmp/realpde_t1_kit_20260902/realpde_t1_starting_kit_v9:/kit:ro -v /home/chyfuture/RealPDE_data:/data:ro -v /home/chyfuture/realpde_runs:/runs:ro -v <remote-artifact-dir>:/out realpde-pytorch-h5py:0831 python /probe.py --data-root /data/p0ab_real_h5_20260830 --manifest /manifest.json --kit-root /kit --checkpoint /runs/loss_opt_v9_20260901_run1/long_E0_s20260901/model_best.pth --out-dir /out --batch-size 32 --code-commit c51b2fbf6d8656c455872d68bb394106f2de18a1 --experiment-id T1-ID-FE-INCR-FROZEN-CNO-E0-RIDGE-S20260902 --baseline-family "registered T1-ID-LOSS-E0-90M model_best; frozen weights"`
- Frozen-CNO ridge configuration: residual target is target minus frozen prediction;
  train-only standardization over `4,202,496` rows; per-pixel closed form
  `(XᵀX + 1e-2·n·I)⁻¹XᵀY`; four fixed packages only; no dev alpha/threshold selection.

Repository-relative local evidence (generated outputs intentionally not committed):

- `../artifacts/feature_summary_batch1/`
- `../artifacts/spatial_diagnostic_batch1/`
- `../artifacts/fe_incremental_probe_s20260902/`
- `../artifacts/fe_incremental_probe_cno_e0_s20260902/`

This final review document and `docs/coordination/STATUS.md` are the safe GitHub handoff
artifacts. The final closure commit SHA is reported in the assistant response after push.

## 8. Required review response

Please review this closure under Track 1 V3 and return one bounded next action only. Until
that review, execution remains `REVIEW_REQUIRED`; do not request a ZIP or local artifacts.
