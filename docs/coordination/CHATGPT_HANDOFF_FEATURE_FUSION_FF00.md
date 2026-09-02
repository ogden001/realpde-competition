# Track 1 Temporal / Spatial Prior Fusion — FF-00 Handoff

**Task:** `FF-00` — Fusion Protocol & Baseline Freeze
**Scope:** protocol design and provenance audit only; no model training or new
performance result was produced in this task.
**Protocol conclusion:** `REVIEW_REQUIRED`
**Execution state:** `REVIEW_REQUIRED`
**Provenance exception:** `BASELINE_PROVENANCE_EXCEPTION_ACCEPTED`
**Reason:** the historical strong-CNO checkpoint is accepted as an immutable
artifact baseline even though its training source commit is not recoverable. The
limitation is explicit below: this does not make the historical training process
fully reproducible from source.

## A. Protocol Summary

### Baseline

The requested strong baseline remains the existing historical reference
`T1-ID-LOSS-E0-90M-S20260901`. It is an `OFFICIAL_WARM_START` / competition-oriented
reference, not the default `CLEAN` offline family. FF-00 does not switch to the
planned clean CNO reference and does not retrain the baseline.

`BASELINE_PROVENANCE_EXCEPTION_ACCEPTED` means that downstream Fusion candidates
may reuse this exact checkpoint. Every later Feature candidate and its matched
Raw-Control must share the same checkpoint SHA, downstream code/protocol, split,
scorer, optimizer/seed/budget policy, and checkpoint selection rule. Any later
change must be a separately recorded protocol variable.

### Split and manifest

- Manifest: `artifacts/loss_optimization_v9_20260901_run1/evidence/manifests/id_seed20260901.json`
- Manifest SHA-256: `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`
- `kind=id`, seed `20260901`, 82 total trajectories
- Split: 50 train / 16 dev / 16 locked-final trajectories
- Future Fusion experiments may read only the 50 train and 16 dev trajectories;
  locked-final and private test are not part of selection or FF-00 execution.
- Window protocol: complete 20 input → 20 target frames, stride 20
- Standard Fusion runtime shape: `H×W=32×64`; prior incremental probes had
  2052 train and 659 dev windows under this protocol.

### Scorer and runtime

- Official Track 1 Starting Kit v9 `scoring.py`
- Scorer SHA-256: `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`
- Model interface: `[B,20,32,64,3] → [B,20,32,64,3]`
- Input channels are raw `u`, raw `v`, and compatibility placeholder `p=0`.
  Pressure is not treated as a physical input in runtime.
- Output is `u/v/p`; official measured-channel metrics use the available velocity
  channels. Fusion features must be computable from the current 20 input frames
  and pixel tensor shape only.
- Runtime evaluation must include inference time and peak GPU memory. No Re/AoA,
  physical coordinates, geometry, body mask, CFD, future trajectory, target,
  locked-final or private-test information is allowed.

### Frozen feature packages

Only these packages are in scope; Feature Discovery remains closed.

| Package | Frozen contents | Runtime availability |
|---|---|---|
| `TEMPORAL6` | `mean_u_20`, `mean_v_20`, `std_u_20`, `std_v_20`, `delta_u`, `delta_v` | statistics of current 20 input frames; delta is last frame minus previous frame |
| `SPATIAL4` | `du_dx_pixel`, `du_dy_pixel`, `dv_dx_pixel`, `dv_dy_pixel` | pixel-space finite differences of the last observed input frame |
| `TEMPORAL6_SPATIAL4` | `TEMPORAL6 + SPATIAL4` | joint package |

Spatial finite differences are frozen at spacing 1: centered interior difference,
forward difference at index 0, and backward difference at the last index. No
smoothing, clipping, new normalization family, mask or coordinate channel is
added. `TKE_proxy` and `vorticity_pixel = dv_dx_pixel - du_dy_pixel` remain
derived summaries, not new primitive package dimensions.

### Budget

This FF-00 task used no training budget. For a later approved Fusion task:

- smoke: at most 10 minutes;
- one formal short candidate: at most 60 minutes;
- initial candidates per task: at most 3;
- no automatic long training, grid scan, locked-final audit or Codabench work.

### Unified metrics

Every approved candidate must report, against its matched Raw-Control and the
frozen strong raw baseline:

- official `Rel-L2`, `TKE`, `MVPE`;
- trajectory-macro Rel-L2/TKE/MVPE over all 16 dev trajectories;
- trajectory-level win rate versus the matched Raw-Control;
- metric deltas versus both controls;
- inference runtime, peak GPU memory, total parameters and added parameters;
- exact command, code commit, checkpoint SHA, manifest SHA and artifact path.

No candidate is selected from Rel-L2 alone.

### Gate

The existing project gates are reused structurally: fixed dev-only selection,
trajectory-level stability evidence, all three protected official metrics, and a
matched control for any capacity change. Prior numeric gates are specific to loss
or point-model tasks and are not copied into FF-00 as universal Fusion thresholds.

Recommended decision logic for ChatGPT/Sol review:

- `GO` only when Rel-L2 has a clear and stable improvement, MVPE does not show a
  systematic regression, TKE does not show a clear systematic regression, and the
  result cannot be explained by added capacity or runtime alone.
- `STOP` when a small Rel-L2 gain is bought with clear TKE degradation, gains are
  concentrated in only a few trajectories, Joint does not beat Temporal while
  adding substantial complexity, or dev retuning is required to preserve the
  claim.
- Any locked-final/private-test dependence is `STOP`.

The frozen ridge probe demonstrated the specific failure pattern to guard
against: Rel-L2/MVPE decrease while TKE increases.

## B. Baseline Provenance

| Field | Frozen fact |
|---|---|
| Experiment ID | `T1-ID-LOSS-E0-90M-S20260901` |
| State / family | historical reference; `OFFICIAL_WARM_START` |
| Checkpoint artifact | `loss_optimization_v9_20260901_run1/raw/long_E0_s20260901/model_best.pth` |
| Checkpoint SHA-256 | `5d02c8da5bcbcbcc47917b0021b1007b2036931a3a51483772f7607deeb4aff6` |
| Initialization | official `sim_real_ft/sim_real_cno.pth`; SHA-256 `82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61` |
| Architecture | vendored `CNO3d(in_dim=3, out_dim=3, out_dim_mult=1, in_size=64, N_layers=3)` |
| Input/output | raw `u/v` + zero `p`; `[B,20,32,64,3] → [B,20,32,64,3]` |
| Training recipe | `MSE + 0.05*TKE`; AdamW, `lr=1e-5`, batch 8, workers 2, seed `20260901`, 7498 updates, approximately 90.7 minutes; best iteration 7498 |
| Dev raw metrics | Rel-L2 `0.168923`; TKE `0.538475`; MVPE `0.136146` |
| Scorer | Starting Kit v9 `scoring.py`, SHA-256 as in Section A |
| Training source commit | **UNKNOWN / NOT RECOVERED** |
| Source artifact fallback | `artifacts/loss_optimization_v9_20260901_run1/source/realpde_loss_official_v9.py`, SHA-256 `6fdf6d2e3268098e3663fd93eb819322786e1524c5091d1e3d69b6ef0c06cb50` |
| Dirty-tree note | historical run metadata records paths, hashes and recipe but no Git SHA; the source artifact is a copied runner file, not a repository commit |

The provenance exception is accepted for checkpoint reuse because the checkpoint
SHA, artifact identifier, architecture, runtime, manifest and scorer are fixed.
The checkpoint may be reused as an immutable artifact baseline, but no report may
claim that its historical training process can be fully reproduced from source.
Downstream FF code must record its own Git commit or complete dirty-diff SHA and
must not inherit this missing-commit limitation.

## C. Frozen Feature Package Definitions

The package names and contents are frozen as follows:

```text
TEMPORAL6 =
    mean_u_20, mean_v_20,
    std_u_20,  std_v_20,
    delta_u,   delta_v

SPATIAL4 =
    du_dx_pixel, du_dy_pixel,
    dv_dx_pixel, dv_dy_pixel

TEMPORAL6_SPATIAL4 = TEMPORAL6 + SPATIAL4
```

All statistics are calculated from currently observed input only. Do not add
Local, Wake, FFT, Spectral, POD, Laplacian, higher-order differences, new temporal
variants, new spatial variants, pressure-derived quantities, or any target-side
quantity.

## D. Matched Raw-Control Definition

The primary causal comparison is **Feature Candidate vs Matched Raw-Control**, not
Feature Candidate vs the old baseline alone.

For every candidate that adds a branch, gate, adapter, encoder/decoder change or
parameters, the Raw-Control must:

1. use the identical modified graph, initialization policy, optimizer, seed,
   train/dev split, budget, checkpoint rule and inference path;
2. retain the same number of trainable parameters within the pre-registered
   matching tolerance, with any mismatch reported explicitly;
3. receive no `TEMPORAL6`, `SPATIAL4` or derived prior. If the branch requires a
   fixed channel width, feed a deterministic tiling/projection of the currently
   available raw `u/v/p=0` tensor to that branch so its shape and capacity are
   matched without injecting new physical information;
4. use the same normalization and serialization rules, fitted on train only.

The frozen strong raw baseline is the old-reference comparator. In the FF-02
matrix below, `B0` specifically means the matched Raw-Control for the modified
graph; keep that label distinct from the frozen strong baseline in result files so
added-capacity effects are not confused with Feature value.

## E. Loss Duplication Audit

The audit read the historical v9 runner/source, the generic differentiable loss
implementation, the v9 loss report, and the precision-injection review package.
The distinction below is between a completed objective experiment and a helper
that merely exists in code.

| Objective | Coverage | Experiment ID / source | Existing objective and conclusion | Worth doing again? |
|---|---|---|---|---|
| Mean consistency | **YES** | Historical E3 within `T1-ID-LOSS-E0-90M-S20260901`; v9 runner `E3` | `L_mean = Rel(mean_t(pred_uv), mean_t(target_uv))`, weight `0.05`, added to Rel/TKE/MVPE. E3 achieved the lowest Rel-L2 in its long run but had the worst TKE; it did not solve the trade-off. | **NO** for the same mean loss; only a materially different, approved formulation could reopen it. |
| Std / fluctuation consistency | **PARTIAL** | Historical E3; generic `code/tools/realpde_tke_finetune.py` / v9 runner | E3 used `L_fluct = Rel(pred - mean_t(pred), target - mean_t(target))`, weight `0.10`. This is temporal fluctuation-field consistency, not an explicit per-channel `std_u/std_v` moment loss. The observed E3 TKE trade-off means the same formulation is not a solution. | **REVIEW_REQUIRED** only for a distinct explicit std-moment objective; do not repeat E3 under a new name. |
| TKE consistency | **YES** | `T1-ID-LOSS-E0-90M-S20260901` E0/E1/E2/E3; precision package N0/N1/N2 | `L_TKE = Rel(TKE_map(pred), TKE_map(target))`, with the E0-family base weight `0.05`. E0 protected TKE better than Rel-dominant E1/E2/E3; later calibrated N0/N1/N2 retained the base and reported N2 as a three-metric Pareto improvement. | **NO** for repeating the same TKE term; retain it as the mandatory protection baseline for any future objective study. |
| Gradient consistency | **NO** as a completed v9 loss experiment | Generic helper only: `spatial_grad_rel_loss`; no registered material v9 run with nonzero gradient weight | The helper compares first-order x/y finite-difference fields. The older generic fine-tuner exposes a default gradient weight, but the registered official v9 loss studies and N0/N1/N2 did not establish a completed, isolated gradient-objective result. | **REVIEW_REQUIRED** as at most one bounded new objective candidate; exact definition/weight must be approved first. |
| Vorticity consistency | **NO** | No completed loss experiment; only derived feature diagnostic | No vorticity loss was trained or evaluated under the frozen official v9 loss protocol. `vorticity_pixel` is the deterministic derived expression `dv_dx_pixel - du_dy_pixel`, not a new primitive feature. | **REVIEW_REQUIRED** only if ChatGPT/Sol explicitly chooses it; it must use current-input pixel derivatives and a matched objective, with no new feature family. |

Historical exact formulas read during the audit:

```text
E0 = MSE + 0.05*TKE
E1 = Rel + 0.05*TKE
E2 = Rel + 0.05*TKE + 0.10*MVPE
E3 = Rel + 0.05*TKE + 0.10*MVPE + 0.05*Mean + 0.10*Fluct

N0 = MSE + 0.05*TKE
N1 = MSE + 0.05*TKE + 0.013757*Rel + 0.009757*MVPE
N2 = MSE + 0.05*TKE + 0.027514*Rel + 0.009757*MVPE
```

The precision-injection report concluded that N2 was the strongest historical
candidate, while the earlier E1/E2/E3 report concluded that Rel/MVPE-dominant
losses traded better point/mean-flow accuracy for worse TKE. These are loss
optimization results, not evidence that a Feature Fusion implementation works.

## F. Proposed Minimal Experiment Matrix

This is protocol design only. No row below was implemented or trained by FF-00.

### FF-01 — Feature Conditioning / Gating

| ID | Input to the modified module | Rule |
|---|---|---|
| G0 | matched Raw-Control | same latent gate/FiLM capacity, raw-only tensor |
| G1 | `TEMPORAL6` | feature conditions latent representation through a gate or scale/bias module; no direct output residual |
| G2 | `TEMPORAL6_SPATIAL4` | same as G1 with the frozen joint package |

Primary comparisons: `G1 vs G0`, then `G2 vs G1`. The latter tests the independent
Spatial increment while holding the conditioning design fixed.

### FF-02 — Multi-Branch Prior Fusion

| ID | Prior branch | Rule |
|---|---|---|
| B0 | matched Raw-Control | same lightweight branch and fusion capacity, raw-only tensor |
| B1 | `TEMPORAL6` | lightweight prior branch fused into the latent path |
| B2 | `TEMPORAL6_SPATIAL4` | same branch budget and fusion point as B1 |

Primary comparisons: `B1 vs B0`, then `B2 vs B1`. The prior branch must stay
lightweight, and total/added parameters must be reported.

### FF-03 — Feature-aware Objective

Input remains raw `u/v/p=0`; this is not an input-feature experiment. First reuse
the existing `MSE + 0.05*TKE` control and do not retrain it solely to fill a table.
The audit status defines the minimum future choices:

| ID | Objective candidate | Protocol status |
|---|---|---|
| O0 | existing Mean/TKE coverage | no duplicate run; reuse historical evidence |
| O1 | explicit per-channel std-moment consistency | only non-duplicate fluctuation candidate currently justified for review |
| O2 | spatial gradient consistency | candidate only if approved; no prior completed v9 result |
| O3 | vorticity consistency | candidate only if approved; derived from current-input pixel gradients |

At most three initial FF-03 candidates may be selected. No objective weights,
checkpoint rule changes, or numeric promotion thresholds are frozen by FF-00.

## G. Risks / Open Questions for ChatGPT / Sol

1. The selected baseline is `OFFICIAL_WARM_START`, while the clean CNO reference is
   only `PLANNED`. Confirm whether Fusion should remain in the competition-oriented
   family or wait for a clean baseline.
2. Confirm the parameter-matching tolerance and whether the deterministic raw-only
   channel tiling/projection described in Section D is the preferred control.
3. Choose at most one bounded first Fusion direction and at most three candidates;
   do not launch all matrix rows automatically.
4. For FF-03, decide whether an explicit std-moment objective is sufficiently
   distinct from historical `L_fluct`, and whether Gradient or Vorticity should be
   prioritized. Do not treat helper-code existence as prior evidence.

## H. Protocol Conclusion

`REVIEW_REQUIRED`

The FF-00 protocol itself is documented, the feature catalog is frozen, the matched
control rule and metric/gate policy are specified, and the Loss duplication audit is
complete. `BASELINE_PROVENANCE_EXCEPTION_ACCEPTED` releases the immutable
checkpoint for downstream reuse while preserving the limitation that the historical
training source is `UNKNOWN / NOT RECOVERED`. No FF-01, FF-02 or FF-03 execution was
started. ChatGPT/Sol review is required before any downstream execution.
