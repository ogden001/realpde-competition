# MF-01 Deep Diagnostic

Status: `REVIEW_REQUIRED` / `MF01_NO_GO`

## Scope and provenance

This is an offline replay of the existing matched `Control@1500` and
`MF-01@1500` Dev predictions. No model or checkpoint was changed, no training
was run, and no locked-final/private-test or Codabench data was accessed.

- Control: `T1-ID-MF01-CONTROL-S20260904`
- Candidate: `T1-ID-MF01-S20260904`
- Manifest SHA-256: `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`
- Replay shape: `(659, 20, 32, 64, 3)`; 16 Dev trajectories; 20 horizons.
- Analysis script: `tools/analyze_mf01_deep.py`
- Small outputs: `artifacts/mf01_deep_analysis_20260904/`

## Verified facts

MF-01 official 1500-update errors improve Rel-L2 by 2.72% and MVPE by 2.80%,
but worsen TKE by 1.79%. The target-side-independent reconstruction diagnostics
also improve: mean-field error changes from `0.144862` to `0.141107`, and
fluctuation-field error from `1.519750` to `1.479491`; fluctuation diagnostic
wins are `16/16`. Thus the TKE regression is not explained by a broad increase
in pointwise fluctuation reconstruction error.

### By trajectory

`by_trajectory.csv` contains all requested profile descriptors, official
metrics, mean/fluctuation diagnostics, amplitude and energy ratios, and every
candidate-minus-control delta. Lower official/diagnostic errors are better.

TKE-best three (most negative ΔTKE): `20325_5` (ID, `-0.052832`), `10125_0`
(Boundary, `-0.027761`), `8850_20` (ID, `-0.019232`). TKE-worst three:
`26700_0` (OOD-like, `+0.050936`), `24150_10` (ID, `+0.049688`), and
`20325_20` (Boundary, `+0.020855`).

By Dataset Profile group, mean ΔTKE is `+0.000145` for 9 ID cases,
`-0.007972` for 3 Boundary cases, and `+0.015699` for 4 OOD-like cases.
This is evidence of an OOD-like shift in risk, but not concentration solely in
OOD: the largest bad case is OOD-like while a large bad case is ordinary ID,
and an OOD-like case (`3750_0`) is TKE-better.

Exploratory Spearman correlations of ΔTKE with input descriptors are positive:
fluctuation RMS `0.638`, delta mean `0.624`, gradient/vorticity/strain each
`0.603`; nearest-Train distance is only `0.091`. With 16 cases these are
descriptive associations, not significance or causality claims.

### By horizon

MF has lower mean velocity RMSE at 19/20 horizons and lower mean fluctuation
RMSE at 13/20 horizons. Fluctuation-error deltas are positive at horizons
8–13 and 19, so MF is not uniformly better in the middle/late rollout. The
mean per-frame TKE-amplitude ratio is lower for MF than Control at every
horizon (`0.514` vs `0.548` averaged over horizons 1–10; `1.321` vs `1.392`
over horizons 11–20). The ratio is not monotonically collapsing with horizon;
both methods become more energetic in the late frames because of the fixed
window diagnostic. Therefore the data support lower predicted fluctuation
energy, but do not establish a uniquely long-horizon-origin TKE failure.

`by_horizon.csv` contains the full 20-step table.

### Energy calibration

MF predicted/target fluctuation RMS ratio is below Control for all 16
trajectories. The MF ratios range from `0.939` to `1.459`; the corresponding
TKE ratios range from `0.879` to `2.116`. Diagnostic-only `alpha*`, defined as
the nonnegative scalar minimizing the TKE-map error after
`prediction = mean + alpha * fluctuation`, has:

`min=0.762`, `p25=0.912`, `median=0.970`, `p75=1.023`, `max=1.144`.

Group medians are ID `0.964`, Boundary `0.920`, OOD-like `1.027`. This is not
an approximately uniform scalar bias: the direction and magnitude vary by
trajectory, and the OOD-like group is wider/shifted. `alpha*` is retrospective
diagnosis only and was not used for tuning.

### By spatial region

The six representative map sets are in `artifacts/mf01_deep_analysis_20260904/figures/`.
Each has target TKE, Control absolute TKE error, MF absolute TKE error, and
`MF−Control` improvement (positive means MF reduces absolute error). Regions
are defined only by target TKE quantiles: top 20% and bottom 20%; no body mask
is inferred.

For bad cases `26700_0` and `24150_10`, MF is worse than Control in the high
target-TKE region by mean absolute-error differences `+7.26e-6` and
`+1.08e-4`, respectively, while it improves low-energy regions. `20325_20`
is nearly neutral in the high-energy region and slightly worse in the low one.
The good cases are mixed: `8850_20` improves high-energy regions but worsens
low-energy regions; `10125_0` improves high-energy regions but is slightly
worse at low energy; `20325_5` is slightly worse at high energy while improving
low energy. Spatial energy-structure/amplitude errors therefore contribute
to bad cases, but do not form one universal map pattern.

## Mechanism summary

These labels describe evidence in this replay, not validated causal claims.

| Hypothesis | Status | Evidence |
|---|---|---|
| `GLOBAL_AMPLITUDE_CALIBRATION` | `PARTIALLY_SUPPORTED` | MF amplitude is lower than Control for every case and median `alpha*=0.970`, but alpha spread and group shifts are substantial. |
| `LONG_HORIZON_VARIANCE_COLLAPSE` | `PARTIALLY_SUPPORTED` | MF has lower framewise TKE amplitude at every horizon, but no monotonic late-horizon collapse; fluctuation RMSE is mixed after t+8. |
| `HIGH_DYNAMIC_TAIL_FAILURE` | `PARTIALLY_SUPPORTED` | ΔTKE tracks fluctuation/delta/gradient tails positively, but good and bad cases occur across the tail groups. |
| `OOD_RELATED_FAILURE` | `PARTIALLY_SUPPORTED` | OOD-like mean ΔTKE is worse than ID, but `24150_10` is ID-bad and `3750_0` is OOD-like-good. |
| `SPATIAL_ENERGY_STRUCTURE_ERROR` | `PARTIALLY_SUPPORTED` | Two bad cases show clear high-energy-region worsening; good cases and `20325_20` are mixed. |
| `INCONCLUSIVE` | `SUPPORTED` | No single amplitude, horizon, distribution, or spatial explanation accounts for all 16 trajectory outcomes. |

## Interpretation boundary

The strongest evidence-backed explanation is: MF-01 improves mean and field
reconstruction while changing the fluctuation representation toward lower and
trajectory-dependent energy, which reduces TKE-map fidelity in some energetic
regions. The official TKE conflict is therefore a statistic/structure issue,
not a simple fluctuation-Rel-L2 failure. A causal decomposition between
amplitude and spatial phase/structure would require a separately pre-registered
control, but this task does not authorize designing or starting MF-02.

`NEXT_ACTION = REVIEW_REQUIRED`
