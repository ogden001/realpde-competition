# ChatGPT / Sol Review Handoff — `T1-ID-FE-DATA01-B1-S20260902`

This is the GitHub-readable canonical handoff. Raw CSVs, generated reports and
archives intentionally remain outside Git under local `artifacts/` paths.

## Send this message in ChatGPT

> Please read this file from the connected GitHub repository and review the Track 1 Batch-1 runtime-feature diagnostic under the V3 coordination protocol. This is a completed, `CLEAN`, descriptive analysis only: frozen manifest `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`; train 50/dev 16; 20-frame, stride-20 input windows; no targets, CFD, Re/AoA, physical coordinates, private mask, locked final, training, or Codabench. `locked_final_accessed=false` is recorded in the local protocol evidence.
>
> The report finds complete finiteness, deterministic redundancy of variance-energy pairs, near-perfect speed/abs(u) correlation, and descriptive train/dev tail differences for v/fluctuation/delta/TKE-proxy features. Quantiles are explicitly bounded-reservoir estimates; moments and counts use all values. Please do not interpret this as model-effect evidence.
>
> Return exactly one bounded `NEXT_ACTION` in this form:
>
> ```text
> NEXT_ACTION
> goal:
> allowed data and resource budget:
> prohibited actions:
> acceptance criteria:
> required evidence / deliverables:
> ```
>
> Choose `GO`, `STOP`, or `REVIEW_REQUIRED`, state why, and do not authorize model training, locked-final access, private-test access, or Codabench unless you explicitly include it above.

## Review facts

| Item | Observation |
|---|---|
| Diagnostic windows | 2,102 train; 675 dev |
| Data quality | all 14 feature values finite; no numeric constant feature |
| Raw central distributions | u train/dev p50 `0.1628/0.3206`, p95 `0.3569/0.3665`; speed is correspondingly close in mean/p95 |
| Fluctuation tails | corrected std_u p95 `0.03260/0.03767`; corrected TKE proxy p95 `0.0007264/0.0009037` (train/dev): mild descriptive tail shift, not a selection gate |
| Strict redundancy | `std_u_20² = u2_prime_mean`; `std_v_20² = v2_prime_mean` to float32 rounding |
| Near redundancy | speed vs abs(u) Pearson `0.99991` train / `0.99992` dev |
| Low correlation | recent delta vs corresponding 20-frame mean: Pearson magnitude below `0.10` in both splits |

Current data-side shortlist: **KEEP** raw u/v, mean/std, recent delta, u/v fluctuations, input-side TKE proxy; **WATCH** speed and TKE proxy; **LOW_VALUE** the two squared fluctuation-energy fields when their std is already present. This is not a model-effect claim.

The earlier handoff values for std_u and TKE proxy were stale: they came from an
early slot-replacement reservoir that overweighted late windows. The local
report/CSV were regenerated with a bounded priority reservoir; all moments and
counts remained all-value statistics. The corrected values above match the
current local report/CSV.

## Spatial follow-up — `T1-ID-FE-SPATIAL-DATA01-S20260902`

The FE-DATA-01 correction was checked against the local CSV/report before this
follow-up. The corrected std_u p95 is `0.03260/0.03767` and corrected TKE proxy
p95 is `0.0007264/0.0009037` (train/dev); the prior handoff values were from an
early late-window-biased slot reservoir and were not reused.

Spatial definition is frozen as pixel-space finite difference with spacing 1:
centered difference at interior pixels; first-order forward difference on index
0 and first-order backward difference on the last index, independently for rows
and columns. No smoothing, clipping, normalization, coordinates, mask, target,
CFD, Re/AoA or locked-final data is used. Actual input shape is H×W=`64×128`.

Implementation: `tools/realpde_spatial_diagnostic_batch1.py` (SHA-256
`2fa8cd5b2b33532c365673f04cd621e6d14f3b1188b6538ca56865306ab267ad`). Exact
command from the workspace root:
`python code/tools/realpde_spatial_diagnostic_batch1.py --data-archive data/train_real.tar.gz --manifest artifacts/loss_optimization_v9_20260901_run1/evidence/manifests/id_seed20260901.json --out-dir artifacts/spatial_diagnostic_batch1`.
The local evidence directory is `artifacts/spatial_diagnostic_batch1/` and
contains the definition JSON, value/trajectory/edge/correlation CSVs and report.

Full train/dev counts are 2,102/675 windows. Value-level mean/p95:

| Feature | train mean / p95 | dev mean / p95 |
|---|---:|---:|
| du_dx_pixel | `0.0007757 / 0.009284` | `0.0008058 / 0.01028` |
| du_dy_pixel | `-0.0004949 / 0.01444` | `-0.0005389 / 0.01526` |
| dv_dx_pixel | `-0.00007716 / 0.003081` | `-0.00007935 / 0.002974` |
| dv_dy_pixel | `0.0001030 / 0.003894` | `0.0001059 / 0.003779` |
| vorticity_pixel | `0.0004177 / 0.01417` | `0.0004595 / 0.01506` |

Image-edge sensitivity (outer-edge/interior abs_mean ratios, train/dev) is
`du_dx 0.800/0.822`, `du_dy 0.615/0.633`, `dv_dx 0.572/0.573`, and
`dv_dy 0.768/0.749`; corresponding abs_p95 ratios are all below 0.82. Thus
the image edge is not the dominant magnitude source under this definition.

Correlation diagnostics show `vorticity_pixel` versus `(dv_dx_pixel-du_dy_pixel)`
Pearson/Spearman `1.0/1.0` (deterministic identity); vorticity vs du_dy is about
`-0.978/-0.904` and vorticity vs dv_dx about `0.192/0.357` in train, with the
same qualitative pattern in dev. Spatial primitive scale changes are
descriptive and mild; no hard gate or threshold was applied.

Spatial shortlist: `du_dx_pixel` **KEEP**, `du_dy_pixel` **KEEP**, `dv_dx_pixel`
**KEEP**, `dv_dy_pixel` **KEEP**, `vorticity_pixel` **KEEP (derived summary)**.
The vorticity label means “retain as a convenient derived summary”, not “new
raw information”. Full evidence remains local under `../artifacts/spatial_diagnostic_batch1/`.

## Local record

- Current state: `AWAITING_NEXT_ACTION` in `docs/coordination/STATUS.md`.
- Local registry contains the full append-only experimental record; this committed handoff contains the facts needed for a bounded review without exposing generated artifacts.
