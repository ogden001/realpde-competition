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
| Fluctuation tails | std_u p95 `0.01390/0.06699`; TKE proxy p95 `0.0001997/0.002485` (train/dev): descriptive tail shift worth monitoring, not a selection gate |
| Strict redundancy | `std_u_20² = u2_prime_mean`; `std_v_20² = v2_prime_mean` to float32 rounding |
| Near redundancy | speed vs abs(u) Pearson `0.99991` train / `0.99992` dev |
| Low correlation | recent delta vs corresponding 20-frame mean: Pearson magnitude below `0.10` in both splits |

Current data-side shortlist: **KEEP** raw u/v, mean/std, recent delta, u/v fluctuations, input-side TKE proxy; **WATCH** speed and TKE proxy; **LOW_VALUE** the two squared fluctuation-energy fields when their std is already present. This is not a model-effect claim.

## Local record

- Current state: `AWAITING_NEXT_ACTION` in `docs/coordination/STATUS.md`.
- Local registry contains the full append-only experimental record; this committed handoff contains the facts needed for a bounded review without exposing generated artifacts.
