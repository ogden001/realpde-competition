# Track 1 Coordination Status

Updated: 2026-09-02

## Current Stage

Point Modeling: **POINT-LOCAL3-BALANCED-L001 COMPLETED / STOP_BALANCED_LOCAL3_EARLY**.
The bounded random-init loss-balance experiment (`MSE + 0.001*TKE`) failed its
1500-step screen; no old LOCAL3 checkpoint was reused and no locked-final or
Codabench access occurred.

Temporal / Spatial Prior Fusion: **FF-00 completed / REVIEW_REQUIRED**. Feature Discovery remains
**CLOSED**; no FF-01, FF-02 or FF-03 experiment is authorized.

FF-00 protocol design and the historical Loss duplication audit are complete.
`BASELINE_PROVENANCE_EXCEPTION_ACCEPTED` accepts
`T1-ID-LOSS-E0-90M-S20260901` as an immutable checkpoint artifact baseline despite
the training source commit being `UNKNOWN / NOT RECOVERED`. The checkpoint SHA,
manifest, scorer, architecture, runtime and metrics are recorded in the handoff.
This does not claim full source-level reproducibility of historical training.

All downstream FF candidates and matched Raw-Controls must share the same baseline
checkpoint and downstream code/protocol; each downstream run must record its own
code commit or complete dirty-diff SHA.

Handoff: `docs/coordination/CHATGPT_HANDOFF_FEATURE_FUSION_FF00.md`

## Latest Completed

`FF-00-FUSION-PROTOCOL-S20260902` — Fusion Protocol & Baseline Freeze — `COMPLETED` /
`REVIEW_REQUIRED`.

- `BASELINE_PROVENANCE_EXCEPTION_ACCEPTED`: the historical
  `T1-ID-LOSS-E0-90M-S20260901` checkpoint is frozen as an immutable artifact baseline.
- Training source commit remains `UNKNOWN / NOT RECOVERED`; this is a documented
  provenance limitation, not a claim of full historical source reproducibility.
- Handoff: `docs/coordination/CHATGPT_HANDOFF_FEATURE_FUSION_FF00.md`.
- No training, GPU job, locked-final/private-test access, Codabench submission or
  FF-01/FF-02/FF-03 execution occurred.

`FE-FINAL-REVIEW-S20260902` — Feature Engineering final review — `CLOSED` /
`REVIEW_REQUIRED`.

- Consolidates `T1-ID-FE-DATA01-B1-S20260902`,
  `T1-ID-FE-SPATIAL-DATA01-S20260902`,
  `T1-ID-FE-INCR-PERSIST-RIDGE-S20260902`,
  `T1-ID-FE-INCR-FROZEN-CNO-E0-RIDGE-S20260902`, and historical FE-01/FE-02.
- Final labels: Temporal `SIGNAL_POSITIVE_BUT_FUSION_NOT_JUSTIFIED`; Spatial
  `WEAK_SIGNAL_POSITIVE_BUT_FUSION_NOT_JUSTIFIED`; Temporal+Spatial has the best
  Rel/MVPE residual signal but its TKE trade-off prevents automatic promotion.
- PERSIST improved Rel-L2/TKE/MVPE consistently. Frozen strong CNO retained Rel/MVPE
  residual signal but every ridge correction worsened TKE; Raw-Control ridge also
  worsened TKE, so this is not evidence that the Features contain no information.
- Historical FE-01/FE-02 outcomes are fusion-implementation evidence only. Required
  interpretation label: `FEATURE_VALUE_POSITIVE_BUT_FUSION_HISTORY_NEGATIVE`.
- GitHub-readable final review: `docs/coordination/CHATGPT_HANDOFF_FE_FINAL_REVIEW.md`.
- This closure did not run inference, ridge, training, locked-final/private-test access or
  Codabench. A future Fusion reopening must be a separately authorized project whose goal
  is “利用 Temporal/Spatial residual signal，同时保护 TKE/fluctuation structure”。

`T1-ID-FE-INCR-FROZEN-CNO-E0-RIDGE-S20260902` — `DONE` / `REVIEW_REQUIRED`.

- Frozen, traceable strong CNO baseline: registered `T1-ID-LOSS-E0-90M-S20260901` E0 `model_best.pth`, SHA-256 `5d02c8da5bcbcbcc47917b0021b1007b2036931a3a51483772f7607deeb4aff6`; weights frozen throughout.
- 50 train / 16 dev only; complete 20→20 windows, stride 20, 2052 / 659 windows, runtime H×W 32×64. GPU inference plus CPU closed-form ridge took 834.0 s; no final/private-test or Codabench access.
- Frozen-CNO raw dev Rel-L2/TKE/MVPE: `0.168923 / 0.538475 / 0.136146` (matches registered E0 reference). Raw-Control ridge: `0.168162 / 0.594538 / 0.135999`.
- Temporal improves Rel-L2 and MVPE but worsens TKE; Spatial independently improves Rel-L2 and MVPE beyond Temporal, while worsening TKE; Joint improves Rel-L2/MVPE further but has the largest TKE worsening. No all-three-metric stable gain was established.
- Under the protected all-three-metric rule, Temporal, Spatial, and Joint are `LOW_INCREMENTAL_VALUE`; their Rel/MVPE trade-off signals remain review-only and do not authorize fusion training.
- Evidence is local under `../artifacts/fe_incremental_probe_cno_e0_s20260902/`; the GitHub-readable handoff is `docs/coordination/CHATGPT_HANDOFF_FE_INCREMENTAL_CNO_E0.md`.

`T1-ID-FE-INCR-PERSIST-RIDGE-S20260902` — `DONE` / `REVIEW_REQUIRED`.

- Minimal supervised incremental-value probe on the existing registered PERSIST baseline; 50 train / 16 dev only, complete 20→20 windows, 2052 / 659 windows, runtime H×W 32×64.
- CPU closed-form ridge only; no CNO inference, model training, locked-final/private-test access or Codabench. Train-only feature normalization and residual fitting were used.
- Dev window-micro errors (Rel-L2 / TKE / MVPE): Raw-Control `0.131353 / 0.987575 / 0.130701`; Raw+Temporal `0.118340 / 0.940578 / 0.109454`; Raw+Spatial `0.130241 / 0.973685 / 0.129355`; Raw+Temporal+Spatial `0.116346 / 0.936060 / 0.107228`.
- Relative to Raw-Control, Temporal and Temporal+Spatial improve all three metrics consistently; Spatial gives a small but same-direction improvement. Shortlist: Temporal, Spatial, and the joint package `KEEP_FOR_MODEL_PROBE`; no automatic model training is authorized.
- Evidence is local under `../artifacts/fe_incremental_probe_s20260902/`; the committed review record is `docs/coordination/CHATGPT_HANDOFF_FE_INCREMENTAL_PROBE.md`.

`T1-ID-FE-SPATIAL-DATA01-S20260902` — `DONE` / `REVIEW_REQUIRED`.

- Spatial pixel-gradient follow-up plus FE-DATA-01 consistency correction: 50 train / 16 dev only, 2102 / 675 windows, H×W 64×128, no training and no locked-final access.
- Raw evidence remains local under `../artifacts/feature_summary_batch1/` and is intentionally not in Git. The committed review record is `docs/coordination/CHATGPT_HANDOFF_FE_DATA01.md`.
- The outcome is a descriptive Spatial KEEP/WATCH/LOW_VALUE shortlist; all five are retained as data-side candidates, with vorticity explicitly derived. It does not authorize feature fusion, model training, a locked-final audit, or a submission.

## Current Task

The detached job on `gpu` completed at 1500 updates with relative improvements
Rel-L2 `-7.286%`, TKE `+14.240%`, MVPE `+1.504%`; because lower error is
better, TKE improved and passed its gate condition. The only failed condition
was Rel-L2, giving decision `STOP_BALANCED_LOCAL3_EARLY`. Handoff:
`docs/coordination/CHATGPT_HANDOFF_POINT_LOCAL3_BALANCED_L001.md`. No further
Point experiment is authorized by this result.

No execution task is active. FF-00 is complete and awaits ChatGPT/Sol review of the
accepted provenance exception and proposed Fusion protocol. Do not start FF-01,
FF-02 or FF-03.

## Execution State

`REVIEW_REQUIRED`

Feature Engineering stage: `CLOSED`.
Temporal / Spatial Prior Fusion stage: `FF-00 completed / REVIEW_REQUIRED`.

Ask ChatGPT/Sol to read `docs/coordination/CHATGPT_HANDOFF_FEATURE_FUSION_FF00.md`,
review `BASELINE_PROVENANCE_EXCEPTION_ACCEPTED`, and return one bounded next action.

## Constraints

- Default Track 1 work uses the frozen 50 train / 16 dev / 16 locked-final manifest and selects only on dev.
- Do not access locked final, private test, or Codabench unless explicitly authorized by the task and the registry protocol.
- Do not start full-scale analysis or long CPU/GPU work without explicit scope and resource authorization; smoke-test first.
- Keep all data, checkpoints, archives, credentials, and absolute private paths out of Git.

## Review Handoff

For the next material result, provide: experiment ID and state; commit SHA or dirty-tree note; manifest/split SHA; exact command and key configuration; artifact identifier or repository-relative path; core metrics; and `GO` / `STOP` / `REVIEW_REQUIRED`.

## Repository Note

The registry and several tools are presently uncommitted local work. Until a commit is created, reports must identify the relevant dirty-tree state rather than implying a reproducible commit SHA.

FF-00 handoff status: completed with `BASELINE_PROVENANCE_EXCEPTION_ACCEPTED`; the
historical source commit remains `UNKNOWN / NOT RECOVERED`, and the conclusion is
`REVIEW_REQUIRED`. There is no active execution task.
