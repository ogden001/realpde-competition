# Track 1 Coordination Status

Updated: 2026-09-02

## Current Stage

Clean offline research: candidate/feature screening under the frozen Track 1 ID protocol.

## Latest Completed

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

No execution task is active. Ask ChatGPT/Sol to read `docs/coordination/CHATGPT_HANDOFF_FE_INCREMENTAL_PROBE.md` from GitHub and review the bounded incremental-value probe.

## Execution State

`REVIEW_REQUIRED`

## Constraints

- Default Track 1 work uses the frozen 50 train / 16 dev / 16 locked-final manifest and selects only on dev.
- Do not access locked final, private test, or Codabench unless explicitly authorized by the task and the registry protocol.
- Do not start full-scale analysis or long CPU/GPU work without explicit scope and resource authorization; smoke-test first.
- Keep all data, checkpoints, archives, credentials, and absolute private paths out of Git.

## Review Handoff

For the next material result, provide: experiment ID and state; commit SHA or dirty-tree note; manifest/split SHA; exact command and key configuration; artifact identifier or repository-relative path; core metrics; and `GO` / `STOP` / `REVIEW_REQUIRED`.

## Repository Note

The registry and several tools are presently uncommitted local work. Until a commit is created, reports must identify the relevant dirty-tree state rather than implying a reproducible commit SHA.
