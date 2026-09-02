# Track 1 Coordination Status

Updated: 2026-09-02

## Current Stage

Clean offline research: candidate/feature screening under the frozen Track 1 ID protocol.

## Latest Completed

`T1-ID-FE-DATA01-B1-S20260902` — `DONE` / `REVIEW_REQUIRED`.

- Pure Batch-1 runtime-feature distribution diagnosis: 50 train / 16 dev only, 2102 / 675 windows, no training and no locked-final access.
- Raw evidence remains local under `../artifacts/feature_summary_batch1/` and is intentionally not in Git. The committed review record is `docs/coordination/CHATGPT_HANDOFF_FE_DATA01.md`.
- The outcome is a descriptive KEEP/WATCH/LOW_VALUE shortlist. It does not authorize feature fusion, model training, a locked-final audit, or a submission.

## Current Task

No execution task is active. Ask ChatGPT/Sol to read `docs/coordination/CHATGPT_HANDOFF_FE_DATA01.md` from GitHub and await a bounded `NEXT_ACTION`.

## Execution State

`AWAITING_NEXT_ACTION`

## Constraints

- Default Track 1 work uses the frozen 50 train / 16 dev / 16 locked-final manifest and selects only on dev.
- Do not access locked final, private test, or Codabench unless explicitly authorized by the task and the registry protocol.
- Do not start full-scale analysis or long CPU/GPU work without explicit scope and resource authorization; smoke-test first.
- Keep all data, checkpoints, archives, credentials, and absolute private paths out of Git.

## Review Handoff

For the next material result, provide: experiment ID and state; commit SHA or dirty-tree note; manifest/split SHA; exact command and key configuration; artifact identifier or repository-relative path; core metrics; and `GO` / `STOP` / `REVIEW_REQUIRED`.

## Repository Note

The registry and several tools are presently uncommitted local work. Until a commit is created, reports must identify the relevant dirty-tree state rather than implying a reproducible commit SHA.
