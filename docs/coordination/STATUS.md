# Track 1 Coordination Status

Updated: 2026-09-02

## Current Stage

Competition execution: P0-A + N2 all-released-data continuation toward a
time-bounded submission candidate.

## Latest Completed

`T1-ID-FE-SPATIAL-DATA01-S20260902` — `DONE` / `REVIEW_REQUIRED`.

- Spatial pixel-gradient follow-up plus FE-DATA-01 consistency correction: 50 train / 16 dev only, 2102 / 675 windows, H×W 64×128, no training and no locked-final access.
- Raw evidence remains local under `../artifacts/feature_summary_batch1/` and is intentionally not in Git. The committed review record is `docs/coordination/CHATGPT_HANDOFF_FE_DATA01.md`.
- The outcome is a descriptive Spatial KEEP/WATCH/LOW_VALUE shortlist; all five are retained as data-side candidates, with vorticity explicitly derived. It does not authorize feature fusion, model training, a locked-final audit, or a submission.

## Current Task

`T1-COMP-P0A-N2-FULL-S20260902` uses the released 82 real PIV trajectories,
official `sim_real_ft` CNO initialization, 20-channel causal P0-A input and
the frozen N2 loss. The original `last@6800` checkpoint remains preserved; a
continuation retains a distinct `model_update_10300.pth` fallback and proceeds
toward `last@15300` under a hard local deadline. No private test,
target-driven checkpoint selection or Codabench upload belongs to this runner.

## Execution State

`RUNNING` — time-bounded full-data continuation; see
`docs/coordination/TRAINING_HANDOFF_2026-09-02.md` for results and lineage.

## Constraints

- This authorized competition run fits all 82 released trajectories and does not make a dev/final selection.
- The normal protocol uses micro-batch 4 / accumulation 2 (effective batch 8).
  A later user-authorized high-memory probe established that larger batches
  were slower for this workload, so the deadline run uses the faster normal
  protocol. The active run has an explicit 23:39 Asia/Shanghai stop.
- Do not access locked final, private test, or Codabench unless explicitly authorized by the task and the registry protocol.
- Do not start full-scale analysis or long CPU/GPU work without explicit scope and resource authorization; smoke-test first.
- Keep all data, checkpoints, archives, credentials, and absolute private paths out of Git.

## Review Handoff

For the next material result, provide: experiment ID and state; commit SHA or dirty-tree note; manifest/split SHA; exact command and key configuration; artifact identifier or repository-relative path; core metrics; and `GO` / `STOP` / `REVIEW_REQUIRED`.

## Launch Evidence

- Smoke: 2 updates, 2 released trajectories, same official container and
  checkpoint; peak allocated CUDA memory `5,065,201,152` bytes (4.72 GiB),
  below the `12 GiB` gate.
- The detached runner writes `status.json`, `run_metadata.json`,
  `model_resume.pth`, `model_last.pth`, and `train.log` under its run artifact
  directory. It is intentionally not polled after the one-time launch check.

## Repository Note

The registry and several tools are presently uncommitted local work. Until a commit is created, reports must identify the relevant dirty-tree state rather than implying a reproducible commit SHA.
