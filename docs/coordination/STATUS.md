# Track 1 Coordination Status

Updated: 2026-09-04

## Current Stage

`REVIEW_REQUIRED`

No execution task is active.

Current priority is no longer to extend training blindly. The completed P0-A + N2 50/16 validation run now reaches 30,900 updates and shows continued late-stage improvement followed by a broad plateau / oscillation regime. Existing late checkpoints should be reviewed before any new long GPU run.

## Latest Completed

### P0-A + N2 validation continuation to 30,900

Reference: `T1-ID-P0A-N2-VALIDATION-30900-S20260903`

- Fixed 50-train / 16-dev protocol, official Track 1 v9 scorer raw errors.
- `OFFICIAL_WARM_START` evidence; not a clean causal baseline.
- 10,300 → 30,900 continuation completed in the prior Codex session.
- No training was rerun while restoring this record.
- No locked-final/private-test access and no new Codabench submission occurred during documentation recovery.
- Best Rel-L2: `0.112398 @ 27,880`.
- Best TKE: `0.492848 @ 30,340`.
- Best MVPE: `0.084671 @ 26,240`.
- `26,240` is the current `BALANCED_DEV_CHECKPOINT_CANDIDATE`: `0.112925 / 0.494840 / 0.084671`.
- Around 22k updates the run enters a broad plateau / oscillation regime rather than a clear long-horizon overfitting collapse.
- Handoff: `docs/coordination/CHATGPT_HANDOFF_P0A_N2_VALIDATION_30900.md`.

### Latest official submission evidence

P0-A + N2 full-data 15,300-update Codabench result:

- Rel-L2 `93.023539`
- TKE `78.355520`
- MVPE `91.894417`
- Time `88.430528`
- SPS `11.431650`
- Final `71.153839`

The previous best simple CNO submission remains final `75.584550`. The strongest visible regression in the new submission is SPS, while TKE improves materially. Handoff: `docs/coordination/CHATGPT_HANDOFF_T1_P0A_N2_FULL15300_CODABENCH.md`.

## Other Open Reviews

### H1 / Point hybrid

- Pure Point and LOCAL3 routes remain stopped.
- H1 scale retains strong Rel-L2/MVPE signal but does not provide stable trajectory-level TKE protection.
- At fixed `alpha=0.5`, Rel-L2 and MVPE improve on 16/16 trajectories; TKE improves on only 2/16, and only 3/16 remain within the `>= -5%` TKE protection threshold.
- Do not automatically start H2, LOCAL5 or joint training.
- Handoff: `docs/coordination/CHATGPT_HANDOFF_HYBRID_CNO_POINT_H1_SCALE_STABILITY.md`.

### Feature Engineering / Fusion

- Feature Discovery remains `CLOSED`.
- Temporal/Spatial information has residual signal, but historical correction/fusion attempts do not establish stable all-three-metric gains on a strong CNO.
- Future feature work should focus on TKE-preserving fusion, not expanding the feature catalog.
- Do not automatically start FF-01/FF-02/FF-03.
- Handoff: `docs/coordination/CHATGPT_HANDOFF_FE_FINAL_REVIEW.md`.

## Current Decision Boundary

Before any new long training run, review the existing late P0-A + N2 checkpoints, especially:

- `26,240`: balanced dev candidate
- `27,880`: best Rel-L2
- `30,340`: best TKE

Do not promote any of them to “best submission checkpoint” from dev metrics alone. Codabench does not publish the final-score combination, and SPS must be protected independently.

## Constraints

- Default Track 1 research uses the frozen 50 train / 16 dev / 16 locked-final manifest and selects only on dev.
- Do not access locked final, private test or Codabench unless explicitly authorized.
- Do not start long CPU/GPU work without an explicit bounded task.
- Keep data, checkpoints, archives, credentials and private absolute paths out of Git.

## Key Navigation

- Overall status: `docs/realpde整体优化概要.md`
- Training summary: `docs/training/training概要.md`
- 30.9k validation handoff: `docs/coordination/CHATGPT_HANDOFF_P0A_N2_VALIDATION_30900.md`
- Submission history: `docs/submission_log.md`
- Experiment registry: `docs/track1_experiment_registry.md`
