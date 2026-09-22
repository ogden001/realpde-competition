# NEXT_ACTION — Execute colleague80 V2 three-arm campaign

Status: `READY_FOR_EXECUTION / REVIEW_REQUIRED_AFTER_RUN`

## Goal

Run the frozen three independent arms from
`docs/colleague_v2_campaign/README.md` and return complete Git evidence.

## Tasks

1. Sync `origin/main`; verify clean worktree and push permission.
2. Verify `REQUIRED_COMMIT` is an ancestor of HEAD.
3. Run:
   `pytest -q tests/test_colleague_v2_campaign.py tests/test_colleague_incremental_screen.py tests/test_colleague_next_two_experiments.py tests/test_post_train_diagnostics.py`.
4. Execute `tools/colleague_80pt/run_v2_campaign.py` with the real released
   train data, colleague base/residual checkpoints, the existing SOTA-V2
   P0-A/MF checkpoint, and the colleague Dev16 split.
5. Run the standard post-train diagnostics produced by each arm; do not omit
   required evidence.
6. Archive lightweight evidence under
   `docs/colleague_v2_campaign/results/20260922_<run-id>/`.
7. Commit, pull --rebase, push to `origin/main`, then stop.

## Constraints

- Do not change model/loss/data semantics, budgets, gates, split, seed or
  checkpoint identities.
- Environment-only fixes are allowed.
- No locked-final/private data.
- No Codabench.
- No automatic Combo/full merge/package.
- Do not delete or overwrite prior runs.

## Deliverables

- execution commit and result commit;
- campaign manifest and exact commands;
- training review logs for A/B/C;
- overall, horizon, trajectory, trajectory×horizon, mean/fluctuation/energy,
  spatial and mechanism-specific diagnostics;
- `summary.json`;
- checkpoint SHAs and remote paths.

## Stop

Return `REVIEW_REQUIRED`. Do not start another experiment.
