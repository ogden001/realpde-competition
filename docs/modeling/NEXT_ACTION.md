# NEXT_ACTION

## Goal
Execute `TMR-01`: transfer the already-implemented teammate-style residual corrector onto the frozen SOTA-V2 50/16 backbones at Stage-A `@30000` and Stage-B `@32500`, then collect one final physical-metric comparison.

This task is execution/reproduction, not a new research-design task. ChatGPT/Sol has already implemented the runner and tests.

## Source
- Runner: `tools/realpde_teammate_residual_transfer.py`
- Tests: `tests/test_teammate_residual_transfer.py`
- Corrector implementation reused without redesign from `tools/realpde_adaptive_probe.py`:
  - 42-channel causal features
  - `ResidualCorrector3D(42, hidden=64, blocks=2, max_delta=0.04)`
  - fixed historical composite corrector loss
  - AdamW `lr=1e-4`, `weight_decay=1e-5`, cosine schedule
  - `2400` updates, batch `8`

## Frozen experiment
Train two correctors, sequentially in the same run:

1. frozen SOTA-V2 `@30000` (Stage A end) + teammate residual corrector;
2. frozen SOTA-V2 `@32500` (Stage B best) + the same teammate residual corrector recipe.

Use the fixed 50 Train Dense-All windows for corrector training. Do **not** access Dev between the two trainings. After both fixed 2400-update trainings finish, evaluate the fixed six points:

- `@30000`, alpha `0.0 / 0.5 / 1.0`
- `@32500`, alpha `0.0 / 0.5 / 1.0`

`alpha=0` is exact backbone parity, `alpha=0.5` is the fixed historical protection diagnostic, and `alpha=1` is the full teammate correction. There is no alpha sweep or Dev-based selection.

## Metrics
Physical prediction only:
- Rel-L2
- TKE
- MVPE
- 16-trajectory paired evidence
- h1-h20 horizon evidence, especially h19/h20
- fluctuation-energy diagnostics

Do not calculate, optimize, report, or use SPS for decision-making in this task.

## Constraints
- Work only on `main`.
- Fixed manifest SHA256: `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`.
- 50 Train / 16 Dev; Train canonical/Dense-All/Dev windows `2052/40488/659`.
- Freeze both SOTA-V2 backbones; no backbone optimizer step.
- No checkpoint search, no second seed, no hyperparameter sweep, no intermediate Dev gate.
- Run both corrector experiments even if the first looks poor; collect results only after both finish.
- No SPS/uncertainty work, full-data refit, locked-final/private data, package build, or Codabench.
- Mechanical compatibility fixes are allowed only when needed to execute the frozen runner; do not change research logic without Sol review.

## Execution protocol
1. Sync clean `main` and read `AGENTS.md`, `docs/CHATGPT_CODEX_WORK_PROTOCOL.md`, and `docs/CODEX_LONG_TASK_EXECUTION_RULES.md`.
2. Run the focused tests and compile/diff checks.
3. Verify the existing `@30000` and `@32500` checkpoints from the SOTA-V2 50/16 run.
4. Launch `tools/realpde_teammate_residual_transfer.py` once in the CUDA container. The runner itself trains both arms before any Dev evaluation.
5. After the run reaches `DONE`, collect all final evidence in one pass.
6. Commit only lightweight evidence under `docs/modeling/reviews/tmr01_teammate_residual_transfer_20260917/`; do not commit checkpoints or large raw logs.
7. Mark final status `REVIEW_REQUIRED` and stop. ChatGPT/Sol performs the scientific interpretation.

## Required final report
Return:
- exact execution commit;
- focused test / compile / diff-check results;
- remote run root and CUDA environment;
- hashes of the two frozen backbones and two trained correctors;
- one six-row table containing `backbone update × alpha × Rel-L2 × TKE × MVPE` plus relative changes versus alpha=0;
- short trajectory/horizon/fluctuation diagnostics;
- confirmation that SPS, full-data, locked-final, package and Codabench were not accessed;
- evidence commit SHA.

Do not emit an automatic GO/NO-GO and do not start a next experiment.
