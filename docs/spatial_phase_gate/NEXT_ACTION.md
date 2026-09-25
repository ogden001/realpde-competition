# NEXT_ACTION - Execute Spatial Phase Gate V1

Status: `READY_FOR_EXECUTION / REVIEW_REQUIRED`

Execute exactly one matched Control-vs-Candidate spatial-phase gate using the already selected Strong Backbone Clean predictor.

## Task

1. Sync clean `main`.
2. Verify Strong Backbone and Strong Residual checkpoint SHA256.
3. Run focused tests and `py_compile`.
4. Run gate preflight.
5. Run exactly two 5k continuation arms:
   - Control: P00 only.
   - Candidate: 50% P00 + 50% balanced P01/P10/P11.
6. Evaluate both on official P00 Seen-Dev12 only.
7. Run the frozen parity checks and GO/NO_GO gate.
8. Archive lightweight evidence to `docs/spatial_phase_gate/results/20260925_runN`.
9. Commit/push evidence to `origin/main`.
10. STOP at `REVIEW_REQUIRED`.

## Hard constraints

- no AoA10 Holdout;
- no locked-final/private;
- no Codabench;
- no full-data refit;
- no submission/package work;
- no phase-ratio/LR/budget/seed/gate sweep;
- no AoA or temporal sampling augmentation composition;
- no automatic second experiment after GO or NO_GO;
- no automatic final long train.

Scientific protocol is frozen in `README.md`; operational steps are frozen in `CODEX_EXECUTION.md`.
