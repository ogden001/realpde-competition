# SPATIAL_PHASE_BACKBONE_SCREEN_NEXT_ACTION

Status: `READY_FOR_EXECUTION / REVIEW_REQUIRED`

Execute exactly one candidate Stage-1 backbone run under `REALPDE_SPATIAL_PHASE_BACKBONE_SCREEN_V1`.

1. Sync current `main`.
2. Record and preserve unrelated pre-existing untracked files.
3. Run focused tests, `py_compile`, and `git diff --check`.
4. Resolve the same approved Clean Baseline data, `sim_real_cno.pth`, and v9 model/kit root.
5. Run screen preflight.
6. Run exactly one candidate backbone to 8,723 updates.
7. Run review against the frozen historical Clean Baseline @8000 and @8723.
8. Archive lightweight evidence.
9. Commit/push only the evidence to `origin/main`.
10. STOP at `REVIEW_REQUIRED`.

The operational contract is `SPATIAL_PHASE_BACKBONE_SCREEN_CODEX_EXECUTION.md`.
