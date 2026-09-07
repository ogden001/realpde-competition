# NEXT_ACTION

## Goal
Repair the RW-01 shuffle confound and revalidate train-window Random Phase against the existing RW-00 fixed reference.

## Tasks
1. Globally and deterministically shuffle each epoch's selected Random Phase indices; preserve independent trajectory phases.
2. Verify sampler invariants, workers 0/2 equivalence, fixed-phase-0 set equivalence, compile, and smoke.
3. Push the frozen execution commit, then run RW-CTRL (Random Phase path, phase 0, global shuffle) for 1000 updates.
4. Only if RW-CTRL passes its registered gate, run RW-01 at 1000/2000/3000/5000/7500 with the registered @3000 early gate.

## Constraints
- Frozen 50 Train / 16 Dev; P0-A CNO, N2, `sim_pretrain/sim_cno.pth`, seed `20260901`, AdamW `1e-5`, batch 8, workers 2.
- Dev remains `start=0, stride=20`; reuse RW-00 without retraining it.
- Only variable is train-window phase randomization. No locked-final, Codabench, full-data, Random Start, stride/phase sweep, or next experiment.
- Final research status is `REVIEW_REQUIRED`; ChatGPT/Sol owns the scientific conclusion.

## Deliverables
- Execution/results commits, metadata, audit, review logs and artifact manifests.
- RW-CTRL gate evidence; RW-01 curve, trajectory comparison and report.
- Historical RW-00/RW-01 result marked `INVALID_COMPARISON / SHUFFLE_CONFOUND`.

## Stop
After results and review evidence are committed and pushed; do not design another experiment.
