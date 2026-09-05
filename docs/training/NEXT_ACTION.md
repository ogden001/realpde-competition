# NEXT_ACTION

## Goal
Run the frozen Horizon Curriculum breadth-first screen after A2, using the fixed 50 Train / 16 Dev protocol.

## Tasks
1. Use `tools/realpde_horizon_curriculum.py` and `tools/run_horizon_curriculum_remote.sh`.
2. Supply the exact frozen manifest, official `sim_pretrain` checkpoint, Direct@3000 reference checkpoint, and official kit paths.
3. Run the curriculum unit test and Python compile check, then execute only after A2 finishes.
4. Return execution commit, PID/log if separately launched, output root, and preflight path.

## Constraints
- Only 50 Train / 16 Dev.
- P0-A Direct CNO, N2, seed `20260901`, AdamW `1e-5`, batch `8`, exactly 3000 updates.
- Only variable: reconstruction-horizon curriculum. TKE remains full-Future20 at every stage.
- No locked-final, Codabench, SPS, full-data, schedule sweep, weight sweep, or next experiment.
- If checksum, version, test, compile, or preflight fails, stop and report. Do not patch research logic.

## Deliverables
- `preflight/preflight.json`
- `run/run_metadata.json`
- `run/update_curve.csv`
- `run/trajectory_comparison.csv`
- `run/horizon_rel_l2.csv`
- `run/gate_result.json`
- `run/summary.json`

## Stop
After the queued detached job is confirmed running, stop polling and report paths.
