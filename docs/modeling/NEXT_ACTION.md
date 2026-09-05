# NEXT_ACTION

## Goal
Run the frozen A2 Multi-scale / coarse+fine breadth-first screen on the fixed 50 Train / 16 Dev protocol.

## Tasks
1. Use `tools/realpde_arch_a2_multiscale.py` and `tools/run_arch_a2_multiscale_remote.sh`.
2. Supply the exact frozen manifest, Direct@1500 start checkpoint, Direct@3000 reference checkpoint, and official kit paths.
3. Run the A2 unit test and Python compile check, then launch the provided script detached.
4. Return execution commit, PID, launcher log, output root, and preflight path.

## Constraints
- Only 50 Train / 16 Dev.
- Exactly Direct@1500 + 1500 updates, seed `20260901`, AdamW `1e-5`, batch `8`, N2.
- The only experimental variable is the frozen 2x-coarse Past20-u/v residual branch.
- No locked-final, Codabench, SPS, full-data, architecture sweep, width/kernel/gate changes, or next experiment.
- If checksum, version, test, compile, or preflight fails, stop and report. Do not patch research logic.

## Deliverables
- `preflight/preflight.json`
- `run/run_metadata.json`
- `run/update_curve.csv`
- `run/trajectory_comparison.csv`
- `run/gate_result.json`
- `run/summary.json`
- final A2 checkpoint in remote run artifacts

## Stop
After detached launch is confirmed running, stop polling and report PID/log/path.
