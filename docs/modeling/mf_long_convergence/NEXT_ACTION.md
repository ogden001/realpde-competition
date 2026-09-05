# Goal

Run an unattended matched long-convergence test of Direct vs Mean/Fluctuation (MF) from absolute update 3000 to 15000.

# Tasks

1. Use the frozen 50 Train / 16 Dev manifest.
2. Start Direct from exact matched Direct@3000.
3. Start MF from Campaign02 C0 / MF@3000.
4. Restore each arm's optimizer state and continue each by 12000 updates.
5. Evaluate both at absolute 6000 / 9000 / 12000 / 15000 and write the paired curve plus final 16-trajectory wins.
6. Run `tools/run_idle_gpu_second_wave_20260906.sh` in a separate clean workspace; it waits for the first overnight queue PID before using the GPU.

# Constraints

- P0-A, N2, seed `20260901`, AdamW `1e-5`, batch `8`, workers `2` are frozen.
- No model/loss/feature changes or hyperparameter sweep.
- No locked-final, Codabench, SPS or full-data training.
- Do not modify research code during execution. If checksum, metadata, test or preflight fails, stop and report.

# Deliverables

- `preflight/preflight.json`
- `formal/paired_convergence.csv`
- `formal/final_trajectory_comparison.csv`
- `formal/gate_result.json`
- Direct and MF checkpoints at 6000 / 9000 / 12000 / 15000
- execution commit, PID, log path, output path

# Stop

After MF@15000 evaluation and report generation. Final research decision remains `REVIEW_REQUIRED` for ChatGPT/Sol.
