# Vorticity Low-Memory Paired Final Handoff

## 1. Execution / provenance

- `old_task_stopped=true`; `old_v1_auto_launch_remaining=false`; no old-task PID required termination. Shared-GPU unrelated processes were not stopped.
- Historical `/home/chyfuture/realpde_runs/vorticity_long_final_20260914/` was preserved. Its C0-LONG@15000 is historical convergence evidence only and was not used by this gate.
- New matched arms: C0-LOWMEM and V1-LOWMEM, both restored from R2@3000 model + optimizer checkpoints and trained to absolute update 12000.
- Frozen protocol: 50 Train / 16 Dev, manifest SHA-256 `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`, P0-A, N2, stride 20, seed `20260901`, AdamW `lr=1e-5`, pressure=0, official v9 scorer.
- Shared-GPU low-memory protocol: micro-batch `4`, accumulation `2`, effective batch `8`, CUDA per-process memory cap `12 GiB`, workers `2`. Both arms used the same configuration.
- C0 loss: N2 only. V1 loss: N2 + exact `15.5385751724 * vorticity_MSE`.
- Starting checkpoint SHA-256: C0 `369d6fc281b0e4ed371e32c1f43aa6c8b610557ec549cdbc9d6ecc7777134d4b`; V1 `e40c43b06b5457a6e2a82d2d2845a53ba4d45a607de28ea0075307fb1e150edb`.
- No locked-final, full-data, SPS, or Codabench access.

## 2. Matched official-v9 raw errors

Lower is better. Improvement is `(C0 - V1) / C0 * 100`.

| Update | C0 Rel-L2 | V1 Rel-L2 | Rel improvement % | C0 TKE | V1 TKE | TKE improvement % | C0 MVPE | V1 MVPE | MVPE improvement % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3000 | 0.161343 | 0.156023 | 3.297 | 0.557973 | 0.555572 | 0.430 | 0.131496 | 0.121585 | 7.537 |
| 4500 | 0.163976 | 0.156759 | 4.402 | 0.556512 | 0.557927 | -0.254 | 0.134355 | 0.122084 | 9.133 |
| 6000 | 0.146370 | 0.141623 | 3.243 | 0.544280 | 0.547351 | -0.564 | 0.115839 | 0.115895 | -0.048 |
| 9000 | 0.139834 | 0.136028 | 2.722 | 0.520521 | 0.529146 | -1.657 | 0.107434 | 0.099985 | 6.933 |
| 12000 | 0.130150 | 0.125012 | 3.948 | 0.518528 | 0.510238 | 1.599 | 0.110468 | 0.108193 | 2.059 |

## 3. Trajectory stability

| Update | Rel wins /16 | TKE wins /16 | MVPE wins /16 | All-three | Rel+MVPE with TKE degradation ≤2% |
|---:|---:|---:|---:|---:|---:|
| 3000 | 16 | 1 | 16 | 1 | 7 |
| 4500 | 16 | 5 | 15 | 5 | 5 |
| 6000 | 16 | 5 | 5 | 1 | 3 |
| 9000 | 14 | 15 | 12 | 11 | 11 |
| 12000 | 16 | 14 | 9 | 7 | 7 |

Complete per-trajectory files are retained under each arm's `eval_*/trajectory_metrics.csv`.

## 4. Late-checkpoint median and gate

Late checkpoints are exactly `6000 / 9000 / 12000`.

- Median Rel-L2 improvement: `3.242654%` — PASS (`>=2%`).
- Median MVPE improvement: `2.059027%` — FAIL (`>=4%`).
- Median TKE improvement: `-0.564134%` — PASS (`>=-1%`).
- At least 2 of 3 late checkpoints with Rel>0, MVPE>0, TKE>=-1%: FAIL; only `12000` passes.

FINAL_GATE = PARK

## 5. Final replay

Independent checkpoint-load → Dev inference → official v9 scorer replay was run for both `C0-LOWMEM@12000` and `V1-LOWMEM@12000`.

- C0 maximum absolute raw-error delta: `0.0`.
- V1 maximum absolute raw-error delta: `0.0`.
- Replay consistency: PASS.

## 6. Runtime / memory

| Arm | Train wall time | Final inference s/window | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|
| C0-LOWMEM | 6372.21 s | 0.025340 | 9.23 | 9.80 |
| V1-LOWMEM | 7075.93 s | 0.024706 | 9.23 | 9.80 |

GPU: NVIDIA GeForce RTX 3090. Both arms stayed below the 12 GiB reserved-memory cap.

## 7. Artifact paths

- Remote root: `/home/chyfuture/realpde_runs/vorticity_lowmem_final_20260915/`
- Preflight: `preflight/old_task_stop.json`, `preflight/C0-4x2-v2/`, `preflight/V1-4x2/`
- Arms: `C0-LOWMEM/`, `V1-LOWMEM/`, each with `eval_03000/`, `eval_04500/`, `eval_06000/`, `eval_09000/`, `eval_12000/` and checkpoints.
- Summary: `matched_metrics.csv`, `trajectory_stability.csv`, `late_checkpoint_gate.json`, `training_curve_c0.csv`, `training_curve_v1.csv`, `runtime.json`, `run_metadata.json`.
- Replay: `replay/replay_check.json`, `replay/replay_metrics.csv`.

## 8. Tests / commits

- Focused tests: `7 passed` for low-memory protocol, old-task filtering, temporal checks, replay directory creation, and gate calculation.
- New/modified scripts compiled locally and in the official container.
- `git diff --check`: passed.
- Implementation commit: `672822b`.
- Final closeout commit is the commit containing this handoff and the registry/modeling updates.

Structured Temporal Dynamics exploration = CLOSED.
