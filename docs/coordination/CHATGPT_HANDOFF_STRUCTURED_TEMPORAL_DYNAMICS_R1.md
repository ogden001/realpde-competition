# Structured Temporal Dynamics R1 Handoff

Status: `REVIEW_REQUIRED`

## Execution status

All four pre-registered arms completed on the frozen 50 Train / 16 Dev split.
The locked-final split, full-data training, SPS, and Codabench were not
accessed. This handoff records execution facts only; it makes no research
ranking or continuation decision.

## Protocol

- Execution commit: `95965b3b0c8e3f124991de109ffa82bbdf2430dc`.
- Manifest SHA-256: `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`.
- Official v9 scorer SHA-256: `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`.
- Shared Direct@1500 checkpoint SHA-256: `5499e60a3b8146bf095070dc76d03c85eae57b5f1ec794444276bab362458ec4`.
- P0-A, N2, stride 20, seed `20260901`, AdamW `1e-5`, effective batch `8`, 1500 continuation updates, and absolute evaluations `2000/2500/3000`.
- Shared-GPU memory adaptation: micro-batch `2`, accumulation `4`, evaluation batch `2`, and a CUDA allocator cap of `12 GiB`. This was the task-authorized OOM recovery; all four arms used the same settings.
- C0/T1/T2 initial Direct prediction parity was exact; pressure maximum absolute value was `0`. T3 initial parity was exact and its mixer output gradient maximum absolute value was `0.1028610542`.
- T1 fixed `lambda_delta=34.9553070068`; T2 fixed `lambda_vort=21.3653831482`.

## Dev raw errors

Lower is better. These are official-scorer raw errors, not a leaderboard composite or a Sol conclusion.

| Arm | Update | Rel-L2 | TKE | MVPE | Inference s/window |
|---|---:|---:|---:|---:|---:|
| C0 | 2000 | 0.185984 | 0.629704 | 0.155478 | 0.055358 |
| C0 | 2500 | 0.181335 | 0.591543 | 0.149392 | 0.055355 |
| C0 | 3000 | 0.197545 | 0.633871 | 0.170807 | 0.055445 |
| T1 | 2000 | 0.181015 | 0.640729 | 0.155011 | 0.055402 |
| T1 | 2500 | 0.177066 | 0.615855 | 0.149378 | 0.055515 |
| T1 | 3000 | 0.186753 | 0.654265 | 0.170106 | 0.055487 |
| T2 | 2000 | 0.182312 | 0.619321 | 0.153477 | 0.055475 |
| T2 | 2500 | 0.176072 | 0.590834 | 0.148284 | 0.055513 |
| T2 | 3000 | 0.199202 | 0.630783 | 0.170158 | 0.055452 |
| T3 | 2000 | 0.187138 | 0.624129 | 0.156135 | 0.055815 |
| T3 | 2500 | 0.182166 | 0.590129 | 0.151651 | 0.055747 |
| T3 | 3000 | 0.198035 | 0.634338 | 0.166572 | 0.055882 |

## Runtime and artifacts

- Training wall time (C0/T1/T2/T3): `2696.05 / 2665.77 / 2703.15 / 2678.68 s`.
- Peak CUDA allocated/reserved memory in bytes: C0 `5033026048/5272240128`; T1 `5032370176/5272240128`; T2 `5032370176/5272240128`; T3 `2604369920/2854223872`. All are below the `12884901888` byte cap.
- T3 added parameter count: `210`; C0/T1/T2 added parameter count: `0`.
- Remote evidence root: `/home/chyfuture/realpde_runs/structured_temporal_dynamics_round1_20260907_lowmem/`.
- Each arm contains `aggregate_metrics.csv`, `training_curve.csv`, `runtime.json`, `run_metadata.json`, checkpoints, and per-evaluation `horizon_metrics.csv`, `trajectory_metrics.csv`, `trajectory_horizon_metrics.csv`, `window_metrics.csv`, and `temporal_metrics.csv`.
- Final `eval_03000/predictions.npz` for every arm contains float32 Dev `u/v` predictions, targets, trajectory identifiers, and window starts.

## Tests and execution exception

- Local test command: `python -m pytest tests/test_structured_temporal_dynamics.py -q` → `5 passed`.
- The first batch-8 shared-GPU execution was stopped by CUDA OOM while another Python process held about 10.8 GiB. Its partial artifacts were retained separately and not used as evidence. The low-memory rerun above completed normally.
