# SOTA-V2 Full-Data Refit

Status: `COMPLETED` / `REVIEW_REQUIRED`

This is the frozen all-82 released-PIV refit. It did not access Dev/final holdouts, private test, Codabench, SPS, or Adaptive Uncertainty. The primary checkpoint is the pre-frozen Dense-epoch mapping at update `53582`; no checkpoint was selected using full-data training loss.

## Provenance and execution

- Execution commit: `b025e44e2202ef40a1cba1a05907650431c4eb3f`
- Warm-start: `/home/chyfuture/RealPDE_data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth`
- Warm-start SHA-256: `82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61`
- Official v9 scorer SHA-256: `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`
- Remote run root: `/home/chyfuture/realpde_runs/sota_v2_full_20260916/run`
- Final checkpoint: `/home/chyfuture/realpde_runs/sota_v2_full_20260916/run/checkpoints/model_update_53582.pth`
- Final checkpoint SHA-256: `f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce8`
- Host/GPU: `gpu` / NVIDIA GeForce RTX 3090 24 GB
- Batch: micro-batch `4`, accumulation `2`, effective batch `8`, workers `2`

Full-data scope was `82` trajectories with `3383` canonical windows and `66755` Dense-All windows. CUDA preflight passed with 20 P0-A features, Direct→MF UV maximum difference `1.1920928955078125e-07`, pressure maximum absolute value `0`, and finite/non-zero loss gradient. The initial micro-batch-8 preflight OOM was handled by the explicitly allowed 4×2 batch fallback; no scientific recipe changed.

## Frozen schedule

- Stage A: updates `1..49461`, LR `1e-5`
- Stage B: updates `49462..53582`, LR `3e-6`, extra Rel-L2 `0.027514`
- Milestones: `12365`, `24730`, `32974`, `41217`, `49461`, `51109`, `53582`
- Final update: `53582` (not `57704`)

## Training milestone evidence

| Update | Reference update | Stage | MSE | TKE | Rel | MVPE | Vorticity |
|---:|---:|:---:|---:|---:|---:|---:|---:|
| 12,365 | 7,500 | A | 0.00008326 | 0.504142 | 0.107128 | 0.101044 | 0.00003902 |
| 24,730 | 15,000 | A | 0.00011504 | 0.437768 | 0.106780 | 0.093875 | 0.00004837 |
| 32,974 | 20,000 | A | 0.00022873 | 0.450651 | 0.105247 | 0.077170 | 0.00008708 |
| 41,217 | 25,000 | A | 0.00017676 | 0.443684 | 0.088750 | 0.060410 | 0.00007724 |
| 49,461 | 30,000 | A | 0.00016611 | 0.395292 | 0.093675 | 0.083211 | 0.00006960 |
| 51,109 | 31,000 | B | 0.00017882 | 0.387312 | 0.097447 | 0.078928 | 0.00007072 |
| 53,582 | 32,500 | B | 0.00007250 | 0.456061 | 0.086659 | 0.063143 | 0.00003557 |

Training runtime was `35208.44 s` (about 9h47m). Peak GPU memory was `9.17 GiB` allocated / `9.76 GiB` reserved. The runner reports `6.4216203` Dense epochs.

Tests: full runner, integrated protocol, and real-fixture tests passed; the three local Dense-All failures remain the known Mac PyTorch/NumPy ABI incompatibility (`Numpy is not available`).

Frozen recipe deviation: `NO` (batch fallback was an allowed CUDA memory adaptation).

Only lightweight evidence is committed here. Checkpoints, dataset, and raw stdout remain on the remote host.
