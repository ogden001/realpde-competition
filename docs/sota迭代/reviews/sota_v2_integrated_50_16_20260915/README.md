# SOTA-V2 Integrated 50/16

Status: `COMPLETED` — frozen competition-oriented 50/16 Dev trial. No locked-final, full-data, Codabench, or SPS optimization was accessed.

## Provenance

- Execution commit: `89fe59dc23d31617d6212039a02d3706e11de40f`
- Manifest SHA-256: `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`
- Official scorer SHA-256: `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`
- Warm-start: `/home/chyfuture/RealPDE_data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth`
- Warm-start SHA-256: `82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61`; checkpoint iteration `5000`; Direct CNO input/output channels `3/3`.
- GPU: NVIDIA GeForce RTX 3090. Micro-batch `8`, accumulation `1`, effective batch `8`, workers `2`.
- Remote run root: `/home/chyfuture/realpde_runs/sota_v2_integrated_50_16_20260916/run`

CUDA preflight passed: 50 train / 16 Dev trajectories, canonical/Dense-All/Dev windows `2052/40488/659`, P0-A feature count `20`, finite prediction/loss/gradient, Direct-to-MF UV maximum difference `1.1920929e-07`, and pressure maximum absolute value `0`.

## Aggregate Dev curve

| Update | Stage | Rel-L2 | TKE | MVPE |
|---:|:---:|---:|---:|---:|
| 7,500 | A | 0.125148 | 0.510103 | 0.108036 |
| 15,000 | A | 0.111479 | 0.484161 | 0.087232 |
| 20,000 | A | 0.110924 | 0.485559 | 0.091191 |
| 25,000 | A | 0.106151 | 0.471133 | 0.082209 |
| 30,000 | A | 0.105989 | 0.474373 | 0.090337 |
| 31,000 | B | 0.100937 | 0.470932 | 0.078285 |
| 32,500 | B | 0.099935 | 0.469293 | 0.075778 |
| 35,000 | B | 0.099978 | 0.471441 | 0.075749 |

Historical balanced anchor: Rel-L2 `0.112925`, TKE `0.494840`, MVPE `0.084671`.

Historical Dense-All strong reference: Rel-L2 `0.110092`, TKE `0.479399`, MVPE `0.080831`.

At 30k, h19/h20/h19+h20 squared-error fractions were `0.086717/0.164335/0.251052`; at 35k they were `0.083978/0.158266/0.242244`. The largest trajectory-level Rel-L2 values at both checkpoints were `25425_15.h5` and `3750_20.h5`; detailed trajectory and horizon evidence is retained under the corresponding evaluation directories.

Training wall time was `23117.20` seconds (about 6h25m). Peak allocated/reserved GPU memory was `9.29/14.19 GiB`.

Frozen recipe deviation: `NO`.

Only lightweight evidence is committed here. Remote checkpoint files, `predictions.npz`, and stdout logs remain in the remote run root.
