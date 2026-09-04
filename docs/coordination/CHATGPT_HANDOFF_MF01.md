# RealPDE Track 1 — MF-01 Handoff

## Result

`MF01_NO_GO` / `REVIEW_REQUIRED`.

MF-01 completed the preregistered 1500 optimizer updates. It improved Rel-L2 and
MVPE but slightly worsened TKE, so it does not meet `MF01_GO_STRONG` or
`MF01_GO_SUPPORTIVE`; no MF-02 was started.

## Provenance and fixed protocol

- Control ID: `T1-ID-MF01-CONTROL-S20260904`
- MF ID: `T1-ID-MF01-S20260904`
- Code commit at launch: `53e1635f718786d6a0eb5d9f5f7b3c6117e8475`
- Manifest SHA-256: `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`
- Official v9 scorer SHA-256: `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`
- Shared initialization: official `sim_pretrain/sim_cno.pth`; SHA-256
  `af85374bfd06c0e386ec803d777396c21484978392213025697c5a7470106b6b`
- Split: 50 train / 16 dev; locked-final not accessed
- P0-A, N2, AdamW `lr=1e-5`, batch 8, workers 2, seed `20260901`
- Updates: 1500; checkpoints: 500 / 1000 / 1500
- Loss: `MSE + 0.05*TKE + 0.027514*Rel-L2 + 0.009757*MVPE`

## Exact model difference

Control used the existing direct CNO `3 -> 3` output. MF-01 used the same CNO
backbone and a five-channel output projection interpreted as mean `u/v`, raw
fluctuation `u/v`, and pressure. The velocity output is
`mean_t(mean_raw) + (fluct_raw - mean_t(fluct_raw))`; pressure is copied from
the pressure projection. No feature, loss, LR, optimizer, batch, seed, or
backbone change was made.

Initialization equivalence smoke passed before training:
`max_abs=2.3841858e-7`, pressure `max_abs=0`, fluctuation temporal-mean
`max_abs=3.1590463e-7`.

## Matched dev results

| Updates | Control Rel-L2 | MF Rel-L2 | ΔRel | Control TKE | MF TKE | ΔTKE | Control MVPE | MF MVPE | ΔMVPE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.245167 | 0.251641 | +2.64% | 0.774996 | 0.819054 | +5.69% | 0.213843 | 0.224508 | +4.99% |
| 1000 | 0.200521 | 0.196584 | -1.96% | 0.658004 | 0.655270 | -0.42% | 0.166719 | 0.179220 | +7.50% |
| 1500 | 0.193675 | 0.188409 | -2.72% | 0.633786 | 0.645156 | +1.79% | 0.165178 | 0.160552 | -2.80% |

Lower is better. At 1500, MF-01 won Rel-L2 on 13/16 trajectories, TKE on
7/16, MVPE on 10/16, and all three on 3/16.

Official v9 subscores at 1500 (Rel/TKE/MVPE/Time/SPS): Control
`91.1712 / 75.9363 / 92.3712 / 84.4044 / 8.6737`; MF-01
`91.3906 / 75.6099 / 92.5689 / 79.1123 / 7.9087`.

## Analysis-only decomposition

Mean diagnostic fields:

- `mean_rel_l2_control = 0.144862`
- `mean_rel_l2_mf01 = 0.141107`
- relative change: `-2.593%`
- trajectory win count: `10/16` (trajectory-level macro diagnostic)

Fluctuation diagnostic fields:

- `fluct_rel_l2_control = 1.519750`
- `fluct_rel_l2_mf01 = 1.479491`
- relative change: `-2.649%`
- trajectory win count: `16/16` (trajectory-level macro diagnostic)

TKE diagnostic fields:

- official raw TKE: Control `0.633786`, MF-01 `0.645156`
- relative change: `+1.794%` error degradation
- trajectory win count: `7/16`

The mean and fluctuation diagnostics are computed per scored window and then
averaged; trajectory win counts use the corresponding trajectory-level macro
diagnostic. Lower is better.

MF-01 reconstructed fluctuation temporal-mean max absolute value:
`1.1772e-7`.

The Rel-L2/MVPE gains are not attributable to Mean alone: both Mean and
Fluctuation reconstruction diagnostics improve, with the fluctuation diagnostic
winning on 16/16 trajectories. However, fluctuation Rel-L2 improvement does not
translate into improved turbulent kinetic energy: official TKE worsens by
1.794% and wins only 7/16. Mark this explicitly as
`FLUCTUATION_REL_L2_IMPROVED_BUT_TKE_DEGRADED`; the fluctuation field itself did
not worsen under the reported reconstruction diagnostic.

## Runtime and artifacts

- Control active time: `1156.83 s`; MF-01 active time: `1932.81 s`
- Peak GPU allocation was not recorded by this runner; `nvidia-smi` showed about
  `18.8 GiB` during MF-01 training.
- Remote artifacts: `/home/chyfuture/realpde_runs/mf01_control_20260904/` and
  `/home/chyfuture/realpde_runs/mf01_s20260904/`
- Checkpoint SHA-256s are recorded in the run directories; 1500:
  Control `5499e60a3b8146bf095070dc76d03c85eae57b5f1ec794444276bab362458ec4`;
  MF-01 `488a8118f489789d385ec90e02856ef6a8482d6fa75c252e2e5d2d1f50e72226`.
- A container-path issue initially hid outputs from the host path; both stopped
  containers exited 0 and all checkpoints/predictions were recovered and verified.

Locked-final accessed: **NO**. Codabench accessed: **NO**.

`NEXT_ACTION = REVIEW_REQUIRED`
