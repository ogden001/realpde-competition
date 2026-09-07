# Matched MF Long Convergence — 2026-09-06

Status: `COMPLETED / REVIEW_REQUIRED`

## Scope and provenance

This was the explicitly requested MF Long run only. Checkpoint Soup, HCUR,
A2, locked-final, Codabench, SPS, and full-data training were not run.

- Execution commit: `3b7104cd52594a2fff8999cfd74c29178946bbe5`
- Experiment ID: `T1-ID-MF-LONG-CONVERGENCE-S20260906`
- Frozen manifest: 50 train / 16 dev; SHA-256
  `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`
- Official scorer: SHA-256
  `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`
- Direct@3000: `/home/chyfuture/realpde_runs/mf_direction_closeout_20260904/model_update_03000.pth`
  SHA-256 `9fa52c905c2603179da39e90da7689a1d65005d5c00d0df2d9fe2e310cf32aeb`
- MF@3000: `/home/chyfuture/realpde_runs/mf_energy_campaign02/c0/model_last.pth`
  file SHA-256 `35687e19953dd4b7502e1f1a2cadf291a5112f31d475bdf4591659393f23e23f`;
  embedded provenance references the expected MF@1500 source SHA
  `488a8118f489789d385ec90e02856ef6a8482d6fa75c252e2e5d2d1f50e72226`.
- MF metadata: `experiment_id=T1-ID-MF-C02-CONT-S20260901`, `mode=c0`,
  payload `iteration=1500`; no rebuild was performed.
- Frozen protocol: P0-A, N2, seed `20260901`, AdamW `1e-5`, batch `8`,
  workers `2`, optimizer state resumed, 12,000 additional updates per arm.

The runner preflight passed and reproduced the matched start point:

| Arm | Rel-L2 | TKE | MVPE |
|---|---:|---:|---:|
| Direct@3000 | 0.1758288145 | 0.5946488380 | 0.1516314447 |
| MF@3000 | 0.1643273234 | 0.5829281807 | 0.1303740144 |

Preflight evidence: `/home/chyfuture/realpde_runs/mf_long_convergence_20260906/preflight/preflight.json`.

## Paired convergence

Values are official v9 raw dev errors. Positive MF improvement means lower
error than Direct at the same absolute update.

| Absolute update | Direct Rel-L2 | MF Rel-L2 | MF Δ Rel-L2 | Direct TKE | MF TKE | MF Δ TKE | Direct MVPE | MF MVPE | MF Δ MVPE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3000 | 0.175829 | 0.164327 | +6.541% | 0.594649 | 0.582928 | +1.971% | 0.151631 | 0.130374 | +14.019% |
| 6000 | 0.143569 | 0.142823 | +0.520% | 0.530114 | 0.533705 | −0.678% | 0.118689 | 0.125881 | −6.059% |
| 9000 | 0.135327 | 0.133306 | +1.493% | 0.531304 | 0.530497 | +0.152% | 0.102030 | 0.101783 | +0.242% |
| 12000 | 0.133273 | 0.129750 | +2.644% | 0.508005 | 0.515253 | −1.427% | 0.108230 | 0.096024 | +11.279% |
| 15000 | 0.123458 | 0.122030 | +1.157% | 0.505743 | 0.504695 | +0.207% | 0.090781 | 0.092569 | −1.969% |

At the final 16-trajectory comparison, MF won Rel-L2 on `12/16`, TKE on
`13/16`, and MVPE on `6/16` trajectories.

## Gate and interpretation

The fixed strong gate requires, at the selected absolute update, at least
3% improvement in Rel-L2 and MVPE, with TKE degradation no worse than 2%.
At update 15000 the changes were Rel-L2 `+1.157%`, TKE `+0.207%`, and MVPE
`−1.969%`; no absolute update satisfied the strong gate. The runner therefore
reported `WASHED_OUT_OR_MIXED`.

This is a completed matched convergence result, not an automatic promotion:
MF retains a small final Rel-L2/TKE advantage, but its initial MVPE advantage
does not persist through the long continuation and the protected gate is not
met. Final research decision remains with ChatGPT/Sol.

## Reproduction and evidence

- Remote output root:
  `/home/chyfuture/realpde_runs/mf_long_convergence_20260906/run/`
- Log: `/home/chyfuture/realpde_runs/mf_long_convergence_20260906/run.log`
- Preflight: `preflight/preflight.json`
- Paired curve: `run/paired_convergence.csv`
- Final trajectory comparison: `run/final_trajectory_comparison.csv`
- Gate: `run/gate_result.json`
- Run metadata: `run/run_metadata.json`
- Direct/MF checkpoints: `run/direct/model_update_{06000,09000,12000,15000}.pth`
  and `run/mf/model_update_{06000,09000,12000,15000}.pth` (remote-only;
  not committed to Git).
- Static validation: `5 passed` for
  `tests/test_mf_long_convergence.py`; `py_compile` passed for
  `tools/realpde_mf_long_convergence.py`.
