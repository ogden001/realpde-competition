# MF Direction Closeout: Direct@3000 vs MF@3000

Status: `PROMISING_PARKED`

## Scope and execution

This is the final breadth-first Mean/Fluctuation fairness validation. Exactly one
Direct continuation was trained from the locked MF-01 matched Direct Control
@1500 checkpoint. No MF detail optimization, locked-final access, Codabench,
SPS, full-data training, loss/feature/MF/model changes, or new model was run.

- Experiment: `T1-ID-MF-DIRECT3000-CLOSEOUT-S20260901`
- Execution commit: `778ef878e1b7a896ff3d9fd0bec6f28739e464bd`
- Parent: `T1-ID-MF01-CONTROL-S20260904`
- Fixed CLEAN split: 50 Train / 16 Dev; manifest SHA `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`
- Protocol: P0-A, N2, seed `20260901`, AdamW `1e-5`, batch/effective batch `8`, workers `2`
- Official v9 scorer SHA: `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`
- Direct@1500 checkpoint SHA: `5499e60a3b8146bf095070dc76d03c85eae57b5f1ec794444276bab362458ec4`
- Direct@3000 checkpoint SHA: `9fa52c905c2603179da39e90da7689a1d65005d5c00d0df2d9fe2e310cf32aeb`
- Direct continuation restored the checkpoint `optimizer_state_dict`, matching Campaign02 C0 continuation semantics.
- Remote artifacts: `/home/chyfuture/realpde_runs/mf_direction_closeout_20260904/`

The execution used a clean archive of the actual Python dependencies. SHA-256:

| File | SHA-256 |
|---|---|
| `realpde_mf_direction_closeout.py` | `5ace01b3611e32591ad188d22ab3ac0b52bed67f10b990f362b39c1d3023bb0f` |
| `realpde_mf01.py` | `cd991f9a8d9d7cc7213211a7e7e626d59d83710d9e0de762bcc959563cc12198` |
| `realpde_loss_official_v9.py` | `6ded951b3aea29152ca1f75b82bdc009e79b1233ac5b31beb1bdd636c13070ce` |
| `realpde_p0_data.py` | `27f9c82e5fafd9f47f71e818e5d662788692ab31b8051a7445a173b3e1126025` |
| `realpde_p0_features.py` | `06bc2dd4882c1dd0e82246d13fe93ee967496dab9d498dd2d17f2ef223219532` |

## Official v9 Dev raw errors

Lower is better. MF@3000 is the existing Campaign02 C0 endpoint at absolute
update 3000 (its continuation summary records 1500 added updates).

| Model | Rel-L2 | TKE | MVPE |
|---|---:|---:|---:|
| Direct@1500 | 0.193675 | 0.633786 | 0.165178 |
| MF@1500 | 0.188409 | 0.645156 | 0.160552 |
| Direct@3000 | 0.175829 | 0.594649 | 0.151631 |
| MF@3000 | 0.164327 | 0.582928 | 0.130374 |

## Checkpoint curve

| Absolute updates | Rel-L2 | TKE | MVPE |
|---:|---:|---:|---:|
| Direct@1500 | 0.193675 | 0.633786 | 0.165178 |
| Direct@2000 | 0.174743 | 0.602396 | 0.160192 |
| Direct@2500 | 0.166000 | 0.572100 | 0.131284 |
| Direct@3000 | 0.175829 | 0.594649 | 0.151631 |
| MF@1500 | 0.188409 | 0.645156 | 0.160552 |
| MF@2000 | 0.171655 | 0.587980 | 0.155509 |
| MF@2500 | 0.164472 | 0.575277 | 0.129577 |
| MF@3000 | 0.164327 | 0.582928 | 0.130374 |

## MF@3000 versus Direct@3000

Relative delta is `(MF - Direct) / Direct`; negative is better.

| Metric | Relative delta | MF trajectory wins |
|---|---:|---:|
| Rel-L2 | `-6.541%` | `16/16` |
| TKE | `-1.971%` | `8/16` |
| MVPE | `-14.019%` | `15/16` |

The gain is broad for Rel-L2 and MVPE and positive, though smaller, for TKE.
There is no official-metric deterioration at MF@3000, and the rule requiring
at least two metrics to improve by approximately 3% is met.

## Direction decision

`PROMISING_PARKED`

Mean/Fluctuation formulation has clear value. The first-stage benefit is now
recorded and the direction is parked. Do not continue RMS decoupling, spectrum
analysis, MF-02, or further MF detail optimization.

`NEXT_ACTION = DIRECTION_CLOSED_FOR_BREADTH_FIRST_EXPLORATION`
