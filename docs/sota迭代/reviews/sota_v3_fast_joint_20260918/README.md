# SOTA-V3 FAST JOINT 50/16

Status: `FAST_JOINT_NO_GO / REVIEW_REQUIRED`

This run used the frozen 50 Train / 16 Dev protocol only. It started from
the existing SOTA-V2 `@32500` backbone and mature residual corrector `@30000`,
then ran the registered 5,000-update joint fine-tune with AoA augmentation.

## Frozen provenance

- Execution source: `c4dede34cf8f56fc8ee4e49279d6c8cb2b52169f`
- Backbone input: `@32500`, SHA256
  `6926722895611c38dee79cf16ce43b87c7de3d4a1305e9db431dadd4c1a7bf47`
- Residual input: `@30000`, SHA256
  `1f72b329d9a160b206a6e9f4fce5e780022b3bac7a9ce4bdf5b8c9efa30b4506`
- Frozen manifest SHA256:
  `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`
- Remote run root: `/home/chyfuture/realpde_runs/sota_v3_fast_joint_20260918`
- Runtime: `4472.47 s`; container exit `0`; OOM: `false`

The run used effective batch 8 (`micro-batch=2`, accumulation 4), backbone LR
`1e-6`, residual LR `1e-5`, 5,000 updates, 1,000-update milestones, and the
frozen `base -> residual -> spatial_tke_map -> final` path.

## Historical replay

The separate update-0 replay passed the `5e-6` guard against the projected
residual comparator. See `historical_replay.json`.

## Dev milestone results

| update | Rel-L2 | TKE | MVPE | normalized objective |
|---:|---:|---:|---:|---:|
| 0 | 0.0889103040 | 0.4692927301 | 0.0711286217 | 3.000000 |
| 1000 | 0.0918049887 | 0.4765858352 | 0.0776540935 | 3.139840 |
| 2000 | 0.0909927934 | 0.4805234969 | 0.0754998540 | 3.108809 |
| 3000 | 0.0918476209 | 0.4803361595 | 0.0777842849 | 3.150141 |
| **4000 selected** | **0.0898898393** | **0.4707271159** | **0.0733584464** | **3.045423** |
| 5000 | 0.0935068056 | 0.4884992838 | 0.0804780051 | 3.224068 |

The selected `@4000` checkpoint did not improve any of the three metrics over
the historical comparator. Average normalized error change was `-1.5141%`;
the single-metric guard also failed. Therefore the registered point gate is
`FAST_JOINT_NO_GO`, and Dev SPS was correctly skipped.

Selected output checkpoint SHAs:

- Backbone: `d776e1c16c91c92b79b3da0bad8c430475ab04682c6f711ae10656106e248cfe`
- Corrector: `dba3a6583410d2b678582e022a635dfd213ee12a68cb483f5acb31fa17ba7a3e`

## Verification and scope

The specified focused suite passed before execution and again during evidence
closeout: `32 passed`.
Evidence includes the aggregate milestone table, selected metrics, horizon
summary, trajectory metrics/anatomy, run metadata, status, and historical
replay JSON.

- `full-data NOT accessed`
- `locked-final/private NOT accessed`
- `Codabench NOT accessed`
- Dev SPS: `NOT RUN` because point Gate was `FAST_JOINT_NO_GO`
- No package was built.
- The stopped `sota_v3_merge_20260918` long-run was not resumed.
