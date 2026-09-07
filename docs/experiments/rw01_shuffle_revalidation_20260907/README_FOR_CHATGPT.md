# RW-01 Shuffle Revalidation Review

Status: `STOP_REVIEW_REQUIRED`; final research status: `REVIEW_REQUIRED`.

## Contract and execution

- Required base: `28a235d9dacb60ab7f66297c73a2598bebe19c25`.
- Implementation / execution commit: `7a862cc6c7ac3629244a9b3e90c404843714f5fa`.
- Frozen protocol: 50 Train / 16 Dev, P0-A CNO, N2, `sim_pretrain/sim_cno.pth`, seed `20260901`, AdamW `1e-5`, batch 8, workers 2.
- The only implementation change is deterministic global shuffling after each epoch's independently selected trajectory phases. Dev remains fixed `start=0, stride=20`.
- RW-CTRL used the Random Phase sampler path with every trajectory forced to phase 0 and global shuffle. It ran 1000 updates. RW-01 was not started because the registered control gate stopped.

## RW-CTRL @1000

| metric | RW-00 reference | RW-CTRL | relative change |
|---|---:|---:|---:|
| Rel-L2 | 0.200521 | 0.196922 | -1.79% error (better) |
| TKE | 0.658004 | 0.654465 | -0.54% error (better) |
| MVPE | 0.166719 | 0.181560 | +8.90% error (worse) |

The control threshold was Rel-L2 or MVPE degradation >5%, or TKE degradation >10%. MVPE exceeded its threshold, so `control_gate.json` records `STOP_REVIEW_REQUIRED`. No RW-01 Random Phase result, trajectory comparison, or formal @7500 gate exists; no further experiment was started.

## Window and process evidence

- `window_audit_summary.json`: four fully traversed sampler epochs; 50 trajectories; 2052 selected windows per epoch; all 200 recorded phases are 0; `invalid_window_count=0`, `non_stride20_count=0`.
- `training.review.log` and `.meta.json`: deterministic full copy of the five-line raw control log, including the three announced phase-0 trajectory starts and the @1000 official scorer evaluation.
- `artifact_manifest.json`: update 0 and 1000 checkpoints with SHA-256 provenance. They do not carry optimizer state and are not resumable.
- Full remote output, including `window_audit.jsonl` and raw log: `/home/chyfuture/realpde_runs/rw01_shuffle_revalidation_20260907/output`.

This is execution evidence only. ChatGPT / Sol must review why the matched phase-0 shuffled control changed MVPE before any new experimental decision.
