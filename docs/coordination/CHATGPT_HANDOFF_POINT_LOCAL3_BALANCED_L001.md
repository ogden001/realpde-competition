# Point LOCAL3 balanced-loss handoff

Reference: `T1-ID-POINT-LOCAL3-BALANCED-L001-S20260902`

## Result

The bounded random-init run completed Phase A at `1500/1500` updates and
stopped at the preregistered screen gate:
`STOP_BALANCED_LOCAL3_EARLY`.

| model | Rel-L2 | TKE | MVPE |
|---|---:|---:|---:|
| PERSIST | 0.13557753 | 1.00000000 | 0.13270709 |
| LOCAL3, lambda=0.001 | 0.14545619 | 0.85760111 | 0.13071164 |

Candidate relative to PERSIST: Rel-L2 `-7.286%`, TKE `+14.240%`, MVPE
`+1.504%` (improvement is positive; TKE therefore worsened by 14.240%). The
screen requires Rel-L2 > 0%, MVPE > 0%, and TKE degradation ≤ 10%; it failed
Rel-L2 and TKE. No 7500-step continuation was run.

## Frozen protocol

- Sole variable: `MSE + 0.05*TKE` → `MSE + 0.001*TKE`.
- Choice was preregistered from the train-only gradient diagnostic:
  `0.05 / 44.764 ≈ 0.00112`, frozen at `0.001`.
- Random initialization; no old LOCAL3 checkpoint reused. Seed `20260901`,
  batch `8`, B3_PACKED, fixed seeded-shuffled train order, AdamW `1e-4`.
- LOCAL3 input/output and architecture unchanged: replicate-padded 3×3 raw
  u/v history plus normalized x/y, `362→256→256→256→128→40` GELU, raw
  residual u/v and zero pressure channel.
- 50 train trajectories / 16 dev trajectories; official v9 dev scorer only
  after Phase A. Locked-final and Codabench were not accessed.

Train-only gradient snapshots stayed finite. Median gradient ratio
`||g_0.001*TKE||/||g_MSE||` was `8.888` at initialization and `1.631` at
update 1500 (four fixed train batches each); no optimizer step was used for
the snapshots.

## Evidence

- Remote run: `/home/chyfuture/realpde_runs/point_local3_balanced_l001_s20260902`
- Remote status: `artifacts/status.json`; checkpoint: `artifacts/last@1500.pt`
- Local small review artifacts (checkpoint excluded from Git):
  `artifacts/point_local3_balanced_l001_s20260902_review/`
- Required files: `report.md`, `README_FOR_CHATGPT.md`, `metrics.csv`,
  `training_curve.csv`, `gradient_snapshot.csv/json`, `screening_gate.json`,
  `summary.json`, `run_metadata.json`, `status.json`.
- Frozen manifest SHA-256:
  `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`.
- `last@1500.pt` SHA-256:
  `95df4c9e69a287dc55290208a07a754e2cebf2706ead521e407e4f68347d6122`.
- Executed runner SHA-256:
  `55eb38d0c7a99e3265adee9a8f7aa696f7caf198d606c099b240a2699ff41a5d`.
- Local implementation commits: `931d039`, `8e9bed6`, `a143332`.

## Boundary / interpretation

This result supports only the bounded statement that reducing the TKE weight
to `0.001` did not rescue this LOCAL3 formulation at the preregistered
1500-step screen. It does not establish that `0.001` is globally optimal, that
all Point models fail, or that any new loss/architecture experiment is
authorized. `LOCAL5`, feature inputs, global encoders, locked-final and
Codabench remain out of scope pending ChatGPT/Sol review.
