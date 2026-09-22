# Colleague 80pt V2 Three-Arm Campaign

Status: **READY_FOR_EXECUTION / MODE A / REVIEW_REQUIRED_AFTER_RUN**

This direction takes the colleague online **80.078849** system as the only
baseline and runs three independent, sufficiently trained experiments. The
purpose is to test complementary mechanisms without mixing variables into one
joint model.

## Frozen baseline

- Online Final: `80.078849`
- Rel-L2: `94.739714`
- TKE: `76.892538`
- MVPE: `93.845797`
- SPS: `40.209117`
- Stage-1 CNO SHA256:
  `ff28aaf0114d57e320e5ea9cf9945d874a7482f22e3f0aeaed5ebbe286573f8a`
- Residual SHA256:
  `909fdc7f8a6a42335e4ea4ce7471fc9a507135e9a29a4927bc7b7318be9c85b2`
- Colleague-aligned Dev raw:
  Rel-L2 `0.0804204196`, TKE `0.4491582513`, MVPE `0.0711169168`.

## Arm A — Pareto-TKE residual

Start from the mature colleague residual checkpoint. Keep the backbone frozen.
The historical residual objective is split into:

- primary: every existing residual loss term except TKE;
- energy: weighted TKE term with `lambda_TKE=0.12`.

If the two gradients conflict, only the TKE gradient is projected orthogonal to
the primary gradient. The primary gradient is never changed.

Budget: `12,000` updates, evaluate every `2,000`, batch `8`, AdamW
`2e-4`, same h96/b2/max-delta=0.04 residual.

Mechanical review gate only:

- TKE raw error improves at least 3%;
- Rel raw error degrades at most 0.5%;
- MVPE raw error degrades at most 0.3%.

The gate is evidence, not an automatic scientific decision.

## Arm B — Strong backbone transplant

Use the frozen existing SOTA-V2 P0-A/MF full@53582 checkpoint
(SHA256 `f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce`)
as the base predictor and
train a **fresh zero-init colleague h96/b2/max-delta=0.04 residual** with the
original colleague scalar residual objective.

No joint backbone training is allowed. This is competition-oriented evidence:
the strong backbone was fitted on all 82 released trajectories, while the
colleague residual protocol fits the 81-trajectory pool and evaluates on its
overlapping Dev16. It must not be described as clean holdout generalization.

Budget: `38,400` updates, evaluate every `4,800`, batch `8`, AdamW
`2e-4`.

Mechanical review gate:

- TKE raw error improves at least 3% versus current80;
- Rel and MVPE raw error each degrade at most 1%.

The runner also records base-before-residual metrics so Sol can judge whether
the residual preserves or destroys the strong-backbone advantage.

## Arm C — Adjacent-AoA mean-field

Reuse the already frozen
`run_aoa_meanfield_long_screen.py` experiment unchanged:

- same-Re nearest adjacent AoA;
- Past20 mean-field shift only;
- same shift applied to Past20 and Future20;
- probability 0.5;
- lambda uniform in [0.2, 0.5];
- 20,000 updates;
- no AoA/Re inference input.

## Non-goals

This campaign does **not**:

- combine A/B/C into one model;
- auto-launch a Combo;
- retrain a full submission pipeline;
- access locked-final/private data;
- build or submit Codabench packages;
- sweep weights, augmentation ranges, architecture width/depth, or SPS knobs.

Any merge decision happens only after Sol reviews the Git evidence.


## Batch-size optimization gate (3090 24G)

Before the long three-arm campaign, run an environment-only batch benchmark.
This is not a scientific experiment and must not inspect validation quality.

Only two frozen profiles are allowed:

| Arm | b8 profile | b16 profile |
| --- | --- | --- |
| A Pareto-TKE | b8, lr 2e-4, 12k, eval 2k | b16, lr 2.8e-4, 6k, eval 1k |
| B Strong Backbone | b8, lr 2e-4, 38.4k, eval 4.8k | b16, lr 2.8e-4, 19.2k, eval 2.4k |
| C AoA Mean-Field | b8, lr 2e-4, 20k, eval 2.5k | b16, lr 2.8e-4, 10k, eval 1.25k |

The b8/b16 profiles have exactly equal sample exposure per arm. For Arm C,
historical b8 step 5000 maps to b16 step 2500.

Each arm benchmarks 200 synchronized training steps at b8 and b16. b16 is
selected only if all conditions pass:

- b16 run succeeds without OOM;
- samples/sec >= 1.20 × b8;
- peak allocated VRAM <= 92% of device memory;
- peak reserved VRAM <= 96% of device memory.

Otherwise that arm falls back to b8. The selection code is frozen in
`tools/colleague_80pt/batch_profiles.py`.

Benchmark outputs:
- `benchmark_results.json`
- `selected_profiles.json`

The formal campaign accepts the latter with `--profile-json` and verifies
that batch, LR, updates and evaluation intervals exactly match a frozen
profile. Codex must not manually edit the profile JSON.
