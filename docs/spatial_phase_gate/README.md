# REALPDE Spatial Phase Gate V1

Status: `READY_FOR_EXECUTION / REVIEW_REQUIRED`

Deadline context: final submission is 2026-09-28. This is a single, bounded gate before any final long training. It must not grow into a parameter sweep.

## Question

Released real-PIV fields are 64x128, while the official model interface is 32x64. Historical training and evaluation use the fixed 2x subsampling phase:

- P00: `[..., 0::2, 0::2]`
- P01: `[..., 0::2, 1::2]`
- P10: `[..., 1::2, 0::2]`
- P11: `[..., 1::2, 1::2]`

The gate asks one question only:

> Does exposing the current best predictor to alternate 2x spatial sampling phases during a short matched continuation improve the official P00 Seen-Dev distribution?

This is observation-operator augmentation, not temporal Random Phase and not a 32x64 crop experiment.

## Frozen anchor

Use the selected Strong Backbone Clean predictor from the completed clean campaign:

- Strong Backbone SHA256: `d340effd68031aba1cd2bc610c676878fb9105dd7f81b9ce15767280e665a5c0`
- Strong Residual SHA256: `d2b4ddd0df6064d053ec918c2671a93b7d9c2a40a15afe854bb3a5badf1438fc`
- Its official P00 Seen-Dev result after residual correction:
  - Rel-L2: `0.07953662`
  - TKE: `0.49601257`
  - MVPE: `0.06441840`

Both gate arms start from the exact same two checkpoints.

## Frozen matched experiment

There are exactly two arms.

### Control

- Train phase: P00 only.
- Resume the selected Strong Residual state.
- Reset optimizer.
- 5,000 updates.
- Batch 8.
- LR `1e-5`.
- Train51, stride-1 temporal windows.
- Existing clean residual architecture and loss unchanged.
- Seed 41.

### Candidate

Everything is identical to Control except training spatial phase exposure:

- P00: 50%.
- P01/P10/P11: remaining 50%, balanced across the three alternate phases.
- Spatial phase assignment seed: `20260925`.
- Past20 and Future20 of a sample must use the same phase.

The candidate does not multiply the update budget or temporal sample budget. It changes only which 32x64 sub-lattice is observed for a training window.

The implementation freezes one deterministic spatial phase per legal temporal window. With 5,000 x 8 = 40,000 consumed samples versus 41,317 dense Train51 windows, this gate is intentionally approximately one dense epoch. If the gate passes, this fixed-per-window mechanism must **not** be silently assumed to be the final long-training schedule; long-training phase scheduling requires a separate Sol review.

## Evaluation

Evaluation is always the historical official P00 observation:

- Seen-Dev12 only.
- stride=20, start=0.
- no spatial augmentation.
- point metrics only for this gate: Rel-L2, TKE, MVPE and their mean v9 point score.
- each arm may select its best checkpoint within the frozen 0-5k continuation using the same point-score rule.

AoA10 Holdout18 is not read by this experiment. It is not needed to answer the gate question.

## Mandatory parity checks

Before accepting any scientific comparison:

1. arm initialization max absolute parameter difference must be exactly 0;
2. temporal sampling audit SHA must be identical between Control and Candidate;
3. Control consumed spatial phase must be 100% P00;
4. Candidate consumed phase exposure must be approximately 50% P00 and all P01/P10/P11 must be present;
5. the only intended scientific difference is training spatial phase.

Failure of any parity check means `BLOCKED`, not a scientific NO_GO.

## Frozen GO gate

Candidate is GO only when every check below passes versus the matched Control:

- mean relative raw-error change across Rel-L2/TKE/MVPE <= `-0.25%`;
- at least two of three raw metrics improve;
- no individual raw metric degrades by more than `0.50%`;
- v9 point score improves.

Otherwise: `NO_GO`.

Do not tune the 50% probability, seed, budget, LR, gate thresholds, or phase weights after seeing the result.

## Hard boundaries

Forbidden in this experiment:

- AoA augmentation or global velocity rotation;
- temporal Random Phase / random-start changes;
- loss-weight changes;
- TKE Pareto projection;
- backbone changes;
- SPS tuning;
- AoA10 Holdout evaluation;
- full-data refit;
- locked-final/private access;
- Codabench;
- submission/package construction;
- automatic second spatial-phase experiment;
- automatic final long train after GO.

All outputs stay `REVIEW_REQUIRED`. Codex executes and archives evidence; Sol makes the scientific decision.
