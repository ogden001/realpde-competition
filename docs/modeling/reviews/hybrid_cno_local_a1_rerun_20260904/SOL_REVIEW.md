# Sol Review — Hybrid CNO + Local A1 Valid Rerun

Status: **`WEAK_SIGNAL_PARKED`**

This is the final Sol review of `T1-ID-HYBRID-CNO-LOCAL-A1-RERUN-S20260904`.
The earlier A1 run was invalid because the local branch was absent from the optimizer; this rerun fixed that implementation error and passed the preflight audit.

## 1. Experiment validity

The rerun is accepted as valid evidence for this specific A1 implementation.

- Optimizer contains all `136` global and `4` local trainable parameter tensors, with `0` missing and `0` duplicates.
- Step-0 zero-init equivalence holds exactly: full output equals global output and local output is zero.
- After one optimizer step, local final-projection gradient and parameter delta are non-zero; after three steps, upstream local Conv weights also change.
- Local output becomes non-zero after training steps.
- Local checkpoint save/reload round-trip has max absolute error `0.0`.
- Locked-final and Codabench were not accessed.

Therefore the rerun can be used for architecture judgment.

## 2. Matched checkpoint evidence

Lower raw error is better. Relative improvement below is `(Direct - A1) / Direct`, so positive means A1 is better.

| Absolute update | Metric | Direct | A1 | A1 improvement |
|---:|---|---:|---:|---:|
| 2000 | Rel-L2 | 0.174743 | 0.174834 | -0.052% |
| 2000 | TKE | 0.602396 | 0.598865 | +0.586% |
| 2000 | MVPE | 0.160192 | 0.161721 | -0.954% |
| 2500 | Rel-L2 | 0.166000 | 0.164873 | +0.679% |
| 2500 | TKE | 0.572100 | 0.570977 | +0.196% |
| 2500 | MVPE | 0.131284 | 0.130400 | +0.674% |
| 3000 | Rel-L2 | 0.175829 | 0.176331 | -0.286% |
| 3000 | TKE | 0.594649 | 0.591687 | +0.498% |
| 3000 | MVPE | 0.151631 | 0.155084 | -2.277% |

The important process-level fact is that **TKE improves at all three matched checkpoints**, while Rel-L2 and MVPE are not stable. A1@2500 gives a small all-three improvement, but the gain is below 1% and disappears by update 3000.

The 2500→3000 degradation is not unique to A1: the matched Direct continuation also worsens over the same interval. Therefore A1@3000 alone must not be interpreted as evidence that the local branch caused the whole late degradation.

## 3. Case and horizon evidence

At A1@3000, trajectory wins are:

- Rel-L2: `2/16`
- TKE: `10/16`
- MVPE: `1/16`

Only `3750_0.h5` improves all three metrics. This reinforces the interpretation that the local branch contains a reproducible TKE/energy signal, but does not robustly improve reconstruction quality.

By horizon, A1@3000 mainly hurts Rel-L2/MVPE over approximately `t+3 ... t+17`, while a few early/late horizons improve. This pattern is consistent with a local correction whose temporal/phase calibration is not robust across the full Future20 horizon.

The final local residual is genuinely active and small rather than dead: sampled Dev windows show residual/global-output ratios on the order of roughly `1%` (with variation across trajectories/windows). Runtime increases from `60.29` to `61.89 ms/window`, about `+2.65%`.

## 4. Evidence limitations

- The raw training log for this rerun is empty, so step-level loss/gradient evolution cannot be retrospectively audited. Milestone metrics and preflight evidence remain available. Future formal runs should follow `docs/TRAINING_LOG_REVIEW_PROTOCOL.md`.
- `trajectory_case_table.csv` does **not** contain the input-side descriptor columns claimed by Luna's `report.md`; therefore descriptor-based good/bad-case linkage is not treated as verified evidence here.
- Luna's `report.md` states that negative delta is improvement, but its implemented delta convention is `(baseline - candidate) / baseline`; under that convention **positive** means improvement. This Sol review uses the correct sign convention.

## 5. Sol decision

**Current A1 implementation: `WEAK_SIGNAL_PARKED`.**

Reason:

1. the implementation is now valid;
2. local modeling produces a repeatable TKE signal across matched checkpoints and 10/16 final trajectories;
3. the best matched checkpoint shows only sub-1% all-metric gain;
4. Rel-L2/MVPE gains are unstable and final trajectory evidence is poor;
5. the effect is too small to justify kernel/channel/gate sweeps under the 60-point exploration rule.

This does **not** close the broader `Local + Global` hypothesis, but the specific `raw-u/v lightweight Conv3D output residual` mechanism should not receive further tuning now.

Next Architecture breadth-first experiment should move to a genuinely different mechanism, preferably **Multi-scale / coarse+fine modeling**, rather than another Local-residual variant.
