# Sol Review — Overnight Integrated Adaptive Probe v5

Status: **`SOL_REVIEWED / CORRECTOR_NO_GO / BASE_UNCERTAINTY_REVIEW_PENDING`**

## Verified facts

1. Baseline parity is repaired and exact.
   - Validation backbone is the canonical P0-A + N2 update `30900` checkpoint.
   - Backbone SHA-256: `e3d5faaf1a71e121b09077dd7dd7d0456a617e2916b8a671986f412fb54f6388`.
   - Frozen manifest SHA-256: `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`.
   - Official scorer SHA-256: `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`.
   - Canonical baseline raw errors are reproduced: Rel-L2 `0.1128446013`, TKE `0.5001028180`, MVPE `0.0872825533`.
   - Standard validation prediction and adaptive baseline prediction have `max_abs_diff = 0`.
   - Auxiliary P0-A feature config now inherits checkpoint spacing: `dx=+0.001710832`, `dy=-0.001710832`.

2. v5 is a fresh fixed-budget validation run.
   - Corrector: `2400` updates from fresh initialization.
   - Base uncertainty head: `1400` updates from fresh initialization.
   - No v4 corrector/head weights were reused.
   - Test suite: `71 passed`.

3. The pre-registered corrector Gate fails.

| Metric | Baseline | Corrected | Relative change |
|---|---:|---:|---:|
| Rel-L2 | `0.1128446013` | `0.0964947194` | `+14.4888%` improvement |
| TKE | `0.5001028180` | `0.5254194140` | `5.0623%` degradation |
| MVPE | `0.0872825533` | `0.0779585391` | `+10.6826%` improvement |

Gate requirements on Rel-L2 and MVPE pass, but both TKE protections fail:
- aggregate TKE degradation must be `<=2%`, observed `5.0623%`;
- trajectories with TKE degradation `>15%` must be `<=2/16`, observed `6/16`.

The six severe TKE degradations are not a single-outlier failure: `10125_0`, `20325_20`, `20325_5`, `22875_15`, `24150_10`, and `25425_15` exceed the frozen `15%` threshold.

4. The run stopped correctly.
   - No corrected uncertainty head was trained after Gate failure.
   - No all-82 corrector refit was started.
   - No package, locked-final/private-test access, or Codabench submission occurred.

## Sol decision

### Exact bounded-corrector recipe

**`STOP_FOR_SOTA / NO_FULL_REFIT`**.

The result is valid and the Gate failure is substantive. The corrector produces large reconstruction gains in Rel-L2 and MVPE, but the energy/TKE damage is both aggregate and trajectory-level. The current frozen recipe must not be promoted to full@43260 or submission packaging.

Do not spend the current SOTA sprint on width/kernel/loss-weight/checkpoint sweeps for this same corrector recipe.

### Broader residual-correction mechanism

**`PARKED_SIGNAL`**.

Rel-L2 and MVPE improvements of roughly `14.5%` and `10.7%` are too large to call the mechanism useless. The verified failure mode is metric conflict: reconstruction correction is useful, but TKE protection is not stable. If reopened later, it needs a materially different TKE-safe mechanism rather than parameter polishing of the current recipe.

### Base adaptive uncertainty head

**`REVIEW_PENDING`**, independently of the failed corrector.

The v5 base uncertainty head was trained on the canonical frozen backbone and its training log remains finite/stable. It has not yet received the fixed adaptive SPS calibration screen under the repaired v5 semantics. Because uncertainty affects SPS without changing the backbone prediction metrics, it remains a separate low-cost candidate.

The old v4 adaptive-calibration numbers must not be reused because v4 used the invalid baseline/config path.

## Next action

Run **calibration only** for the already-trained v5 base uncertainty head:
- no retraining;
- fixed `floor = 0 / 0.0025 / 0.005 / 0.0075` × `mult = 0.5 / 1 / 1.5 / 2 / 2.5 / 3 / 4` grid;
- official v9 SPS on the frozen 16-trajectory / 659-window Dev split;
- compare directly with the canonical static-bounds reference `abs=0.0075, rel=0.02`, SPS `39.112385`;
- stop after evidence commit for Sol review; do not package or submit automatically.

`NEXT_ACTION = BASE_UNCERTAINTY_CALIBRATION_ONLY`
