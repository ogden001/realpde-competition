# Sol Review — Overnight Integrated Adaptive Probe v5

Status: **`SOL_REVIEWED / CORRECTOR_NO_GO / BASE_UNCERTAINTY_GO_PACKAGE`**

## Verified facts

1. Baseline parity is repaired and exact.
   - Validation backbone is the canonical P0-A + N2 update `30900` checkpoint.
   - Backbone SHA-256: `e3d5faaf1a71e121b09077dd7dd7d0456a617e2916b8a671986f412fb54f6388`.
   - Frozen manifest SHA-256: `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`.
   - Official scorer SHA-256: `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`.
   - Canonical baseline raw errors are reproduced: Rel-L2 `0.1128446013`, TKE `0.5001028180`, MVPE `0.0872825533`.
   - Standard validation prediction and adaptive baseline prediction have `max_abs_diff = 0`.
   - Auxiliary P0-A feature config inherits checkpoint spacing: `dx=+0.001710832`, `dy=-0.001710832`.

2. v5 is a fresh fixed-budget validation run.
   - Corrector: `2400` updates from fresh initialization.
   - Base uncertainty head: `1400` updates from fresh initialization.
   - No v4 corrector/head weights were reused.

3. The pre-registered corrector Gate fails.

| Metric | Baseline | Corrected | Relative change |
|---|---:|---:|---:|
| Rel-L2 | `0.1128446013` | `0.0964947194` | `+14.4888%` improvement |
| TKE | `0.5001028180` | `0.5254194140` | `5.0623%` degradation |
| MVPE | `0.0872825533` | `0.0779585391` | `+10.6826%` improvement |

Both TKE protections fail: aggregate TKE degradation is `5.0623%`, and `6/16` trajectories exceed the frozen `15%` degradation threshold.

4. Base uncertainty calibration is valid under repaired v5 semantics.
   - Static reference SPS: `39.112385`.
   - Best adaptive calibration: `floor=0.0025`, `mult=1`.
   - Adaptive SPS: `41.496072`.
   - Delta versus static: `+2.383686` SPS points.
   - Coverage: `0.836394`.
   - Mean UV width: `0.0260351`.
   - No retraining was performed for calibration.

## Sol decision

### Exact bounded-corrector recipe

**`STOP_FOR_SOTA / NO_FULL_REFIT`**.

The current corrector recipe is not promoted. Its Rel-L2/MVPE gains are real, but TKE damage is aggregate and trajectory-level. Do not spend the current SOTA sprint on parameter polishing of this recipe.

### Broader residual-correction mechanism

**`PARKED_SIGNAL`**.

The reconstruction signal is strong enough to revisit later only with a materially different TKE-safe mechanism.

### Base adaptive uncertainty head

**`GO_PACKAGE_REVIEW`**.

This head changes uncertainty bounds only and leaves the backbone prediction unchanged. The repaired canonical calibration improves official-v9 Dev SPS from `39.112385` to `41.496072`, so it is worth composing with the current online SOTA backbone `full@43260`.

The uncertainty head remains the validation-trained v5 `base_head@1400`; do not refit it on all-82 data. Freeze calibration at:

```text
floor = 0.0025
mult = 1.0
uv_half_width = floor + mult * sigma
pressure_half_width = 0
```

## Next action

Build one candidate only:

`P0-A + N2 full@43260 + v5 base Adaptive Uncertainty Head`

Requirements:
- no corrector;
- no retraining or recalibration;
- full@43260 prediction must remain numerically equivalent to the current backbone path;
- only `lower/upper` may change;
- package + clean-room smoke only;
- do not submit Codabench automatically.

`NEXT_ACTION = FULL43260_BASE_ADAPTIVE_PACKAGE_ONLY`
