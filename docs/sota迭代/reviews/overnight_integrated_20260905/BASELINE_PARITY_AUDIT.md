# Baseline parity audit — `V4_INVALID_BASELINE`

Status: `REVIEW_REQUIRED / V4_INVALID_BASELINE`

This was a read-only audit. No training, full refit, package, locked-final/private-test access, or Codabench submission was performed.

## Frozen provenance

- Checkpoint: `/runs/p0a_n2_simreal_validation_20260903/continuation_10300_to30900/run/model_last.pth`
- Checkpoint SHA-256: `e3d5faaf1a71e121b09077dd7dd7d0456a617e2916b8a671986f412fb54f6388`
- Payload iteration: `30900`; feature set: `P0-A`
- Frozen manifest SHA-256: `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`
- Dev: 16 trajectories / 659 windows
- Official scorer SHA-256: recorded in `baseline_parity_v2.json`
- v4 run root: `/home/chyfuture/realpde_runs/overnight_integrated_20260905_v4/`

The checkpoint payload has no manifest SHA field (`null`); the manifest used by this audit is the frozen manifest above.

## Result

| Path | Feature spacing (`dx`, `dy`) | Rel-L2 | TKE | MVPE |
|---|---:|---:|---:|---:|
| Canonical / historical P0-A | `+0.001710832`, `-0.001710832` | `0.1128446013` | `0.5001028180` | `0.0872825533` |
| Current Gate baseline | `+0.003421664`, `-0.003421664` | `0.1299866587` | `0.6650677919` | `0.1606314480` |

The canonical path exactly reproduces the SOTA README validation@30900 values. Loader-only parity is exact: input and target diffs are zero, and predictions are `max_abs_diff=0`, `mean_abs_diff=0`. With the current Gate feature configuration, prediction differences are `max_abs_diff=0.1796105504`, `mean_abs_diff=0.0045644701`.

The v4 training metadata records the doubled spacing (`±0.003421664`) while the canonical checkpoint feature metadata records (`±0.001710832`). Thus the v4 corrector was trained against a non-canonical backbone/configuration. Per `NEXT_ACTION.md`, v4 validation is marked `V4_INVALID_BASELINE`; no rerun is authorized in this task.

## Evidence

- Machine-readable result: `baseline_parity_v2.json`
- Per-trajectory result: `trajectory_metrics_v2.csv`
- Complete audit stdout/stderr: `baseline_parity_audit_v2.review.log`
- Existing v4 Gate result is retained unchanged for provenance; its apparent gains are not promoted after this audit.
