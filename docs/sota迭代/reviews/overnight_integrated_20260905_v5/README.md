# Overnight Integrated Adaptive Probe v5 — `REVIEW_REQUIRED`

Execution commit: `b451c18e16d0cd11b66f8ff8927d640123d6e2ad`.
Required base commit: `1ea0e29fb122079887425ecdfc99d51db1a32fd1`.

## Scope and provenance

This run followed `docs/sota迭代/NEXT_ACTION.md` exactly. The auxiliary
pipeline inherits `P0FeatureConfig` from the selected backbone checkpoint;
validation uses the exact update-30900 checkpoint, frozen 50/16 manifest,
official v9 scorer, and 16 Dev trajectories / 659 windows.

- Validation backbone SHA-256: `e3d5faaf1a71e121b09077dd7dd7d0456a617e2916b8a671986f412fb54f6388`
- Validation feature config: `dx=+0.001710832`, `dy=-0.001710832`, P0-A only
- Frozen manifest SHA-256: `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`
- Official scorer SHA-256: `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`
- Remote run root: `/home/chyfuture/realpde_runs/overnight_integrated_20260905_v5/`
- Container artifact root: `/runs/overnight_integrated_20260905_v5/`

The preflight reproduced canonical baseline raw errors and exact prediction
parity before training. Validation corrector and base uncertainty head were
then trained from fresh initialization for the fixed 2400 and 1400 updates;
no v4 weights were reused.

## Results

Preflight baseline: Rel-L2 `0.1128446013`, TKE `0.5001028180`, MVPE
`0.0872825533`; prediction parity max absolute difference `0.0`.

The fixed Gate failed:

| Metric | Base | Corrected / Gate | Gate condition |
|---|---:|---:|---|
| Rel-L2 | `0.1128446013` | `0.0964947194` | improvement `14.49%` — pass |
| TKE | `0.5001028180` | `0.5254194140` | degradation `5.06%` — fail |
| MVPE | `0.0872825533` | `0.0779585391` | improvement `10.68%` — pass |

Six of sixteen trajectories exceed the `15%` TKE degradation limit. Gate
status is `CORRECTOR_NO_GO`. Per the frozen action, the result is
`REVIEW_REQUIRED / NO_FULL_REFIT`; no corrected-head training, all-82 refit,
package, locked-final/private-test access, or Codabench submission occurred.

## Base uncertainty calibration

Using the already-trained base uncertainty head@1400, the fixed 28-row grid
was evaluated without retraining. The best row is `floor=0.0025,
mult=1`, SPS `41.496072`, coverage `0.836394`, and mean UV width `0.0260351`.
This exceeds the canonical static reference (`floor=0.0075,
relative=0.02`, SPS `39.112385`) by `+2.383686` SPS points. See
`BASE_UNCERTAINTY_CALIBRATION.md` and `calibration_grid_base.json`.
Calibration execution commit: `c90e556666876baecf2733c3ad47152c5710c74a`.

## Reproduction and evidence

The exact remote commands are represented by the complete logs in this
directory. `validation_corrector_training.review.log` has 2400 update rows;
`validation_base_head_training.review.log` has 1400 update rows.

- `runtime_snapshot.json`: host, released-data, scorer, and checkpoint provenance
- `validation_sps_screen.json`: canonical 30900 preflight raw metrics
- `baseline_parity.json`, `trajectory_metrics.csv`: fixed-config prediction parity
- `metrics.json`: validation training metadata and update histories
- `gate_result.json`, `trajectory_gate.csv`: official Gate result and per-trajectory TKE evidence
- `calibration_grid_base.json`, `calibration_grid_base.csv`: fixed base-head calibration grid
- `BASE_UNCERTAINTY_CALIBRATION.md`: calibration handoff and static-reference comparison
- `artifact_manifest.json`: generated remote artifact manifest; the probe is inference-only
- `*.review.log`: complete stdout/stderr or per-update raw logs, including warnings/errors

An initial attempt using a pre-created output directory was refused by the
runner before training; it produced no experiment artifact. The successful
fixed-budget run is retained in the two per-update logs and runner stdout log.
