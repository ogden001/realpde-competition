# v5 base Adaptive Uncertainty calibration — `REVIEW_REQUIRED`

This is a calibration-only evaluation of the already-trained v5 base
uncertainty head@1400. No retraining, corrected-head training, corrector
refit, package, locked-final/private-test access, or Codabench submission was
performed.

## Frozen provenance

- Execution commit: `c90e556666876baecf2733c3ad47152c5710c74a`
- Backbone: canonical validation@30900, SHA-256 `e3d5faaf1a71e121b09077dd7dd7d0456a617e2916b8a671986f412fb54f6388`
- Base probe SHA-256: `be027b9bfb6431522e0a51586102e7d98a34acc6aefa7412e2aa52a294821db0`
- Manifest SHA-256: `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`
- Official scorer SHA-256: `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`
- Frozen Dev: 16 trajectories / 659 windows

## Result

The complete fixed 4×7 grid is in `calibration_grid_base.csv` and
`calibration_grid_base.json`.

| Candidate | Floor | Mult | SPS | Coverage | Mean UV width |
|---|---:|---:|---:|---:|---:|
| Base head best | `0.0025` | `1` | `41.496072` | `0.836394` | `0.0260351` |
| Canonical static reference | `0.0075` | `0.02` | `39.112385` | `0.735240` | recorded in prior SOTA evidence |

The base head grid best exceeds the canonical static reference by
`+2.383686` SPS points on the frozen Dev. The model prediction raw errors
remain the canonical baseline values because calibration changes only bounds:
Rel-L2 `0.1128446013`, TKE `0.5001028180`, MVPE `0.0872825533`.

Status remains `REVIEW_REQUIRED`; this result is not packaged or submitted.

## Evidence

- `calibration_grid_base.json`: full 28-row grid and provenance
- `calibration_grid_base.csv`: full 28-row tabular grid
- `base_uncertainty_calibration_v2.review.log`: complete calibration stdout/stderr
