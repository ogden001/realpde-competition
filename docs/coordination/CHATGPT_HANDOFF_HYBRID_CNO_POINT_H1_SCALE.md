# H1 Residual Scale Probe Handoff

Reference: `T1-ID-HYBRID-CNO-POINT-H1-SCALE-S20260902`

## Result

This was a replay/post-processing probe only. No neural-network training,
`optimizer.step()`, loss change, or checkpoint update was performed. The
existing H1 correction was evaluated as
`Y_alpha = Y_CNO + alpha * (Y_H1 - Y_CNO)` with CNO pressure copied unchanged.

Train-only selection over the fixed grid `0.0, 0.1, ..., 1.0` selected:

`alpha_star = 0.5`

The selection first required TKE degradation `<=5%`, then maximized Rel-L2
improvement with the preregistered tie-breaks. Dev was not used for selection.

## Dev evaluation

| model | alpha | Rel-L2 | TKE error | MVPE | Rel improvement | TKE improvement | MVPE improvement |
|---|---:|---:|---:|---:|---:|---:|---:|
| FROZEN_CNO | 0.0 | 0.19082105 | 0.64406884 | 0.14425756 | 0.000% | 0.000% | 0.000% |
| H1 original | 1.0 | 0.14110152 | 0.70690584 | 0.10270503 | 26.056% | -9.756% | 28.804% |
| H1 scaled | 0.5 | 0.15642738 | 0.66956341 | 0.11520854 | 18.024% | -3.958% | 20.137% |

The scaled candidate passed the fixed dev gate:
Rel-L2 improvement `>0`, MVPE improvement `>0`, and TKE degradation `<=5%`.
Decision: `GO_H1_SCALE_CALIBRATION`.

TKE is an error metric (lower is better), so `-3.958%` means TKE error
worsened by 3.958%, while remaining inside the 5% protection line.

## Provenance and boundary

- Original H1 state: `STOP_HYBRID_POINT_H1_EARLY` at `last@1500`.
- CNO checkpoint SHA-256:
  `499ec748cc5db1b7f3ad24029a464e921ad31c3d383b69626ec540eba392903e`
- H1 `last@1500` SHA-256:
  `fb6735bff296cc53f028b894b74691697f7475f5bbfda9ae8ee0dcd70d1e3bd2`
- Manifest SHA-256:
  `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`
- Scale-probe runner SHA-256:
  `6dde2508786836221359094b01a2ccb919dc0a937ca97efe5684bca6b916a5a8`
- Source commit: `524574c`
- Train: 50 trajectories; dev: 16 trajectories; batch 8; seed `20260901`;
  pipeline `B3_PACKED`.
- Replay equivalence: alpha 0 and alpha 1 max absolute difference `0.0`;
  pressure exact.
- `dev accessed: YES` (one fixed evaluation, after train-only selection)
- `dev used for alpha selection: NO`
- `locked-final: NO`
- `Codabench: NO`
- `H2: NOT EXECUTED`
- `joint training: NOT EXECUTED`

Evidence is in the remote run
`/home/chyfuture/realpde_runs/hybrid_cno_point_h1_scale_s20260902/artifacts_parent/artifacts/`
and the small local review directory
`artifacts/hybrid_cno_point_h1_scale_s20260902_review/` (not committed).

## Interpretation boundary

This supports only that a train-selected global scalar can retain substantial
Rel-L2/MVPE gain while bringing the H1 TKE trade-off inside the preregistered
line on this dev set. It does not establish global optimality, final-CNO
superiority, or authorize H2, joint training, per-channel/per-horizon scaling,
loss changes, or any new experiment.
