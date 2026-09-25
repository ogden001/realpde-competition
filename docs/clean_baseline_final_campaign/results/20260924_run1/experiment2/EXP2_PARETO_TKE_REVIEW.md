# Experiment 2 — Pareto-TKE Residual Clean

Status: `REVIEW_REQUIRED`

## Outcome

**Gate: `NO_GO`.** No frozen beta passed all four Seen-Dev checks, so there is no selected checkpoint and AoA10 Holdout18 was not evaluated.

## Protocol and provenance

- Execution commit: `396dc89dec46ac1118a9b1dd33709784903d8e98`.
- Split: Train51 / Seen-Dev12 / AoA10 Holdout18; trajectory-disjoint; `7575_0.h5` excluded. Train stride 1; Seen-Dev stride 20, start 0.
- Both arms started from Clean Stage1 best SHA `6aec4edb5e42746080e19baaafa2d6b719bb57befb1ff9035ece7889ec280036`; sim-pretrain SHA `82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61`.
- Shared fixed settings: ResidualCorrector3D h96/b2/dropout0/max_delta0.04; 22,000 updates; batch 8; seed 41; AdamW lr `2e-4`, weight decay `1e-5`; loss weights point 1.0, MSE 0.05, temporal 0.03, grad 0.015, p_zero 0.01, residual-MSE 0.25, delta penalty 0.02.
- Control: TKE 0.06, scalar gradient. Pareto: TKE 0.12, `project_tke`. Runtime-only workers: 2.
- Exact init state parity: max abs diff `0.0`. Sampling audit identical: `True`; both audit SHA256 `4e5f6ab85ec1bfb264f5336390b6ed8a93740a85813df697489cc486d1c5c85a`.

## Seen-Dev results

Best checkpoints selected by point score (selection step is recorded in each arm summary). All values are raw errors; lower is better.

| Arm | Rel-L2 | TKE | MVPE | Point score |
|---|---:|---:|---:|---:|
| Matched control | 0.07899997 | 0.50985795 | 0.06491581 | 90.9140 |
| Pareto, uninterpolated | 0.08059170 | 0.49069273 | 0.06582545 | 91.0797 |

Pareto relative change vs matched control: mvpe_raw +1.401%, rel_l2_raw +2.015%, tke_raw -3.759%.

## Frozen interpolation gates

| beta | Rel-L2 | TKE | MVPE | mean change | Gate |
|---:|---:|---:|---:|---:|---|
| 0.50 | +31.650% | +28.037% | +38.968% | +32.885% | NO_GO |
| 0.65 | +23.774% | +19.292% | +33.141% | +25.402% | NO_GO |
| 0.80 | +10.615% | +5.605% | +16.042% | +10.754% | NO_GO |
| 1.00 | +2.015% | -3.759% | +1.401% | -0.114% | NO_GO |

Selected beta: `None`. Selected checkpoint SHA: `N/A`. The beta-1.00 candidate meets the TKE and mean-change checks, but fails the Rel-L2 and MVPE degradation limits.

## Runtime and safety

- Campaign elapsed: 6h 52m (including matched training, Seen-Dev evaluations, and frozen interpolation checks).
- Arm elapsed, model_init to DONE: control 3h 51m; Pareto 2h 47m.
- Sampled peak GPU memory: 10127 MiB; sampled host RAM used: 24Gi. Exp1 shared the GPU during part of Exp2.
- OOM: no. Locked-final/private: not accessed. Codabench: not accessed. Full-data refit/submission packaging: not started.

## Evidence map

- `campaign_summary.json`: parity, matched metrics, beta metrics/gates, Holdout access flag.
- `arms/`: full run configs and summaries, stepwise eval milestones, sampling audits, compact trajectory/horizon CSVs.
- `seen_dev/` and `interpolation/`: per-trajectory and per-horizon Seen-Dev metrics for control, Pareto, and all four betas.
- `checkpoint_and_sampling_sha256.json`: checkpoint digests only; checkpoint binaries are not included.
- `runtime_and_provenance.json`: commit, split manifest digest, environment, runtime, and safety fields.

All reported predictive metrics use Seen-Dev12. This package records evidence for Sol review; it makes no recommendation to merge the Pareto method.
