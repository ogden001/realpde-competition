# MF Energy Campaign 02 — Full Evidence Review

Status: `REVIEW_REQUIRED` / `ANALYSIS_ONLY_AUTHORIZED`

## Scope and provenance

This is prediction-only replay; no training, model/Loss/Feature change, SPS, locked-final, Codabench, full-data run, or Campaign03 design was performed.

- Base: `0687bf8c4e41f99e38a4638f9d13fc7ec0c735b9`; current `main` is its descendant.
- Fixed CLEAN split: 50 train / 16 Dev; manifest SHA `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`.
- Reused MF-01@1500, C0@2000/2500/3000, E4/E5/E6/E7/E8@3000 predictions and existing Dataset Profile.
- Analysis tool: `tools/analyze_mf_campaign02_review.py`.
- Artifacts: `artifacts/mf_campaign02_full_review_20260904/` and remote `/home/chyfuture/realpde_runs/mf_energy_campaign02/full_review_20260904/`.

## C0 convergence anatomy

Trajectory-macro means across 16 Dev trajectories (diagnostic errors; official columns are also in `checkpoint_convergence.csv`):

| Update | official Rel | official TKE | official MVPE | Mean error | Fluctuation error | TKE ratio |
|---:|---:|---:|---:|---:|---:|---:|
| MF@1500 | 0.187592 | 0.309327 | 0.123855 | 0.140704 | 1.469356 | 1.2649 |
| C0@2000 | 0.170966 | 0.345679 | 0.118200 | 0.125837 | 1.375686 | 1.0161 |
| C0@2500 | 0.163799 | 0.302300 | 0.086747 | 0.112686 | 1.410456 | 1.1474 |
| C0@3000 | 0.163528 | 0.297007 | 0.090627 | 0.112027 | 1.400134 | 1.1537 |

From MF@1500 to C0@3000, Rel, Mean error and MVPE improve on 16/16 trajectories; official TKE improves on 10/16. The large gain is therefore broad, not an outlier effect. Mean reconstruction supplies the clearest stable contribution: mean error drops about 20.4% and improves on 16/16. Fluctuation reconstruction also improves on 16/16, but only about 4.7% in the macro diagnostic and remains non-monotonic. The TKE ratio moves closer to one overall, but the 2000–3000 curve oscillates, so longer training improves energy organization only partially rather than monotonically.

By Dataset Profile, the 1500→3000 mean-error improvement is broad across Boundary, ID and OOD-like cases. OOD-like cases have higher TKE error and remain less stable; the MF@1500 tail regression is not exclusively OOD. Dynamic-tail descriptors remain useful descriptive risk markers, but no single profile family explains all outcomes.

## E5 RMS mechanism

Against C0, E5 TKE wins 15/16. This is substantive rather than a collection of tiny wins: TKE delta median `-0.03099`, P25 `-0.03963`, P75 `-0.01324`, mean `-0.02893`; the only loser is `22875_15` (`+0.00922`). E5 improves fluctuation diagnostic error on average (`-0.00276`) while mean error worsens on average (`+0.00162`), consistent with a Mean–Fluctuation trade-off, but not enough to prove causality.

MVPE worsens by `+0.002586` on average; the largest trajectory degradation is `24150_10` (`+0.01440`), where fluctuation error improves but mean error worsens `+0.00540`. Probe analysis does not support a probe-local explanation: median probe delta is `+0.00030`, mean is `-0.00080`, and 53.1% of trajectory×probe entries worsen. The damage is mixed across probes and trajectories, with evidence for global/trajectory-dependent mean-flow trade-off rather than a single bad probe.

Across horizons, E5's effect is present through the Future20 curve and is not confined to a late-horizon rescue; the complete per-horizon velocity, fluctuation and RMS columns are in `horizon_analysis.csv`.

## E6 anatomy

E6's Rel improvement is broad in Rel (12/16 wins versus C0) but does not transfer to MVPE: only 1/16 MVPE wins and aggregate MVPE worsens by about 10.9%. The decomposition does not establish a pure Mean or pure Fluctuation source; both components and spatial regions contribute, while the probe/mean-flow statistic is damaged. Target-derived high-energy weighting also fails its intended test: top-20% TKE-map improvement is negative for `26700_0` (`-7.24e-5`), `24150_10` (`-1.57e-4`) and `20325_20` (`-6.77e-6`). The weight emphasizes target energy during training but does not supply the missing predictive spatial organization.

## Negative evidence and cross-experiment map

- E4 has only 3/16 TKE wins versus C0 and is consistent with weak, non-broad benefit.
- E7/E8 remain effectively identity; prior alpha statistics and exact zero-init replay show no clamp/saturation artifact. Their final alpha standard deviations are `2.15e-5` and `4.22e-5`.
- C0 is the only arm with broad simultaneous improvement over MF@1500. E5 is the only arm with broad TKE wins, but it sacrifices MVPE. E6 is Rel-favorable but MVPE-destructive. E7/E8 are not useful corrections.
- No stable universal case family emerged. `26700_0`, `24150_10`, and `20325_20` remain informative high-energy bad cases, but their response differs by objective.

## Final mechanism ratings

| Hypothesis | Status |
|---|---|
| `MF1500_WAS_UNDERTRAINED` | `SUPPORTED` |
| `MF_LONGER_TRAINING_IMPROVES_MEAN` | `SUPPORTED` |
| `MF_LONGER_TRAINING_IMPROVES_FLUCTUATION` | `PARTIALLY_SUPPORTED` |
| `MF_LONGER_TRAINING_RESTORES_ENERGY_STRUCTURE` | `PARTIALLY_SUPPORTED` |
| `E5_RMS_PROVIDES_STABLE_TKE_SIGNAL` | `PARTIALLY_SUPPORTED` |
| `E5_TKE_GAIN_TRADES_AGAINST_MEAN` | `PARTIALLY_SUPPORTED` |
| `E5_MVPE_DAMAGE_IS_PROBE_LOCALIZED` | `NOT_SUPPORTED` |
| `E6_REL_GAIN_COMES_FROM_FLUCTUATION` | `INSUFFICIENT_EVIDENCE` |
| `E6_REL_GAIN_COMES_FROM_MEAN` | `INSUFFICIENT_EVIDENCE` |
| `E6_HIGH_ENERGY_WEIGHTING_WORKS_AS_INTENDED` | `NOT_SUPPORTED` |
| `GAIN_INPUT_SIGNAL_INSUFFICIENT` | `PARTIALLY_SUPPORTED` |
| `TEMPORAL_DYNAMICS_REMAINS_PLAUSIBLE` | `PARTIALLY_SUPPORTED` |
| `SPATIAL_ENERGY_STRUCTURE_REMAINS_PLAUSIBLE` | `PARTIALLY_SUPPORTED` |

### Verified facts

1. MF@1500 was materially under-converged; C0 continuation improves Rel/Mean/MVPE on all 16 Dev trajectories.
2. The largest stable C0 contribution is Mean reconstruction; fluctuation reconstruction improves too, but energy organization is noisier.
3. E5's 15/16 TKE wins are broad and sizeable, not merely threshold-level wins, but MVPE worsens broadly enough to fail the protected gate.
4. E5 MVPE damage is not localized to official probes.
5. E6's high-energy target weighting does not repair high-energy TKE fidelity and introduces a severe MVPE trade-off.

### Unresolved questions

- Whether E5's TKE benefit can be retained without mean-flow/MVPE damage.
- Whether MF fluctuation errors are primarily temporal-dynamics or spatial-energy-structure errors.
- Whether a richer runtime-safe signal exists that can support non-identity gain without joint-optimization collapse.

No new experiment is proposed here. `NEXT_ACTION = REVIEW_REQUIRED`.
