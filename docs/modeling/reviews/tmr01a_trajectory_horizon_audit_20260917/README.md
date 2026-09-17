# TMR-01A — TMR-01 trajectory × horizon audit

Final status: `REVIEW_REQUIRED`.

This is an analysis-only replay of the frozen TMR-01 `@32500` backbone plus its existing corrector. No training, alpha selection, or new experiment was performed. The per-horizon TKE and MVPE quantities below are explicitly diagnostics, not official per-frame scores.

## Provenance and parity

- Required / execution commit: `3cbc779b33a1b2cb6c0572a05dadbaf53774732a`.
- Manifest SHA256: `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`.
- Frozen `@32500` backbone SHA256: `6926722895611c38dee79cf16ce43b87c7de3d4a1305e9db431dadd4c1a7bf47`.
- Frozen TMR-01 corrector SHA256: `fa1c1b46f71a16da7b08c2f36a4de0f735dd9880b007c636d1b774ccbe084dc0`.
- Official v9 scorer SHA256: `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`.
- Runtime: `realpde-pytorch-h5py:0831`, PyTorch `2.2.2+cu121`, NumPy `1.26.4`, h5py `3.16.0`, RTX 3090.
- Scope: Dev16, 659 canonical windows, Future20, alpha `0 / 0.5 / 1.0`.

Aggregate replay parity was exact for all three physical metrics:

| Alpha | Rel-L2 | TKE | MVPE | Max absolute delta vs TMR-01 |
|---:|---:|---:|---:|---:|
| 0.0 | 0.0999346152 | 0.4692927301 | 0.0757779852 | 0 |
| 0.5 | 0.0929593444 | 0.4856766164 | 0.0739125460 | 0 |
| 1.0 | 0.0897218958 | 0.4985924065 | 0.0733211562 | 0 |

## Alpha=1 horizon stability

Early/middle/late is defined as h1–h7 / h8–h14 / h15–h20. Improvements are `1 - candidate / alpha0`; negative values mean diagnostic error worsened.

| h | Rel improved /16 | TKE diag improved /16 | MVPE diag improved /16 | Median Rel | Median TKE | Median MVPE |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16/16 | 16/16 | 16/16 | +34.74% | +13.36% | +31.56% |
| 2 | 16/16 | 15/16 | 16/16 | +19.64% | +6.48% | +22.01% |
| 3 | 16/16 | 14/16 | 16/16 | +17.17% | +8.79% | +32.45% |
| 4 | 14/16 | 9/16 | 12/16 | +4.66% | +0.45% | +4.36% |
| 5 | 15/16 | 5/16 | 14/16 | +4.15% | -0.49% | +3.66% |
| 6 | 15/16 | 3/16 | 12/16 | +2.68% | -1.49% | +3.54% |
| 7 | 16/16 | 3/16 | 16/16 | +3.83% | -0.81% | +10.39% |
| 8 | 16/16 | 1/16 | 16/16 | +3.74% | -1.05% | +9.82% |
| 9 | 16/16 | 3/16 | 14/16 | +3.32% | -1.36% | +5.27% |
| 10 | 15/16 | 4/16 | 13/16 | +2.80% | -1.25% | +2.50% |
| 11 | 15/16 | 6/16 | 11/16 | +2.67% | -0.64% | +1.38% |
| 12 | 15/16 | 6/16 | 11/16 | +2.77% | -0.47% | +1.69% |
| 13 | 15/16 | 9/16 | 11/16 | +2.89% | +0.05% | +1.88% |
| 14 | 16/16 | 8/16 | 14/16 | +3.56% | -0.02% | +6.71% |
| 15 | 16/16 | 12/16 | 16/16 | +8.38% | +2.80% | +20.67% |
| 16 | 16/16 | 11/16 | 16/16 | +12.19% | +3.93% | +24.27% |
| 17 | 16/16 | 9/16 | 16/16 | +10.30% | +3.44% | +21.17% |
| 18 | 16/16 | 15/16 | 14/16 | +7.33% | +3.03% | +11.58% |
| 19 | 16/16 | 16/16 | 14/16 | +12.78% | +9.46% | +15.72% |
| 20 | 16/16 | 16/16 | 16/16 | +24.69% | +40.48% | +36.87% |

The mechanism pattern is non-monotonic: Rel and MVPE gains are strongest at h1–h3, flatten across h4–h14, and rise again at h15–h20. TKE diagnostic improvement is weakest at h5–h12, then becomes positive again at h15–h20.

## Ten worst TKE diagnostic points for alpha=1

These are sorted by TKE diagnostic improvement ascending. The paired Rel/MVPE values are shown for the same trajectory×horizon.

| Trajectory | h | TKE improvement | Rel improvement | MVPE improvement |
|---|---:|---:|---:|---:|
| 10125_0.h5 | 5 | -7.67% | +7.37% | +4.44% |
| 10125_0.h5 | 6 | -7.45% | +4.41% | -0.05% |
| 10125_0.h5 | 7 | -5.47% | +4.30% | +11.44% |
| 10125_0.h5 | 8 | -4.79% | +4.70% | +11.57% |
| 10125_0.h5 | 9 | -4.23% | +5.84% | +4.98% |
| 10125_0.h5 | 4 | -4.20% | +9.47% | +13.79% |
| 20325_5.h5 | 17 | -3.95% | +8.90% | +20.09% |
| 24150_10.h5 | 16 | -3.81% | +6.58% | +20.77% |
| 11400_10.h5 | 9 | -3.75% | +3.43% | +3.17% |
| 11400_10.h5 | 8 | -3.73% | +4.16% | +5.65% |

## Ten strongest Rel diagnostic points for alpha=1

| Trajectory | h | Rel improvement | TKE improvement | MVPE improvement |
|---|---:|---:|---:|---:|
| 3750_20.h5 | 1 | +38.74% | +4.00% | +46.87% |
| 8850_20.h5 | 1 | +38.24% | +11.78% | +47.49% |
| 11400_15.h5 | 1 | +38.01% | +18.39% | +35.60% |
| 8850_5.h5 | 1 | +37.99% | +18.36% | +36.97% |
| 10125_0.h5 | 1 | +36.41% | +8.10% | +32.61% |
| 8850_10.h5 | 1 | +36.23% | +18.56% | +31.50% |
| 3750_0.h5 | 1 | +36.09% | +22.57% | +33.65% |
| 20325_20.h5 | 1 | +34.80% | +14.55% | +30.37% |
| 11400_10.h5 | 1 | +34.67% | +20.11% | +30.72% |
| 20325_5.h5 | 1 | +34.48% | +15.82% | +31.85% |

The strongest Rel points are all h1 and have positive TKE diagnostics; the largest TKE losses occur mainly in the middle horizon and are not the same points as the strongest early Rel gains.

## Per-trajectory pattern (alpha=1)

`Rel peak` is the segment with the largest mean Rel improvement. `TKE min` is the segment with the smallest mean TKE diagnostic improvement; a positive value means that segment was still positive on average. `MVPE failures` lists horizons with negative diagnostic improvement.

| Trajectory | Rel peak | TKE min | MVPE failures |
|---|---|---|---|
| 10125_0.h5 | early (+15.2%) | middle (-1.2%) | h6, h11, h12 |
| 11400_10.h5 | late (+14.7%) | middle (-2.9%) | h4, h6, h11–h14 |
| 11400_15.h5 | late (+13.9%) | middle (-1.6%) | h4–h6, h10, h11 |
| 13950_5.h5 | late (+14.8%) | middle (-0.1%) | none |
| 16500_10.h5 | late (+11.6%) | middle (+0.1%) | none |
| 20325_20.h5 | early (+10.6%) | middle (-0.6%) | h4, h9–h13, h18–h19 |
| 20325_5.h5 | late (+13.2%) | middle (-0.1%) | none |
| 22875_15.h5 | late (+9.4%) | middle (-1.1%) | none |
| 24150_10.h5 | early (+9.7%) | middle (-1.3%) | h18–h19 |
| 25425_15.h5 | early (+9.6%) | middle (-1.1%) | none |
| 26700_0.h5 | early (+12.7%) | middle (+1.1%) | h11–h13 |
| 3750_0.h5 | early (+20.4%) | middle (+5.4%) | none |
| 3750_20.h5 | early (+17.9%) | middle (+2.5%) | h13 |
| 8850_10.h5 | late (+14.6%) | middle (-1.0%) | h4–h6, h12–h14 |
| 8850_20.h5 | early (+14.6%) | middle (+0.5%) | h9–h10 |
| 8850_5.h5 | late (+16.9%) | middle (-1.7%) | none |

## Scope

- Training: `NOT_PERFORMED`.
- SPS: `NOT_ACCESSED`.
- Uncertainty: `NOT_ACCESSED`.
- Full-data: `NOT_ACCESSED`.
- Locked-final/private: `NOT_ACCESSED`.
- Package: `NOT_BUILT`.
- Codabench: `NOT_ACCESSED`.
- TMR-02: `NOT_STARTED`.

The machine-readable evidence is in `aggregate_parity.json`, `by_horizon.csv`, `by_trajectory_horizon.csv`, `horizon_trajectory_stability.csv`, `run_metadata.json`, and `status.json`.
