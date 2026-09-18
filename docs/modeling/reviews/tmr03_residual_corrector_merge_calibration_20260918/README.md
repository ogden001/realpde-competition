# TMR-03 — Residual Corrector Merge Calibration

Status: `REVIEW_REQUIRED`.

This is the one-shot, analysis-only Dev16 replay requested after TMR-02. It
used execution commit `03eac259891f72e7260746f4f5b4b767d33100a4`, frozen
SOTA-V2 `@32500`, and frozen residual Corrector `@30000` (`1f72b329d9a160b206a6e9f4fce5e780022b3bac7a9ce4bdf5b8c9efa30b4506`). No training,
sweep, locked-final/private evaluation, SPS, uncertainty, packaging, or
Codabench access was performed.

## Raw Dev16 metrics

Improvements are relative reductions in the corresponding error metric.

| variant | Rel-L2 | Rel improvement | TKE | TKE improvement | MVPE | MVPE improvement | fluctuation energy ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.09993462 | 0.0000% | 0.46929273 | 0.0000% | 0.07577799 | 0.0000% | 0.87387983 |
| corrected | 0.08499187 | 14.9525% | 0.49182430 | -4.8012% | 0.07112862 | 6.1355% | 0.76358797 |
| window_energy | 0.08688745 | 13.0557% | 0.49914995 | -6.3622% | 0.07112862 | 6.1355% | 0.87387983 |
| spatial_tke_map | 0.08891030 | 11.0315% | 0.46929273 | 0.0000% | 0.07112862 | 6.1355% | 0.87387983 |

Projection audit passed. The maximum relative energy-map error was
`1.212e-7` for `window_energy` and `6.336e-6` for `spatial_tke_map`; maximum
absolute temporal-mean changes versus `corrected` were `1.490e-7` and
`2.980e-7`, respectively.

## Gain retention and trajectory stability

Relative to the raw `corrected` gain, `window_energy` retained 87.31% of the
Rel-L2 gain and 100.00% of the MVPE gain. `spatial_tke_map` retained 73.78% of
the Rel-L2 gain and 100.00% of the MVPE gain.

Trajectory-level comparisons use the 16 Dev trajectories in
`trajectory_metrics_long.csv`:

| variant | Rel improved /16 | Rel median / worst | TKE improved /16 | TKE median / worst | MVPE improved /16 | MVPE median / worst |
|---|---:|---:|---:|---:|---:|---:|
| corrected | 16/16 | 15.95% / 8.76% | 3/16 | -8.02% / -36.38% | 15/16 | 16.15% / -0.83% |
| window_energy | 16/16 | 14.50% / 4.96% | 15/16 | 12.64% / -1.96% | 15/16 | 16.15% / -0.83% |
| spatial_tke_map | 16/16 | 12.23% / 4.55% | 11/16 | 6.68% / -3.31% | 15/16 | 16.15% / -0.83% |

For the diagnostic horizon tables, the Rel-L2 and MVPE gains are strongest in
the early and late horizon groups and smaller in the middle group. The
`window_energy` projection reduces those gains modestly while preserving the
corrected temporal mean; its TKE-contribution diagnostic improves over the
corrected variant in all three groups only in the aggregate sense shown by the
tables, not as an official per-frame score. The `spatial_tke_map` preserves
the corrected temporal mean and restores the backbone TKE map, with smaller
Rel-L2 gains. Per-horizon TKE and MVPE fields are diagnostic decompositions,
not official per-frame scores.

## Evidence and scope

The CSV/JSON files in this directory are the lightweight evidence copied from
the remote replay run. `by_horizon.csv` has 80 data rows,
`by_trajectory_horizon.csv` has 1280 data rows, and
`horizon_trajectory_stability.csv` has 60 data rows.

The final decision remains with ChatGPT/Sol: promote one merge candidate to a
separate full-data decision, or close the Residual Corrector direction. No
follow-on experiment is authorized by this evidence record.
