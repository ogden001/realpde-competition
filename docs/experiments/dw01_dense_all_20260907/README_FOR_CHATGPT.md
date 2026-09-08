# DW-01 Dense-All Temporal Supervision

Status: `REVIEW_REQUIRED`.

## Frozen contract and provenance

- Execution commit: `bc4fc7a14d611c30799f613148e8b1a3ba28dbfa` (`feat: add dense-all temporal window training`), which was equal to `origin/main` at launch.
- Train/dev manifest SHA-256: `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`; 50 train / 16 dev. No locked-final, Codabench, or full-data training was accessed.
- Initialization: official `sim_pretrain/sim_cno.pth`, SHA-256 `af85374bfd06c0e386ec803d777396c21484978392213025697c5a7470106b6b`.
- P0-A CNO, N2 (`MSE=1`, `TKE=0.05`, `Rel=0.027514`, `MVPE=0.009757`), AdamW `1e-5`, batch 8, workers 2, seed `20260901`; v9 scorer SHA-256 `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`.
- Sole variable: train window pool. DW-01 uses every legal start `0..T-40`, globally shuffled each epoch. Dev remains fixed `start=0`, stride 20.

## Dense audit

Canonical train pool was 2,052 windows. Dense-All was 40,488 windows (`19.730994x`): per-trajectory `min/median/max = 243/829/829`; 5,061 updates per dense epoch; 30k updates = 5.927682 dense epochs. Six shuffled epochs were audited; all had 40,488 windows and zero invalid windows. See `DW-01_Dense-All/window_audit_summary.json`.

## Raw dev curve

| Update | Dense epoch | Rel-L2 | TKE | MVPE |
|---:|---:|---:|---:|---:|
| 3,000 | 0.593 | 0.154785 | 0.613524 | 0.122739 |
| 5,000 | 0.989 | 0.150122 | 0.532512 | 0.120898 |
| 7,500 | 1.482 | 0.141061 | 0.518828 | 0.104072 |
| 10,000 | 1.978 | 0.132711 | 0.516492 | 0.112057 |
| 15,000 | 2.964 | 0.125811 | 0.492969 | 0.100903 |
| 20,000 | 3.956 | 0.120383 | 0.491995 | 0.090018 |
| 25,000 | 4.940 | 0.114120 | 0.485924 | 0.084585 |
| 30,000 | 5.928 | **0.110092** | **0.479399** | **0.080831** |

The run completed all 30,000 updates normally (`stop_reason=max_updates`) in 25,122.65 optimizer-active seconds. `update_curve.csv` is the machine-readable curve.

## Future20 prediction-only by-horizon replay (2026-09-08)

The registered DW-01 checkpoints at 7,500/20,000/30,000 updates were replayed on the frozen 50/16 manifest with fixed-start, stride-20 Future20 windows (659 windows, 16 trajectories). A matched RW-00 fixed@7,500 replay is included as an A/B reference. No training, locked-final data, Codabench, or checkpoint modification was used. Per-frame Rel-L2/RMSE, temporal-energy diagnostics, and the official v9 probe-geometry diagnostic are in `by_horizon/`.

| Replay | Rel-L2 | TKE | MVPE | Parity |
|---|---:|---:|---:|---|
| DW-01 @7.5k | 0.1410613 | 0.5188255 | 0.1040725 | PASS |
| DW-01 @20k | 0.1203829 | 0.4919937 | 0.0900179 | PASS |
| DW-01 @30k | 0.1100919 | 0.4793978 | 0.0808311 | PASS |
| RW-00 fixed @7.5k | 0.1492184 | 0.5285975 | 0.1183250 | PASS |

Window-horizon means (early h1–5 / mid h6–15 / late h16–20) show improvement from 7.5k to 30k: DW-01 frame Rel-L2 is 0.1183/0.1241/0.1787 at 7.5k and 0.0894/0.1008/0.1387 at 30k; late-horizon error remains dominant. Late temporal-energy ratio falls 2.27→1.43 and late probe error 0.2306→0.1988. These are diagnostics, not official per-frame leaderboard scores.

Artifacts: `by_horizon/by_horizon.csv` (20 rows per replay), `by_horizon/by_trajectory_horizon.csv` (trajectory × horizon), `by_horizon/summary.json`, and replay provenance metadata. **REVIEW_REQUIRED**: ChatGPT / Sol must perform final frame-by-frame review before any scientific KEEP/NO-GO decision.

## What this run does and does not establish

Within DW-01, 20k to 30k improved Rel-L2 and MVPE on all 16 dev trajectories. TKE improved on 2/16 and worsened on 14/16 trajectories, although aggregate TKE improved from `0.491995` to `0.479399`. This is a late-stage trade-off signal, not a claim of universal per-case TKE improvement; `trajectory_late_comparison.csv` contains every case.

The only available long canonical curve is `T1-ID-P0A-N2-VALIDATION-30900-S20260903`. Its recorded initialization is `sim_real_ft/sim_real_cno.pth` (SHA `82e842...4f61`), while DW-01 is `sim_pretrain`; several other provenance fields are missing in that historical metadata. `closest_official_warm_start_comparison.csv` gives transparent nearest-update deltas, but every row is `INCOMPATIBLE_PROVENANCE`. It must not be interpreted as a matched causal estimate of Dense-All. A matched canonical control is a Sol decision, not automatically launched here.

## Review package

- `DW-01_Dense-All/summary.json` and `run_metadata.json`: raw aggregate metrics and immutable run provenance.
- `DW-01_Dense-All/eval_*/trajectory_metrics.csv`: all 16 dev cases at every registered evaluation.
- `DW-01_Dense-All/training.review.log` and `.meta.json`: deterministic full-copy review log; remote raw source was `/home/chyfuture/realpde_runs/dw01_dense_all_codex_20260907/outputs/DW-01_Dense-All.train.log`, SHA-256 `d7cff3388ee071fad535ebd8ba22803c247ec637ee516ee1bce8a8ce8bfd8060`.
- `artifact_manifest.json`: hashes for Git review evidence. Checkpoints and full remote artifacts remain outside Git at `/home/chyfuture/realpde_runs/dw01_dense_all_codex_20260907/outputs/DW-01_Dense-All/`.

`NEXT_ACTION = REVIEW_REQUIRED`. No KEEP/NO-GO conclusion or follow-on experiment is authorized by this package.
