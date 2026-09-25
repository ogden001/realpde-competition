# REALPDE Direct-CNO 7.5k Matched Control

Status: `REVIEW_REQUIRED`

This is the frozen one-variable control: `Dense-All + P0-A + Direct-CNO + N2 + Vorticity`, compared with the archived Strong Backbone `MF-CNO @ 7,500` result. Codex reports measurements only; Sol owns the final research interpretation.

## Protocol and validation

- Execution source commit: `3011f84cce888f976b2e0cd6ad1536b82bc81cc6`.
- Run: 7,500 optimizer updates exactly; seed 41; training batch 8; AdamW; LR `1e-5`; Stage-B not used.
- Data: Clean Train51 / Seen-Dev12; 41,317 dense train windows and 491 fixed dev windows.
- Official init: `sim_real_cno.pth`, SHA256 `82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61`.
- P0-A, N2 and vorticity weight were unchanged. No AoA10, locked-final/private, full-data, SPS selection, Codabench, interpolation, or follow-up experiment.
- Required tests: PASS, 32 passed locally and on the remote environment. `py_compile`: PASS. `git diff --check`: PASS. The local test run emitted one environment NumPy/PyTorch warning; all tests passed.
- GPU preflight: PASS at batch 8; 41,317 / 491 windows; finite prediction, loss and nonzero finite gradient; historical two-window BN forward reproduced; smoke state restored before training. Preflight peak reserved: 10.25 GiB.
- Formal process peak CUDA reserved: **10.3301 GiB**, below the 12 GiB hard cap. No OOM. Existing spatial-phase training was not stopped, paused, or modified; it completed on its own during this run.
- Shared-GPU runtime is recorded for provenance only and is invalid for performance comparison.

## Direct-CNO evolution

| Update | Rel-L2 | TKE | MVPE | Point score | F18 Rel | F19 Rel | F20 Rel | F18→F20 growth |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.714846 | 8.191961 | 0.885932 | 54.1980 | 0.611895 | 0.559495 | 0.585662 | -4.29% |
| 2,500 | 0.142453 | 0.563875 | 0.109075 | 88.7287 | 0.143215 | 0.166571 | 0.196040 | 36.89% |
| 5,000 | 0.130509 | 0.528300 | 0.115367 | 89.1750 | 0.134309 | 0.157566 | 0.192150 | 43.07% |
| 7,500 | 0.121219 | 0.527317 | 0.096141 | 89.6114 | 0.125025 | 0.142270 | 0.183280 | 46.60% |

## Direct vs frozen MF-CNO @ 7,500

The MF reference is frozen from `20260925_strong_backbone_tail_origin_run1/checkpoint_curve.csv`; it was not rerun.

| Metric | Direct-CNO | MF-CNO | Direct vs MF |
|---|---:|---:|---:|
| Rel-L2 | 0.12121927 | 0.12161895 | -0.329% |
| TKE | 0.52731705 | 0.50564378 | +4.286% |
| MVPE | 0.09614098 | 0.09572569 | +0.434% |
| Point score | 89.6114 | 89.8399 | -0.228 points |
| F18 Rel | 0.12502476 | 0.12581090 | -0.625% |
| F19 Rel | 0.14227024 | 0.14629239 | -2.749% |
| F20 Rel | 0.18328033 | 0.19566910 | -6.331% |
| F18→F20 growth | 46.5952% | 55.5264% | -8.931 percentage points |

Direct-CNO retains a substantial tail rise (46.60% from F18 to F20). Relative to MF, its F19/F20 errors and growth are lower, while TKE is 4.29% higher and MVPE is 0.43% higher. Overall Rel-L2 is nearly unchanged. These are the comparison facts; no causal verdict or next experiment is selected here.

## Direct fluctuation anatomy @ 7,500

| Horizon | Fluctuation amplitude ratio | Fluctuation cosine |
|---:|---:|---:|
| F18 | 0.7642 | 0.1798 |
| F19 | 1.1542 | 0.0835 |
| F20 | 1.7044 | 0.0562 |

F20 fluctuation amplitude remains nonzero and is 1.70× target, with low cosine alignment. The output therefore has not collapsed to a zero-fluctuation prediction; its late fluctuation alignment remains weak.

## Evidence and provenance

- `run_config.json`: frozen configuration, split counts, official init SHA, and safety flags.
- `preflight.json`: batch-8 forward/loss/gradient smoke, restored state and peak reserved memory.
- `aggregate_metrics.csv` and `summary.json`: all four evaluation milestones.
- `eval_XXXXX/`: per-horizon, per-trajectory, trajectory×horizon, mean-field and fluctuation diagnostics.
- `runtime.json`: resource record; runtime explicitly invalid for scientific comparison.
- `checkpoint_sha256.json`: official init and saved milestone checkpoint hashes. Checkpoints themselves are not included.
- `provenance.json`, `SHA256SUMS`, `DONE`.

Remote run directory: `/hy-tmp/realpde_runs/direct_cno_7500_control_20260925_run1`.

No checkpoint, H5 file, prediction array, or full console log is included in this archive. No next experiment was started.
