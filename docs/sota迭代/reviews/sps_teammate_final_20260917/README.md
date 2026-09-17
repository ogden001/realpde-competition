# SPS teammate final execution — 2026-09-17

Status: `REVIEW_REQUIRED`. Execution used the frozen runner at commit `7418b06f80ac2c0314884f3188daf98be277ffea`. The runner completed Phase A, continued through Phase B despite the Phase-A result, built the clean package, and passed clean-room smoke. Codabench was not accessed or submitted.

## Result

| Item | Result |
|---|---:|
| Phase-A baseline Dev SPS | `45.07008160038756` |
| Phase-A candidate Dev SPS | `45.30535126566061` |
| Delta | `+0.2352696652730515` |
| Phase-A Gate | `SPS_TEAMMATE_NO_GO` |
| Submission recommendation | `false` |
| Selected head iteration | `1600` |
| Best floor / multiplier | `0.0025 / 1.0` |
| Candidate mean UV width | `0.023694433271884918` |
| Width ratio vs baseline | `1.0047119572381529` |
| Coverage | `0.8601796897250408` |
| Point prediction parity | `0.0` |

The SPS gain did not reach the frozen `+1.5` requirement. Width and parity guards passed. Phase B nevertheless ran as explicitly required.

## Phase B and package

- Full point predictor: `@53582`, 82 released PIV trajectories, 3383 canonical stride-20 windows.
- Full-specific head: `/home/chyfuture/realpde_runs/sps_teammate_final_20260917/run/phase_b_full_specific_teammate35/teammate35_full_head.pth`
- Full-head SHA256: `622c521d8cda12d86e625962979753bc221c3c15eca663b0851ef6c4936671e2`
- Full-head updates: `1600` (copied from Phase-A selection; no full-data step selection).
- Reused calibration: `floor=0.0025`, `mult=1.0` (no full-data recalibration).
- Package: `/home/chyfuture/realpde_runs/sps_teammate_final_20260917/package_clean/submission.zip`
- ZIP bytes: `30260970` (`<256 MiB`)
- ZIP SHA256: `cc236a4926d36568ea9eb2d4c770f08b0a1c058ec3e85f82256d70d3df8db685`
- Package build status: `REVIEW_REQUIRED`, with `submission_recommended=false`.

## Clean-room smoke

- Report: `smoke_report.json`
- Status: `PASS`
- Point prediction max absolute diff: `0.0` at tolerance `1e-7`
- Deterministic max absolute diff: `0.0`
- Prediction/lower/upper: valid shape, `float32`, finite
- Pressure prediction: `0.0`
- Pressure interval width: `0.0`
- Prediction inside interval: passed
- Fallback: none
- Fixture: derived only from frozen Train trajectory `16500_0.h5`, using the pipeline-equivalent `::2` spatial downsample to the verifier's required `20×32×64` input; no locked-final/private trajectory was used.

## Frozen assets

| Asset | Path | SHA256 |
|---|---|---|
| 50/16 manifest | `/home/chyfuture/RealPDE_data/id_seed20260901.json` | `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347` |
| Validation checkpoint `@32500` | `/home/chyfuture/realpde_runs/sota_v2_integrated_50_16_20260916/run/checkpoints/model_update_32500.pth` | `6926722895611c38dee79cf16ce43b87c7de3d4a1305e9db431dadd4c1a7bf47` |
| Full checkpoint `@53582` | `/home/chyfuture/realpde_runs/sota_v2_full_20260916/run/checkpoints/model_update_53582.pth` | `f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce` |
| Official scorer | `/home/chyfuture/realpde_assets/realpde_t1_starting_kit_v9/scoring.py` | `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39` |

The runner and remote preflight verified the exact manifest hash shown above.

## Verification and bounded fixes

- Required tests: `pytest -q tests/test_sps_teammate_uncertainty.py tests/test_sps_teammate_package.py tests/test_sps_stride1_replica.py tests/test_sota_v2_adaptive.py tests/test_sota_v2_adaptive_package.py` → `26 passed`.
- Related package/dense-window tests → `11 passed`.
- Full suite: `pytest -q` → `193 passed, 1 skipped`.
- All test runs emitted one non-fatal Torch/NumPy initialization warning; no test failed.
- Remote runtime was the existing `realpde-pytorch-h5py:0831` container with Torch `2.2.2+cu121`, NumPy `1.26.4`, and RTX 3090 CUDA enabled.
- Bounded fixes only: set `PYTHONPATH=/repo/tools` for the runner's existing same-directory imports; mapped host output paths to container `/runs/...`; created the small downsampled Train smoke fixture required by the verifier's fixed `32×64` input contract. No feature, head, loss, split, Gate, calibration, backbone, inference logic, or budget was changed.
- Locked-final/private data: not accessed. Codabench: not accessed and not submitted.

## Evidence files

- `calibration_summary.json`
- `calibration_grid.csv`
- `checkpoint_evals.json`
- `head_training_summary.json`
- `full_head_summary.json`
- `package_build.json`
- `smoke_report.json`
