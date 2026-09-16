# SPS stride=1 replica — 2026-09-17

Status: **`COMPLETED / SPS_REPLICA_NO_GO / REVIEW_REQUIRED`**

## Scope and execution

- Execution commit: `01b137f13c77d14f9f2262eaf41e83f7a8ca10fe`.
- Required commit `7672318209c8a62386ead6ab219310e5e3abcf81` is an ancestor; local `HEAD == origin/main` and `git push --dry-run origin HEAD:main` passed before launch.
- Runner: `tools/realpde_sps_stride1_replica.py`.
- Remote run root: `/home/chyfuture/realpde_runs/sps_stride1_replica_20260916/run`.
- Environment: `gpu`, Docker image `realpde-pytorch-h5py:0831`, NVIDIA RTX 3090, CUDA-enabled Torch 2.2.2+cu121.
- Data: 50 Train / 16 Dev frozen manifest, `40488` dense-all train windows and `659` Dev windows. No locked-final/private Future20 or Codabench access.
- Validation backbone: iteration `32500`, SHA-256 `6926722895611c38dee79cf16ce43b87c7de3d4a1305e9db431dadd4c1a7bf47`.
- Official scorer SHA-256: `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`.

The only bounded production-code fix was an environment-only fallback in `tools/realpde_p0_data.py` for the local PyTorch/NumPy `Numpy is not available` bridge error. A separate test-only compatibility edit uses direct NumPy arrays in the checkpoint-soup fixture. CUDA execution retained the original `torch.from_numpy` path and the frozen experiment semantics were unchanged.

## Frozen Phase-1 recipe

- Same SOTA-V2 validation backbone and adaptive head architecture `in_channels=15, hidden=32, blocks=2`.
- Gaussian NLL, `1400` updates, frozen seed/optimizer schedule, and the fixed 28-row floor × multiplier calibration grid.
- Sole variable: head-training windows changed from canonical fixed stride 20 to all legal `dense_all` stride-1 windows.
- Phase-1 head SHA-256: `3050700cb178d3fd752ddb72a2ce8564c92f6362eb07bd9d4fcd6a88e10a8a8c`.

## Gate result

| quantity | frozen baseline | stride-1 candidate |
|---|---:|---:|
| Dev SPS | 45.07008160038756 | 45.06056722847220 |
| SPS delta | — | -0.00951437191536 |
| mean UV width | 0.02358330972492695 | 0.02367074787616730 |
| width ratio | 1.0000000000 | 1.0037076285 |
| coverage | 0.8558713772975425 | 0.8573753902991296 |

The width guard passed, but the required SPS gain of `+1.5` did not. The registered result is therefore **`SPS_REPLICA_NO_GO`**. Prediction parity against the replayed frozen backbone was `0.0`; raw replay metrics remained Rel-L2 `0.09993461519479752`, TKE `0.4692927300930023`, and MVPE `0.07577798515558243`.

Phase 2 was skipped immediately by the runner (`SKIPPED_PHASE1_NO_GO`). No full-specific head, package build, or clean-room smoke was produced; `package_clean/` is absent.

Additional diagnostics: sigma/error Pearson `0.6768027687200431`, Spearman `0.6347138619136505`, sampled points `199946`.

## Verification

- `pytest -q tests/test_sps_stride1_replica.py tests/test_dense_all_windows.py tests/test_adaptive_probe.py`: `21 passed` after the bounded local ABI compatibility fix.
- Related adaptive/package/fixture tests: `12 passed, 1 skipped`.
- Full local suite: `180 passed, 1 skipped`.
- `python -m py_compile` over the modified loader, runner, builder, and related tests: passed.
- The initial local Dense-All and checkpoint-soup failures were reproduced as the known Mac Torch/NumPy ABI error (`Numpy is not available`); the minimal compatibility edits resolved them without changing the remote CUDA path.

## Evidence files

- `calibration_summary.json`
- `calibration_grid.csv` / `calibration_grid.json`
- `by_horizon.csv`
- `by_channel.csv`
- `head_training_summary.json`
- `run_summary.json`

This is a local Dev-only offline result. It is evidence for Sol/ChatGPT review, not a leaderboard result and not an authorization to submit.
