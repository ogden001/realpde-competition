# REALPDE JOINT TRAINING V1

Status: `REVIEW_REQUIRED`

## Execution

- Execution commit: `cbf25c73400e753de5d3d9b701118f9b01954b3e` (`main` at launch)
- Host/GPU: `I2b5847212d00701227` / `NVIDIA GeForce RTX 4090`
- Environment: Python `3.11.12`, Torch `2.9.1+cu128`, CUDA `12.8`, driver `570.211.01`
- Run root: `/hy-tmp/realpde_runs/realpde_joint_training_v1_4090_20260926_run2`
- Start time (UTC): `2026-09-26T01:45:49Z`
- Formal runtime: 4127.3 s (68.8 min)
- Completion: 6000 joint updates; state `DONE`, status `REVIEW_REQUIRED`; trainer exited after writing final Seen-Dev evaluation and summary.
- Formal optimizer steps: 6000; smoke optimizer steps: 2. Smoke used a temporary in-memory 2-step limit; no scientific source was changed.
- Launch parameters: batch size 8 per branch; eval batch size 8; 4 workers; prefetch factor 2; recovery every 500 updates; milestones 1k–6k.
- First launch attempt was rejected by the runner's non-empty output directory guard before update 1; a fresh run directory was used for the successful run.

## Initialization and provenance

- A@57k: `/hy-tmp/realpde_runs/sota_merge_clean_stage_a_4090_restore_20260926/checkpoints/model_update_057000.pth`; SHA256 `1bdde38d902c2bec516570cadb8703797457c7db9c3fda67eb9a911021d1b873`.
- Corrector warm start: `/hy-tmp/realpde_assets/joint_training_v1/corrector_tmr02_update_30000.pth`; SHA256 `1f72b329d9a160b206a6e9f4fce5e780022b3bac7a9ce4bdf5b8c9efa30b4506`; 30000 source updates; mode `warm_start_cross_backbone`.
- Corrector architecture: `in_channels=42, hidden=64, blocks=2, max_delta=0.04`.
- Historical corrector backbone SHA256: `6926722895611c38dee79cf16ce43b87c7de3d4a1305e9db431dadd4c1a7bf47`.
- Split manifest SHA256: `1a0935dd7fbc4860b0d2ca5d49e410dfe03191cf8177d9f8ced336cd69736e2a`.
- Joint checkpoint pairs exist at 1k, 2k, 3k, 4k, 5k, 6k. Update 0 is the initial A@57k + warm-start corrector pair. See [`checkpoint_inventory.csv`](./checkpoint_inventory.csv); no checkpoint files are included in this evidence commit.

## Evaluation protocol and completion

- Seen-Dev: 12 trajectories, 491 windows at each update.
- AoA10 diagnostic: the 18 trajectory IDs in the clean manifest's holdout list, 756 windows at each update. AoA10 was used only for the requested generalization diagnostic.
- Both splits were evaluated at 0, 1k, 2k, 3k, 4k, 5k, 6k with the existing `realpde_sota_merge_joint_runtime.evaluate_pair` path, existing metric/scorer implementation, and both base and final corrected predictions. No new evaluator was introduced.
- Sequentially loaded/evaluated each milestone pair; 0k used the original source backbone and corrector warm-start. Evaluation optimizer steps: 0.
- All seven evaluation points have `metrics.json`, base/final by-horizon summaries, and aggregate CSV rows.

## Seen-Dev curve

Runner-selected Seen-Dev point checkpoint: update 4,000, corrected point 91.270851.

| Update | Base Rel-L2 | Base TKE | Base MVPE | Base point | Corrected Rel-L2 | Corrected TKE | Corrected MVPE | Corrected point |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.100464 | 0.465893 | 0.073000 | 90.934038 | 0.091394 | 0.465893 | 0.071889 | 91.088923 |
| 1,000 | 0.098131 | 0.462884 | 0.070472 | 91.041630 | 0.089242 | 0.462884 | 0.068002 | 91.215287 |
| 2,000 | 0.097761 | 0.463023 | 0.069575 | 91.059667 | 0.088702 | 0.463023 | 0.066924 | 91.238804 |
| 3,000 | 0.097921 | 0.462455 | 0.070398 | 91.050665 | 0.088584 | 0.462455 | 0.067122 | 91.243753 |
| 4,000 | 0.097389 | 0.462246 | 0.068844 | 91.085203 | 0.088298 | 0.462246 | 0.065814 | 91.270851 |
| 5,000 | 0.097360 | 0.462718 | 0.069040 | 91.077414 | 0.088211 | 0.462718 | 0.065816 | 91.266951 |
| 6,000 | 0.097416 | 0.462568 | 0.068926 | 91.080006 | 0.088225 | 0.462568 | 0.065783 | 91.268919 |


## AoA10 diagnostic curve

Highest corrected point among the diagnostic rows: update 2,000, corrected point 90.456222. This is a diagnostic fact only and does not select a model.

| Update | Base Rel-L2 | Base TKE | Base MVPE | Base point | Corrected Rel-L2 | Corrected TKE | Corrected MVPE | Corrected point |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.102848 | 0.497881 | 0.099393 | 90.147542 | 0.094484 | 0.497881 | 0.097162 | 90.307920 |
| 1,000 | 0.101531 | 0.492873 | 0.098445 | 90.235365 | 0.093030 | 0.492873 | 0.094895 | 90.418052 |
| 2,000 | 0.101358 | 0.491419 | 0.097765 | 90.263876 | 0.092639 | 0.491419 | 0.093801 | 90.456222 |
| 3,000 | 0.101521 | 0.496183 | 0.099024 | 90.191290 | 0.092559 | 0.496183 | 0.094554 | 90.394940 |
| 4,000 | 0.101151 | 0.494647 | 0.098326 | 90.223881 | 0.092476 | 0.494647 | 0.094166 | 90.418527 |
| 5,000 | 0.100971 | 0.494384 | 0.098193 | 90.231426 | 0.092256 | 0.494384 | 0.093962 | 90.427804 |
| 6,000 | 0.101087 | 0.494160 | 0.098098 | 90.233513 | 0.092270 | 0.494160 | 0.093804 | 90.432393 |


## Candidate review facts

- Runner Seen-Dev selection: update 4,000.
- Maximum AoA10 corrected point in the diagnostic curve: update 2,000.
- No automatic “generalization-safe” designation was assigned. Updates 2,000 and 4,000 are the separate AoA10 and Seen-Dev point leaders for Sol's review; all other milestone rows remain in the tables. AoA10 did not alter the runner's Seen-Dev selection.

## Validation record

- Remote `py_compile`: passed for both joint entry/runtime modules.
- Remote focused pytest group: 32 passed, 1 warning.
- Local focused pytest group: 31 passed, 1 failed because the local Torch 2.6.0 / NumPy 1.26.4 environment raised `RuntimeError: Numpy is not available`; the same focused group passed in the remote training environment.
- Smoke: passed through two updates and resume; update 1 loss components and backbone/corrector gradient norms were finite and nonzero; recovery save/resume and Seen-Dev evaluation worked.
- Formal training log: no NaN/Inf/fatal/Traceback/error match observed at completion.

## Access boundaries

- Locked-final accessed: **NO**
- Private accessed: **NO**
- Codabench accessed: **NO**
- Subsequent SPS/full/final experiments: **not started**

## Evidence files

- [`aggregate_metrics.csv`](./aggregate_metrics.csv): Seen-Dev base/final metrics, 0–6k.
- [`aoa10_aggregate_metrics.csv`](./aoa10_aggregate_metrics.csv): AoA10 base/final metrics, 0–6k.
- [`checkpoint_inventory.csv`](./checkpoint_inventory.csv): initial and milestone checkpoint paths, sizes, metadata, and SHA256.
- `eval_seen_dev_*/` and `eval_aoa10_*/`: base/final by-horizon summaries and per-update metrics JSON.
- [`run_config.json`](./run_config.json), [`launch_metadata.json`](./launch_metadata.json), [`summary.json`](./summary.json), [`status.json`](./status.json), [`aoa10_eval_metadata.json`](./aoa10_eval_metadata.json): configuration and run/evaluation provenance.
