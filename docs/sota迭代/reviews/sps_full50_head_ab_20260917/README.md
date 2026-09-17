# SPS-FULL50-01 — frozen full predictor uncertainty-head A/B

Status: `REVIEW_REQUIRED`

This is a single-variable A/B experiment. Both arms use the frozen full
SOTA-V2 point predictor `@53582`; only uncertainty-head training scope differs.
The candidate is trained on the frozen 50 Train split, while the baseline is
the existing all-82 full-specific teammate35 head from the previous run.

## Result

| arm | uncertainty training scope | SPS | coverage | mean UV width | floor | mult |
|---|---|---:|---:|---:|---:|---:|
| Baseline | 82 released trajectories / 3383 windows | 44.37669830149426 | 0.8445542114362171 | 0.02329275757074356 | 0.0025 | 1.0 |
| Candidate | frozen 50 Train / 2052 windows | 44.38579352103458 | 0.8668990619060593 | 0.02529468946158886 | 0.0025 | 1.0 |

- `delta_sps = +0.009095219540320443`
- fixed updates: `1600`
- point prediction parity max abs: `0.0`
- bounds were fixed; no checkpoint search or calibration grid search was run.

## Paired 16-Dev result

- candidate wins: `9`
- ties: `0`
- candidate losses: `7`
- median delta SPS: `0.02400265965285797`
- mean delta SPS: `0.013918108660559758`
- min delta SPS: `-0.4534624087035226`
- max delta SPS: `0.4490741401167284`

## Frozen provenance

- execution commit: `b4acdf2297a931b6f85faccc44bb33cb189747b3`
- Sol required commit: `ee62a3e9bf9278581768033bcc56271f3a8b0736`
- manifest SHA256: `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`
- full point checkpoint `@53582` SHA256: `f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce`
- official scorer SHA256: `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`
- baseline full82 head SHA256: `622c521d8cda12d86e625962979753bc221c3c15eca663b0851ef6c4936671e2`
- candidate head SHA256: `a66a8db871db515a0a64a985128099ef26581907d8c1a1e5e72558e8f4e97d86`

The baseline metadata was verified as `head_scope=full_specific_teammate35`,
`selected_updates=1600`, and backbone SHA matching full `@53582`.

## Evidence and remote artifacts

Tracked lightweight evidence:

- `summary.json`
- `head_training_summary.json`
- `per_trajectory_paired.csv`
- `per_trajectory_paired.json`

Remote-only artifacts (not copied to Git):

- candidate head: `/home/chyfuture/realpde_runs/sps_full50_head_ab_20260917_retry1/run/teammate35_full53582_train50_head.pth`
- raw runner log: `/home/chyfuture/realpde_runs/sps_full50_head_ab_20260917_retry1/runner.log`
- isolated execution source: `/home/chyfuture/realpde_runs/sps_full50_head_ab_20260917/worktree_b4acdf2`

The first attempt stopped before training because the released-data directory
mount was one level too high. The retry used the existing data directory
`/home/chyfuture/RealPDE_data/p0ab_real_h5_20260830`; no algorithmic or
experimental parameter changed.

## Verification

- required SPS tests: `12 passed`
- full local suite: `194 passed, 2 failed, 1 skipped`
- the two full-suite failures were existing macOS DataLoader-worker/OMP shared-memory aborts in `test_dense_all_windows.py` and `test_random_phase_windows.py`; no SPS-FULL50 test failed.
- runtime: existing `realpde-pytorch-h5py:0831`, RTX 3090, CUDA-enabled Torch.
- locked-final/private: **NOT accessed**
- Codabench: **NOT accessed**
- package submission: **NOT performed**

This is **NOT point-model OOF validation**. The full `@53582` point predictor
was fitted using all released trajectories; only the uncertainty-head training
scope held out the 16 Dev trajectories.
