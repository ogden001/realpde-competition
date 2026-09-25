# REALPDE Strong Backbone F19/F20 Tail-Origin Diagnostic

Status: `REVIEW_REQUIRED`

This is a zero-training inference diagnostic on the frozen Strong Backbone Exp1 run. It evaluates only Clean Train51 / Seen-Dev12 and does not select or modify a model.

## Protocol and checks

- Tests: PASS — 27 passed across `test_strong_backbone_tail_origin.py`, `test_tail_fluctuation_coherence.py`, `test_tail_error_anatomy.py`, and `test_strong_backbone_tail_diagnostic.py`.
- `py_compile`: PASS for `tools/diagnose_strong_backbone_tail_origin.py`.
- `git diff --check`: PASS before evidence archiving.
- Optimizer steps: 0; training/fine-tuning: NO.
- Seen-Dev: 12 trajectories / 491 fixed windows.
- Official init: `sim_real_cno.pth`, SHA256 `82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61`.
- Execution source commit: `906e93096375215e6051df466b4717e7a74d5f52` (the required diagnostic script/test/protocol commit; run from an isolated remote source copy).
- All eight exact Exp1 milestone checkpoints passed filename/payload iteration and P0-A checks. Update 0 was rebuilt from official direct CNO through P0-A input expansion and MF five-channel projection initialization.
- AoA10: NO; locked-final/private: NO; Codabench: NO; full-data refit: NO; submission packaging: NO.
- The inference ran with `eval_batch_size=8`, `workers=2`, `--require-cuda`, concurrently with the existing spatial-phase training. That training remained running and was not signaled or modified.

## Checkpoint evolution

`Rel-L2`, `TKE`, and `MVPE` are whole-sequence metrics. Horizon error growth is `(F20 Rel / F18 Rel - 1) * 100`.

| Update | Rel-L2 | TKE | MVPE | F18 Rel | F19 Rel | F20 Rel | F18→F20 growth |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.448323 | 7.028714 | 0.494521 | 0.458839 | 0.474377 | 0.436713 | -4.82% |
| 7,500 | 0.121619 | 0.505644 | 0.095726 | 0.125811 | 0.146292 | 0.195669 | 55.53% |
| 15,000 | 0.114857 | 0.488585 | 0.092804 | 0.128553 | 0.157139 | 0.208997 | 62.58% |
| 20,000 | 0.110829 | 0.483561 | 0.086893 | 0.117850 | 0.140664 | 0.189000 | 60.37% |
| 25,000 | 0.110634 | 0.482264 | 0.083985 | 0.119739 | 0.148652 | 0.203370 | 69.84% |
| 30,000 | 0.106184 | 0.482600 | 0.079554 | 0.114273 | 0.140567 | 0.196621 | 72.06% |
| 31,000 | 0.102752 | 0.472804 | 0.074977 | 0.112157 | 0.133566 | 0.179610 | 60.14% |
| 32,500 | 0.101223 | 0.470762 | 0.072911 | 0.112266 | 0.133289 | 0.174276 | 55.24% |
| 35,000 | 0.101111 | 0.472205 | 0.075205 | 0.113164 | 0.133291 | 0.171836 | 51.85% |

The official-prediction cliff is absent at initialization, appears by 7,500 updates, and remains at the final checkpoint. Overall TKE falls from 7.029 at initialization to 0.472 at 35,000 while the final F18→F20 Rel gap is 56.67 percentage points larger than at initialization. This supports a training-associated origin for the cliff. The cliff peaks at update 30,000 rather than increasing monotonically through the final checkpoint.

## F19+F20 fluctuation energy share

Shares below are percentages of each component's Future20 fluctuation energy; the target share is 12.06% at every checkpoint.

| Update | Raw prediction | Centered prediction | GT |
|---:|---:|---:|---:|
| 0 | 8.45% | 27.93% | 12.06% |
| 7,500 | 11.83% | 36.13% | 12.06% |
| 15,000 | 11.74% | 34.33% | 12.06% |
| 20,000 | 11.33% | 29.14% | 12.06% |
| 25,000 | 11.49% | 31.79% | 12.06% |
| 30,000 | 11.29% | 32.46% | 12.06% |
| 31,000 | 10.79% | 26.03% | 12.06% |
| 32,500 | 10.83% | 26.55% | 12.06% |
| 35,000 | 10.60% | 26.45% | 12.06% |

Raw tail share does not rise monotonically and remains below GT after update 7,500. Centered tail share is substantially above GT throughout, but also does not grow from initialization to the final checkpoint. This is evidence of centered tail-energy concentration, but not a monotonic increase that by itself explains the evolving Rel cliff.

## Raw MF F20 anatomy

The raw and centered fluctuation metrics compare against the GT temporal fluctuation. Frame Rel compares pre-centering pseudo-prediction (`mean_field + raw_fluct`) with official reconstructed prediction (`mean_field + centered_fluct`).

| Update | Raw amp ratio | Centered amp ratio | Raw cosine | Centered cosine | Pre-center Rel | Reconstructed Rel |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9.4486 | 1.9241 | -0.0082 | -0.0009 | 1.0517 | 0.3134 |
| 7,500 | 3.1251 | 1.8897 | 0.0142 | 0.0518 | 0.3442 | 0.2173 |
| 32,500 | 2.5317 | 1.6522 | 0.0561 | 0.1225 | 0.2799 | 0.1985 |
| 35,000 | 2.5332 | 1.6446 | 0.0565 | 0.1217 | 0.2839 | 0.1968 |

The raw head has a large amplitude error at initialization and very weak F20 cosine alignment, and remains poorly aligned at the final checkpoint. Its raw F20 amplitude ratio decreases from 9.45 to 2.53 during training. Temporal centering improves F20 Rel at every listed update; at 35,000 it reduces the pre-center frame Rel by 30.66%. The zero-mean reconstruction therefore does not create or amplify the observed F20 Rel cliff in these comparisons.

## Flags and interpretation

- `cliff_present_at_init`: false.
- `cliff_grows_materially_during_training`: true (final growth +56.67 percentage points versus update 0).
- `raw_head_f20_amplitude_grows_materially`: false (9.45 at update 0, 2.53 at update 35,000).
- `mf_centering_materially_worsens_f20`: false (final reconstructed F20 Rel is 30.66% lower than pre-center Rel).

Assessment:

- Architecture / temporal-boundary origin: **REJECTED** for the official-prediction cliff; update 0 has no F18→F20 Rel cliff.
- Training-objective / energy-allocation origin: **SUPPORTED** for a training-associated cliff. Overall metrics improve sharply from initialization while the horizon cliff appears; the centered tail energy share is overrepresented but is not monotonically increasing.
- Raw MF head is primary source: **SUPPORTED** as the remaining source of weak tail fluctuation alignment/amplitude error; the raw F20 head remains inaccurate before centering. The extreme initialization amplitude is reduced by training and centering.
- MF zero-mean reconstruction is primary source: **REJECTED**; centering consistently improves the F20 frame Rel in the inspected checkpoints.

Most likely mechanism: the F19/F20 Rel cliff develops during optimization as the model learns poorly aligned late-horizon fluctuations; MF centering mitigates the raw-head error instead of producing the cliff.

## Evidence files

- `checkpoint_curve.csv`: all nine update-level whole metrics, F18/F19/F20 errors, growth, and tail shares.
- `evolution_summary.json`: threshold-based evolution flags.
- `checkpoint_sha256.json`: hashes for official init and eight exact milestones.
- `run_manifest.json`: data counts and safety/protocol flags.
- `update_XXXXX/`: per-horizon official prediction and raw MF component metrics for every update.
- `DONE`: diagnostic completed.

Remote output: `/hy-tmp/realpde_runs/strong_backbone_tail_origin_20260925_run1`.
