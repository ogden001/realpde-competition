# Track 1 P0-A + N2 full-data model — Codabench result handoff

## Executive result

This is the official Codabench evaluation of the selected 15,300-update
full-data Track 1 model.  It is the first official score for this exact
P0-A + N2, all-82-trajectory refit; it is not a local proxy and it did not
use private-test information during training or package construction.

| Metric | Codabench score |
|---|---:|
| Rel-L2 | 93.023539 |
| TKE | 78.355520 |
| MVPE | 91.894417 |
| Time | 88.430528 |
| SPS | 11.431650 |
| **Final** | **71.153839** |

Relative to the previous best-known official submission
`submission_cno_tke1200_bounds_rel00.zip` (final `75.584550`), this model is
`4.430711` final-score points lower.  Its TKE score is materially higher
(+7.421195), while Rel-L2 and MVPE are close but slightly lower.  The dominant
visible regression is SPS (-16.348886), which coincides with the lower final
score.  Because Codabench does not disclose the final-score formula, this is
an evidence-based diagnosis rather than a claim of exact causality.

| Score | P0-A + N2 full 15,300 | Previous best CNO | Delta |
|---|---:|---:|---:|
| Rel-L2 | 93.023539 | 93.542062 | -0.518523 |
| TKE | 78.355520 | 70.934325 | +7.421195 |
| MVPE | 91.894417 | 92.167656 | -0.273239 |
| Time | 88.430528 | 87.236663 | +1.193865 |
| SPS | 11.431650 | 27.780536 | -16.348886 |
| **Final** | **71.153839** | **75.584550** | **-4.430711** |

## Selected model

### Network

- Base network: official Track 1 CNO `CNO3d`.
- Input/output: `(N, 20, 32, 64, 3)` to the same shape, channel order
  `[u, v, p]`.
- CNO configuration: `in_dim=20`, `out_dim=3`, `out_dim_mult=1`,
  `in_size=64`, `N_layers=3`.
- Initialization: official competition warm-start
  `sim_real_ft/sim_real_cno.pth`; its 3-channel input-lift weights are copied
  into the first three channels of the 20-channel lift and the 17 new-channel
  lift weights start at zero.  Thus the expanded model initially preserves the
  warm-start's raw-channel behavior.
- Inference uses one cached model instance, `model.eval()` and
  `torch.inference_mode()`.
- No Re, AoA, coordinates, mask, HDF5, metadata or network access is used at
  inference.  The model receives only the supplied 20-frame input tensor.
- The packaged wrapper outputs zero pressure, consistent with this Track 1
  PIV-oriented competition route.

### P0-A feature engineering

The raw `[u, v, p]` history is augmented to 20 causal channels with 17 features
computed only from the supplied 20-frame window:

1. speed magnitude;
2. `du/dx`, `du/dy`, `dv/dx`, `dv/dy`;
3. vorticity, absolute vorticity, and strain magnitude;
4. temporal `delta_u` and `delta_v`;
5. history-window `u`/`v` means and standard deviations;
6. demeaned `u` and `v` fluctuations;
7. history TKE proxy.

The spatial derivatives use the grid spacings stored in the selected checkpoint
feature configuration.  These features are deterministic and causal within the
provided history window.

### Loss: N2

Training applies the loss to the two measured velocity channels:

`L = MSE + 0.05 * TKE + 0.027514 * Relative-L2 + 0.009757 * MVPE`.

`TKE`, `Relative-L2`, and `MVPE` follow the runner's frozen implementations in
`tools/realpde_p0a_n2_full.py`.  They were designed to improve physical
velocity statistics alongside pointwise field error, not to optimize the
unobserved Codabench final composite directly.

## Evidence and training process

### Method-selection validation (before all-data refit)

The architecture/feature/loss decision used a fixed, trajectory-disjoint
50-train / 16-dev split, seed `20260901`, the same official warm-start, N2
loss, effective batch 8, and 4,100 optimizer updates.  The dev replay used the
official Track 1 v9 scorer raw errors.

| Raw dev error (lower is better) | Raw 3-channel N2 | P0-A + N2 | P0-A relative improvement |
|---|---:|---:|---:|
| Rel-L2 | 0.145684 | 0.141550 | 2.84% |
| TKE | 0.576014 | 0.546986 | 5.04% |
| MVPE | 0.122189 | 0.120323 | 1.53% |

This matched-budget result is the GO evidence for P0-A + N2: all three raw
dev errors improved.  Continuation to 10,300 updates improved the same dev
run further (Rel-L2 `0.123734`, TKE `0.515453`, MVPE `0.096836`), but no
validation checkpoint was used to claim that the later all-data checkpoint was
a known global optimum.

This is an `OFFICIAL_WARM_START` competition experiment, not clean offline
causal evidence: the official `sim_real_ft` initialization is real-PIV
fine-tuned and its relationship to the local 50/16 split is not disclosed.

### Full-data competition refit

After the selection decision, the exact method was refit on all 82 released
real PIV trajectories, with no dev/locked-final/private-test labels in this
stage:

- seed: `20260901`;
- windows: 3,383;
- default micro-batch/accumulation: `4 x 2` (effective batch 8);
- initial refit: 6,800 updates, about 16.08 data traversals;
- continuation milestones preserved: 6,800, 10,300, 12,481 and final 15,300;
- selected checkpoint: `model_last.pth`, iteration `15300`;
- checkpoint SHA256:
  `3ea4e4def03ae2f1d970975e4217358e1d762b88a69bdddfdf844d551baaa3e4`.

The initial full refit completed in 5,384 seconds with 5,065,201,152 bytes
(4.72 GiB) peak allocated CUDA memory.  A later high-memory batch-16 probe
used 18.46 GiB but was slower per optimizer update, so the deadline run kept
the effective-batch-8 protocol.

The all-data stage intentionally has no held-out validation set: it is a
competition refit after method selection.  Its training loss is a diagnostic,
not a checkpoint-selection metric.

## Submission and reproducibility

- Experiment ID: `T1-COMP-P0A-N2-FULL-S20260902-LAST15300`.
- Implementation commit used for submission construction: `7831297`.
- Submission archive SHA256:
  `f0cd4cb9edfa9b7389780906f47bc83f86fe6b82640172ffc2611bbee90706e2`.
- ZIP size: 29,704,242 bytes (28.3 MiB; below the 256 MiB limit).
- Package contents: root-level `submission.py`, selected inference-only model
  state, complete CNO runtime source tree, and vendored `einops`; no training
  data, optimizer state, other checkpoints, logs, caches or test artifacts.
- Clean-room package smoke (remote RTX 3090 official-compatible container):
  checkpoint load, `N=1` predict, output `(1,20,32,64,3)`, `float32`, finite
  output, deterministic replay, root-level entry, and size checks all passed.
  The recorded first-call end-to-end time was 0.422520 seconds and peak CUDA
  allocation 142,225,920 bytes.  It includes lazy model loading and is not a
  Codabench runtime guarantee.

The ZIP, checkpoint, data, logs and other generated artifacts remain outside
Git by design.  The repository contains the runner, package implementation,
hashes, experimental protocol and this result record.

## Interpretation for next analysis

1. P0-A + N2 is supported as a feature/loss improvement on the fixed local dev
   protocol and the official result strongly improves TKE over the prior best
   CNO submission.
2. It is not the current leaderboard best because SPS fell sharply.  Do not
   interpret the physical-metric gains alone as a submission win.
3. The next investigation should isolate the SPS mechanism under the official
   submission protocol: prediction interval construction/coverage, pressure
   handling, rollout safety behavior and any interaction between the new
   velocity dynamics and the evaluator's safety criterion.  Avoid using
   private-test feedback as a repeated tuning signal.
4. The former `tke1200_bounds_rel00` CNO remains the best official result
   recorded in this repository until a new submission beats final `75.584550`.
