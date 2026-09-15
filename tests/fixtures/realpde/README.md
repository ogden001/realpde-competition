# RealPDE real-data test fixtures

These fixtures are small, stable samples of the official RealPDE Competition
Train release. They are intended for automated tests and local interface
validation only—not for training, hyperparameter tuning, checkpoint selection,
or model selection.

## Provenance and license

- Split: **Train only**. No Dev Future, Final, or locked-final data was used.
- Source archive: `data/train_real.tar.gz` from the official competition data
  release; the frozen Train manifest is
  `artifacts/loss_optimization_v9_20260901_run1/evidence/manifests/id_seed20260901.json`.
- Selected source trajectories: `3750_10.h5`, `20325_0.h5`, and
  `16500_0.h5`. All three are listed in that manifest's `train` split. The
  known duplicate `7575_0.h5` was not used.
- The official release data card states that the data are released for
  non-commercial research and competition use under **CC BY-NC 4.0**. This
  private repository is used for that non-commercial research/competition
  purpose. Keep this attribution and license context if the fixtures are
  copied elsewhere.
- Official references: [competition rules](https://realpdecompetition.github.io/)
  and [official competition data card](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data).
- Hidden validation/test data were not accessed or redistributed.

## Format

The native real PIV fields are `64×128`; the competition's official scored
resolution is `32×64`. Each fixture uses the official `::2, ::2` spatial
subsample, keeps the original numeric values without normalization, and
contains these top-level HDF5 datasets:

`u`, `v`, `p`, `x`, `y`, `re`, `aoa`

Real PIV has no measured pressure channel, so `p` is stored as an exactly-zero
array with the same shape as `u` and `v`. Array datasets use HDF5 gzip
compression.

| Fixture | Source | Frames | Resolution | Re | AoA | Fixed windows | Dense-All windows |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `train_sample_a.h5` | `3750_10.h5` | 80 | 32×64 | 3750 | 10° | 3 | 41 |
| `train_sample_b.h5` | `20325_0.h5` | 80 | 32×64 | 20369 | 0° | 3 | 41 |

Each committed fixture has a same-stem JSON summary containing the retained
frame range, shapes, Re/AoA, u/v min/max/mean/std, `dx`/`dy`, fixed and
Dense-All window counts, and the fixture SHA-256.

## 868-frame candidate

The `16500_0.h5` Train trajectory was checked as a possible
`train_full_868.h5` fixture. The gzip-compressed 32×64 candidate measured
21,564,226 bytes (>20 MB), so it is intentionally **not committed**. Its
remote source path is:

`gpu:/home/chyfuture/RealPDE_data/p0ab_real_h5_20260830/16500_0.h5`

If a future size policy permits a full fixture, its Dense-All protocol must
produce exactly 829 legal windows for `Tin=20`, `Tout=20`, `stride=1`.
