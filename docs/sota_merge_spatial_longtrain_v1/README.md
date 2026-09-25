# SOTA Merge Spatial Long-Train V1

Status: `REVIEW_REQUIRED`

This runner implements the long-training backbone plan for both the clean 51/12
development protocol and the all-82 released-trajectory full refit.

## Recipe

Stage A is open-ended representation learning. Every legal stride-1 temporal
window is expanded into four real 2x spatial views: P00, P01, P10 and P11.
The four phases are equally represented. Past20 and Future20 always use the same
phase. The backbone keeps the validated Strong Backbone recipe: P0-A, MF-CNO,
N2 loss and vorticity supervision, FP32, effective batch 8, LR 1e-5.

Stage B is a short P00-only alignment tail forked from any immutable Stage-A
checkpoint. It adds the validated extra Rel term at LR 3e-6 and carries AdamW
state by default. Stage A may continue running while a Stage-B fork is evaluated.

## Clean Stage A

```bash
python -u tools/train_sota_merge_backbone.py stage-a \
  --scope clean \
  --data-root /path/to/train_real \
  --manifest configs/clean_baseline_v1_split.json \
  --init-checkpoint /path/to/sim_real_cno.pth \
  --kit-root /path/to/realpde_t1_starting_kit_v9 \
  --out-dir /runs/sota_merge_clean_stage_a \
  --max-updates 30000 \
  --eval-interval 1000 \
  --checkpoint-interval 1000 \
  --preload-to-ram \
  --require-cuda
```

Continue the same Stage A later:

```bash
python -u tools/train_sota_merge_backbone.py stage-a \
  --scope clean \
  --data-root /path/to/train_real \
  --manifest configs/clean_baseline_v1_split.json \
  --init-checkpoint /path/to/sim_real_cno.pth \
  --kit-root /path/to/realpde_t1_starting_kit_v9 \
  --out-dir /runs/sota_merge_clean_stage_a \
  --resume-checkpoint /runs/sota_merge_clean_stage_a/checkpoints/model_latest.pth \
  --max-updates 60000 \
  --eval-interval 1000 \
  --checkpoint-interval 1000 \
  --preload-to-ram \
  --require-cuda
```

`--max-updates` is the total update to reach, not an additional count.

## Clean Stage B fork

```bash
python -u tools/train_sota_merge_backbone.py stage-b \
  --scope clean \
  --data-root /path/to/train_real \
  --manifest configs/clean_baseline_v1_split.json \
  --init-checkpoint /path/to/sim_real_cno.pth \
  --kit-root /path/to/realpde_t1_starting_kit_v9 \
  --backbone-checkpoint /runs/sota_merge_clean_stage_a/checkpoints/model_update_020000.pth \
  --out-dir /runs/sota_merge_clean_stage_b_from_20k \
  --max-updates 6000 \
  --eval-interval 500 \
  --checkpoint-interval 500 \
  --require-cuda
```

## Full Stage A

Full mode uses exactly all 82 released H5 trajectories under `--data-root` and
forbids a manifest. It has no Dev checkpoint selection.

```bash
python -u tools/train_sota_merge_backbone.py stage-a \
  --scope full \
  --data-root /path/to/train_real \
  --init-checkpoint /path/to/sim_real_cno.pth \
  --kit-root /path/to/realpde_t1_starting_kit_v9 \
  --out-dir /runs/sota_merge_full_stage_a \
  --max-updates 50000 \
  --checkpoint-interval 1000 \
  --preload-to-ram \
  --require-cuda
```

Map clean updates to equal full-data epoch exposure:

```bash
python tools/train_sota_merge_backbone.py map-updates --stage A --clean-updates 30000
python tools/train_sota_merge_backbone.py map-updates --stage B --clean-updates 3000
```

## Hard invariants

- official sim_real checkpoint SHA
- clean split exactly 51 train / 12 Seen-Dev
- clean dense windows exactly 41,317
- full scope exactly 82 released trajectories / 66,755 dense windows
- effective batch exactly 8
- Stage A exhaustive P00/P01/P10/P11 with equal phase counts
- same phase for Past20 and Future20
- Stage B P00-only
- full mode has no Dev selection
- fresh Direct -> MF initialization parity
- no explicit locked-final/private path
- no Codabench action

Residual correction and residual-aware SPS remain the already validated frozen
recipes and are not changed by this backbone experiment code.
