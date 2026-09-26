# Full82 Corrector Training Review Data

Status: `RUNNING_HUMAN_IN_THE_LOOP / REVIEW_REQUIRED`

Snapshot time: 2026-09-26 19:34:53 +08:00. This is a read-only operational snapshot for Sol review, not a model-selection or evaluation result. Training was not changed or interrupted.

## Run and recipe

- Run root: `/hy-tmp/realpde/runs/sota_merge_full82_corrector_20260926`
- Process: PID `23432`; alive; elapsed `01:53:03`; CPU `43.2%`
- Launch script: `/hy-tmp/realpde/runs/sota_merge_full82_corrector_20260926/launch_full82_corrector.py`
- Trainer: `/hy-tmp/realpde/projects/sota_merge_clean_stage_a_386920e/code/tools/realpde_residual_corrector_longtrain.py`
- Backbone: Full Stage-A `model_update_090000.pth`; frozen; SHA256 `94d2d2d37110d76fd5104c7e6498b380b022e20f1f1cb621e17ba67165040e6c`
- Data: all 82 H5; configured/validated Dense-All window count `66,755`; input 20, output 20; spatial subsample 2; batch 8
- Optimizer/schedule: AdamW, LR `1e-4`, weight decay `1e-5`, cosine `T_max=50,000`, max updates `50,000`
- Milestones: 5k, 10k, 15k, 20k, 25k, 30k, 33k, 35k, 40k, 45k, 49.5k, 50k
- Automatic evaluation, Joint, and SPS are disabled.

## Progress and recorded loss

At snapshot: update `20,700 / 50,000` (`41.4%`), epoch 2, current logged loss `0.1775463`, LR `6.33451e-5`. Average observed speed from process elapsed time was approximately `3.05 updates/s`; this is a rough extrapolation, not a runtime guarantee.

Milestone records are stochastic training-batch losses, not Dev metrics:

| Update | Epoch | Loss | LR |
|---:|---:|---:|---:|
| 5,000 | 0 | 0.183261 | 9.75528e-5 |
| 10,000 | 1 | 0.190880 | 9.04508e-5 |
| 15,000 | 1 | 0.174979 | 7.93893e-5 |
| 20,000 | 2 | 0.183756 | 6.54508e-5 |

The JSONL log had 20,346 valid records at the read-only numeric scan; no non-finite numeric values were found through update 20,346. The current record at 20,700 also had finite values. No evaluation was run for this report.

## Saved checkpoints

Present in the run root, each approximately 6.0 MB:

- `corrector_update_5000.pth` (2026-09-26 18:09)
- `corrector_update_10000.pth` (2026-09-26 18:36)
- `corrector_update_15000.pth` (2026-09-26 19:04)
- `corrector_update_20000.pth` (2026-09-26 19:31)

At snapshot update 20,700, the latest saved milestone was 20,000.

## Sampling semantics review finding

Relevant code paths:

- `tools/realpde_sota_v2_integrated.py`: `dense_loader()` constructs `H5WindowDataset(... stride=20, sub_sample=2, window_mode="dense_all")`, then uses `DenseAllWindowSampler` and `DataLoader(... batch_size=8, drop_last=True)`.
- `tools/realpde_p0_data.py`: in `dense_all` mode, dataset refs are built for every legal start with `range(0, T - 20 - 20 + 1)`, i.e. start stride 1; the configured stride 20 does not thin these starts. The sampler begins with all dataset indices and shuffles globally using `SeedSequence([seed, epoch])`; run seed is `20260901`.
- Spatial reads use `[..., ::sub_sample, ::sub_sample]` with `sub_sample=2`, i.e. P00. No random spatial phase or four-phase augmentation is used in this path.
- **Mismatch for Sol review:** `drop_last=True` means 66,755 references are not all consumed in each epoch. With batch 8, the loader emits 8,344 full batches (66,752 samples) and drops 3 references at the end of each shuffled epoch. Epoch transition log positions (updates 1, 8,345, 16,689) are consistent with this. No runtime change was made.

Thus legal temporal starts are Dense-All/stride-1 and there is no explicit temporal subsampling, but the stronger requirement “all windows retained each epoch” is not met exactly because of the 3-sample incomplete batch drop. Spatial semantics match P00-only/subsample-2.

## GPU and disk observations

- At 19:34:53, RTX 4090 snapshot: utilization `0%`, memory `2,008 / 24,564 MiB`, temperature `58 C`.
- Earlier 60-second sampling (18:25:54–18:26:55): SM utilization average `29.7%`, min `0%`, max `99%`; 28 of 60 samples were zero. This indicates intermittent utilization; it does not identify the bottleneck.
- `/hy-tmp`: 21 GB available of 50 GB.

## Historical spatial-phase gate

Existing archived gate outcome is `NO_GO` at `docs/spatial_phase_gate/results/20260925_run1/SPATIAL_PHASE_GATE_REVIEW.md`; archived in commit `1fee054ff022576fb634e08e7445af459583a1d6` (`Archive spatial phase gate evidence`, 2026-09-25). No gate was rerun.

## Repository state at report creation

On the GPU host, branch `main`, base HEAD `386920e532e54ba1cbf42c93eb33c5ca165c0cd6`. The working tree already had modifications in `tools/realpde_residual_corrector_longtrain.py` and `tools/train_sota_merge_backbone.py`; the report commit stages only this Markdown file and does not include those modifications.
