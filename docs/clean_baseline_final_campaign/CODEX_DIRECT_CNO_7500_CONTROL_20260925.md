# Codex Task — Direct-CNO 7.5k Matched Control

Status: `REVIEW_REQUIRED`

## Goal

Run one matched control to test whether the Strong Backbone F19/F20 cliff is caused primarily by the MF mean/fluctuation output parameterization.

Scientific variable only:

`MF-CNO output -> Direct-CNO output`.

Reference:

`Dense-All + P0-A + MF-CNO + N2 + Vorticity @ 7,500 updates`.

Control:

`Dense-All + P0-A + Direct-CNO + N2 + Vorticity @ 7,500 updates`.

Use the prepared runner:

`tools/train_clean_direct_cno_7500_control.py`.

## Required code

Current implementation commits include:

- `d4b58907188e79488b530de5e75b3069624f1a08`
- `2d19063cffeb29d5f6277eccb93769c1a7505a97`
- `e60dc134cbd5a42e0f21c8dc3f5c379f120fb017`
- `1b7f843a82586994e292f6c5f1b4ed308a2c7480`
- `c8ee91d3936713e7bbc293e8abc747d7a8dd42cd`
- `c5d4c8096775a90995c0bc8de059aed893150931`

Run from current `main` containing these commits. Do not reset unrelated newer commits.

## HARD CONSTRAINTS

Scientific protocol:

- Clean Train51 / Seen-Dev12 only.
- Train windows = 41,317 dense windows.
- Seen-Dev = 12 trajectories / 491 fixed windows.
- Official `sim_real_cno.pth` initialization.
- Official init SHA256:
  `82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61`
- P0-A unchanged.
- Direct-CNO output, exactly as implemented by the prepared runner.
- N2 unchanged.
- Vorticity loss unchanged.
- Seed = 41.
- Training batch = 8 exactly.
- AdamW semantics unchanged from the clean Strong Backbone trainer.
- LR = `1e-5`.
- Optimizer updates = exactly 7,500.
- No Stage-B.
- Milestone evaluation: 0 / 2,500 / 5,000 / 7,500.
- No AoA10.
- No locked-final/private.
- No Codabench.
- No full-data.
- No SPS selection.
- No checkpoint interpolation.
- No loss/feature/model/augmentation change.
- No hyperparameter sweep.
- No automatic follow-up experiment.

Resource hard constraint:

- this new experiment's own CUDA peak reserved memory must remain **<=12 GiB**;
- training batch=8 is scientific and must not be reduced;
- do not introduce gradient accumulation as a substitute;
- do not change FP32/BF16/FP16 precision;
- do not stop, pause, renice, or modify existing approved GPU jobs.

Historical matched-preflight detail is frozen:

- reproduce the historical one-forward BatchNorm side effect before update 1;
- the additional batch-8 memory/gradient smoke must restore model state before formal training.

If the exact protocol cannot run within the 12 GiB process-memory cap, do not alter science. Report `BLOCKED_RESOURCE`.

## SOFT / OPERATIONAL CONSTRAINTS

May adapt without blocking:

- GitHub vs verified Git bundle synchronization;
- worktree / checkout / temporary source-copy path;
- Python / venv path;
- CUDA physical index / `CUDA_VISIBLE_DEVICES`;
- DataLoader workers;
- evaluation batch size;
- pin-memory / prefetch;
- output directory / log / tmux name;
- temporary directories;
- unrelated pre-existing untracked files.

Existing GPU tasks are **not** a blocking condition.

This experiment is explicitly authorized to run concurrently on the same RTX 3090-class GPU while other approved training is active, provided:
- no OOM occurs;
- this experiment's own peak CUDA reserved memory stays <=12 GiB;
- total host RAM remains within the project limit.

GPU utilization = 100% is not a reason to wait.

Because the GPU is shared:
- runtime / latency / throughput are invalid for scientific comparison;
- mark runtime results as not comparable;
- point metrics remain valid.

## Preflight

Before the formal run:

1. sync code to current `main` containing the required implementation;
2. verify the official init SHA;
3. verify Train51 / Seen-Dev12 counts;
4. run tests:

```bash
python -m pytest -q   tests/test_clean_direct_cno_7500_control.py   tests/test_clean_baseline_final_campaign.py   tests/test_sota_v2_integrated.py   tests/test_tail_fluctuation_coherence.py

python -m py_compile tools/train_clean_direct_cno_7500_control.py
git diff --check
```

5. run a preflight-only GPU smoke with the same scientific batch=8;
6. confirm process peak CUDA reserved <=12 GiB.

A shared-GPU OOM is a resource condition, not permission to change batch or precision.

## Formal execution

Suggested output:

`/hy-tmp/realpde_runs/direct_cno_7500_control_20260925_run1`

Suggested command:

```bash
python -u -B tools/train_clean_direct_cno_7500_control.py   --manifest "/hy-tmp/realpde_runs/strong_backbone_clean_20260924_run1/train_dev_manifest.json"   --data-root "/hy-tmp/realpde_data/train_real"   --init-checkpoint "/hy-tmp/realpde_assets/colleague-80pt/20260921/extracted/handoff_colleague_80pt_archive_20260921/checkpoints/sim_real_cno.pth"   --kit-root "/hy-tmp/realpde_t1_kit_v9/realpde_t1_starting_kit_v9"   --out-dir "/hy-tmp/realpde_runs/direct_cno_7500_control_20260925_run1"   --workers 2   --eval-batch-size 8   --require-cuda
```

Path differences may be adapted only if the exact assets are verified by SHA / manifest.

## Frozen MF @7,500 reference

From:

`docs/clean_baseline_final_campaign/results/20260925_strong_backbone_tail_origin_run1/checkpoint_curve.csv`

Reference values:

- Rel-L2 = `0.12161894887685776`
- TKE = `0.5056437849998474`
- MVPE = `0.09572568535804749`
- F18 Rel = `0.12581089650476363`
- F19 Rel = `0.14629238745939646`
- F20 Rel = `0.1956690987404701`
- F18->F20 growth = `55.52635278539756%`

Do not rerun MF for this control.

## Required evidence

At 0 / 2,500 / 5,000 / 7,500 preserve:

- overall Rel-L2 / TKE / MVPE;
- point score;
- by-horizon F1..F20;
- by-trajectory;
- trajectory × horizon;
- mean-field diagnostics;
- fluctuation diagnostics by horizon;
- F18 / F19 / F20 tail metrics.

At 7,500 report Direct vs frozen MF reference:

- Rel delta %;
- TKE delta %;
- MVPE delta %;
- F18 delta %;
- F19 delta %;
- F20 delta %;
- F18->F20 growth difference in percentage points;
- Direct F18/F19/F20 fluctuation amplitude ratio and cosine.

Do not use shared-GPU runtime as comparison evidence.

## Interpretation boundary

Codex may summarize facts but does not choose the next experiment.

The causal question is:

- if Direct @7,500 has a much flatter tail while maintaining meaningful predictions, MF representation/training dynamics are implicated;
- if Direct @7,500 still has a similar cliff, the cause is upstream of MF, likely P0-A / N2 / vorticity or their interaction.

Final interpretation belongs to Sol.

## Delivery

Archive lightweight evidence under:

`docs/clean_baseline_final_campaign/results/<run_name>/`

Include:
- README;
- run_config / preflight / summary / runtime;
- aggregate metrics;
- milestone CSV / JSON diagnostics;
- SHA256 / provenance.

Do not commit:
- H5;
- checkpoints;
- large prediction arrays;
- full console logs.

Commit + push results to `origin/main`, verify remote contains the result commit, then STOP.

Do not start another experiment.
