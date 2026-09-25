# SPATIAL_PHASE_BACKBONE_SCREEN_CODEX_EXECUTION

Status: `IMPLEMENTED_BY_SOL / EXECUTE_ONLY / REVIEW_REQUIRED`

Codex executes the code already written on `main`. Codex is not the scientific code author for this task.

## Hard constraints

These are mandatory. Violation means `BLOCKED`.

- Use exactly `tools/run_spatial_phase_backbone_screen.py`.
- The candidate must reuse `tools/colleague_80pt/train_clean_baseline_cno.py`.
- Do not create a new training implementation.
- Do not rerun the historical Control.
- Start from the approved official `sim_real_cno.pth` whose SHA256 is already frozen by Clean Baseline V1.
- Use the canonical Train51 / Seen-Dev12 split.
- Candidate updates = `8723`.
- Eval interval = `1000`.
- Batch size = `8`.
- LR = `1e-4`.
- Seed = `41`.
- Spatial phase mix = `0.5`.
- Spatial phase seed = `20260925`.
- Seen-Dev remains P00 only.
- Model architecture, Stage-1 loss, optimizer and cosine scheduler remain unchanged.
- Do not modify temporal window sampling.
- Do not add AoA augmentation, velocity rotation, loss tuning, TKE projection, Residual, Strong-Backbone Stage A/B changes, or SPS work.
- Do not access AoA10 Holdout, locked-final/private, Codabench.
- Do not start full-data refit, package/submission work, another phase ratio, another seed, or automatic follow-up training.
- If source code or scientific tests fail, STOP and report `BLOCKED`. Do not patch the scientific code yourself.
- Do not use `git add .` or `git add -A`.

A GO from review does not authorize any next experiment.

## Soft constraints

These may be adjusted only for non-scientific execution reasons and must be reported.

- Preferred workers = 4. Workers may be reduced to 2 if the host has a DataLoader/process resource issue before scientific training is underway.
- CUDA device selection, absolute data/model paths, detached-session mechanism, log location, and process monitoring may be adjusted.
- Environment/dependency/path problems may be repaired if they do not alter code semantics, model, data, loss, sampling, optimizer, scheduler, batch, LR, seeds, update count or evaluation.
- Prefer detached execution so SSH/Codex disconnection cannot kill training.
- Monitor GPU memory, host RAM, disk and process liveness.
- If a process failure happens after optimizer updates have begun, do not silently restart with a changed setup. Preserve evidence and report `BLOCKED` for Sol review.

## Repository hygiene

First:

```bash
git checkout main
git fetch origin
git pull --rebase origin main
git status --porcelain=v1
git rev-parse HEAD
git rev-parse origin/main
git push --dry-run origin HEAD:main
```

Existing unrelated untracked files are allowed if every pre-existing entry is `?? <path>` and none overlaps:

```text
tools/run_spatial_phase_backbone_screen.py
tools/colleague_80pt/train_clean_baseline_cno.py
tests/test_spatial_phase_backbone_screen.py
tests/test_spatial_phase_gate.py
docs/spatial_phase_backbone_screen/
```

Record their exact list and leave them untouched. Tracked/staged changes or overlapping untracked files mean `BLOCKED`.

## Tests before optimizer updates

Run:

```bash
python -m pytest -q \
  tests/test_spatial_phase_backbone_screen.py \
  tests/test_spatial_phase_gate.py \
  tests/test_clean_baseline_v1.py

python -m py_compile \
  tools/run_spatial_phase_backbone_screen.py \
  tools/colleague_80pt/train_clean_baseline_cno.py \
  tools/colleague_80pt/realpde_h5_feature_adapter_train.py

git diff --check
```

All must pass before training.

## Approved assets

Use the same Clean Baseline assets already present on the GPU host.

Typical paths:

```bash
REAL_ROOT=/hy-tmp/realpde_data/train_real
SIM_PRETRAIN=<approved sim_real_cno.pth with frozen Clean Baseline SHA256>
KIT_ROOT=<same v9 model/starting-kit root used by Clean Baseline V1>
OUT=/hy-tmp/realpde_runs/spatial_phase_backbone_screen_20260925_run1
```

Trust the SHA check in preflight, not the filename alone.

## Preflight

```bash
python -u -B tools/run_spatial_phase_backbone_screen.py preflight \
  --real-root "$REAL_ROOT" \
  --sim-pretrain "$SIM_PRETRAIN" \
  --kit-root "$KIT_ROOT" \
  --out-root "$OUT" \
  --workers 4
```

Require preflight success before training.

## Run exactly one candidate

```bash
CUDA_VISIBLE_DEVICES=<GPU> python -u -B tools/run_spatial_phase_backbone_screen.py candidate \
  --real-root "$REAL_ROOT" \
  --sim-pretrain "$SIM_PRETRAIN" \
  --kit-root "$KIT_ROOT" \
  --out-root "$OUT" \
  --workers 4
```

Do not run a Control arm.

## Review

After candidate completes:

```bash
python -u -B tools/run_spatial_phase_backbone_screen.py review \
  --real-root "$REAL_ROOT" \
  --sim-pretrain "$SIM_PRETRAIN" \
  --kit-root "$KIT_ROOT" \
  --out-root "$OUT" \
  --workers 4
```

The decision is Candidate @8000 versus the frozen historical Control @8000. Candidate @8723 versus historical Control @8723 is supporting evidence.

Do not replace this with best-vs-best checkpoint comparison.

## Archive lightweight evidence

Use a fresh destination, for example:

```bash
DEST=docs/spatial_phase_backbone_screen/results/20260925_run1
mkdir -p "$DEST/candidate_backbone"

cp "$OUT/preflight.json" \
   "$OUT/summary.json" \
   "$OUT/SPATIAL_PHASE_BACKBONE_SCREEN_REVIEW.md" \
   "$DEST/"

cp "$OUT/candidate_backbone/run_config.json" \
   "$OUT/candidate_backbone/sampling_audit.json" \
   "$OUT/candidate_backbone/summary.json" \
   "$DEST/candidate_backbone/"

for D in "$OUT"/candidate_backbone/eval_step_*; do
  NAME=$(basename "$D")
  mkdir -p "$DEST/candidate_backbone/$NAME"
  cp "$D/metrics.json" "$D/by_horizon.csv" "$DEST/candidate_backbone/$NAME/"
done
```

Do not copy checkpoint binaries, H5 files, full console logs, prediction arrays or large runtime artifacts into Git.

## Commit and push

```bash
git status --short
git add "$DEST"
git diff --cached --check
git diff --cached --name-only
git commit -m "Archive spatial phase backbone screen evidence"
git pull --rebase origin main
git push origin main
git rev-parse HEAD
git rev-parse origin/main
```

Verify that only the explicit result destination was staged.

## Final report

Return:

```text
REALPDE_SPATIAL_PHASE_BACKBONE_SCREEN_V1
Status: REVIEW_REQUIRED / BLOCKED
Execution commit: ...
Tests: PASS / FAIL
Candidate: COMPLETE / BLOCKED
Candidate phase assignment: ...
Historical Control @8000: Rel / TKE / MVPE / point
Candidate @8000: Rel / TKE / MVPE / point
@8000 relative deltas: ...
Historical Control @8723: Rel / TKE / MVPE / point
Candidate @8723: Rel / TKE / MVPE / point
@8723 relative deltas: ...
Candidate best iteration: ... (diagnostic only)
Gate: GO / NO_GO / BLOCKED
AoA10 Holdout accessed: NO
Locked-final/private accessed: NO
Codabench accessed: NO
Residual training started: NO
Full-data refit started: NO
Submission packaging started: NO
Automatic follow-up started: NO
Results commit: ...
Remote main verified: YES / NO
```

Then STOP.
