# SPATIAL_PHASE_GATE_CODEX_EXECUTION - REALPDE_SPATIAL_PHASE_GATE_V1

Status: `IMPLEMENTED_BY_SOL / EXECUTE_ONLY / REVIEW_REQUIRED`

Codex role: sync the approved code, resolve already-approved assets, run tests/preflight, execute the two frozen matched arms, run review, archive lightweight evidence, push results to `origin/main`, then stop.

Do not redesign the experiment and do not start any follow-up.

## 1. Sync and repository gate

```bash
git checkout main
git fetch origin
git pull --rebase origin main
git status --short
git rev-parse HEAD
git rev-parse origin/main
git push --dry-run origin HEAD:main
```

Require:

- current branch is `main`;
- no staged changes and no tracked-file modifications/deletions;
- `HEAD == origin/main`;
- dry-run push succeeds.

Pre-existing untracked files are allowed **only** under this rule:

1. capture their exact list before execution with `git status --porcelain=v1`;
2. confirm every non-empty status entry is `?? <path>` (untracked only);
3. confirm none overlaps task-owned paths, especially:
   - `tools/run_spatial_phase_gate.py`
   - `tools/colleague_80pt/realpde_h5_feature_adapter_train.py`
   - `tools/colleague_80pt/residual_multi.py`
   - `tools/evaluate_clean_residual_checkpoint.py`
   - `tests/test_spatial_phase_gate.py`
   - `docs/spatial_phase_gate/`
4. leave all such pre-existing untracked files untouched;
5. never use `git add -A` or `git add .`; stage only the explicit result destination.

Tracked/staged changes, or untracked files overlapping task-owned paths: STOP and report `BLOCKED`.

Do not stash/reset/clean pre-existing files as part of this task. The presence of unrelated, recorded untracked files by itself is **not** a blocker.

## 2. Resolve approved assets

Use the existing clean-campaign assets only.

Required checkpoint digests:

```text
Strong Backbone:
d340effd68031aba1cd2bc610c676878fb9105dd7f81b9ce15767280e665a5c0

Strong Residual:
d2b4ddd0df6064d053ec918c2671a93b7d9c2a40a15afe854bb3a5badf1438fc
```

The completed run recorded these checkpoint locations:

```text
/hy-tmp/realpde_runs/strong_backbone_clean_20260924_run1/exp1_strong_backbone_clean/strong_backbone/checkpoints/model_best.pth
/hy-tmp/realpde_runs/strong_backbone_clean_20260924_run1/exp1_strong_backbone_clean/residual_22k/model_best.pth
```

Trust SHA256, not path. If paths have moved, locate only these approved assets by digest/path knowledge. Do not search locked-final/private locations.

Typical released real-PIV root from the clean run:

```text
/hy-tmp/realpde_data/train_real
```

Resolve the same v9 starting-kit root used by the Strong Backbone Clean run.

## 3. Focused tests and compile

Run before any optimizer update:

```bash
python -m pytest -q \
  tests/test_spatial_phase_gate.py \
  tests/test_clean_baseline_v1.py \
  tests/test_clean_baseline_final_campaign.py

python -m py_compile \
  tools/run_spatial_phase_gate.py \
  tools/colleague_80pt/realpde_h5_feature_adapter_train.py \
  tools/colleague_80pt/residual_multi.py \
  tools/evaluate_clean_residual_checkpoint.py

git diff --check
```

Any scientific-code/test failure: `BLOCKED`.

Environment/path repair is allowed only when it does not change phase semantics, data split, model, loss, optimizer, LR, update budget, batch size, seeds, evaluation or gate.

## 4. Preflight

Set variables, using real approved paths:

```bash
REAL_ROOT=/hy-tmp/realpde_data/train_real
STRONG_BACKBONE=/hy-tmp/realpde_runs/strong_backbone_clean_20260924_run1/exp1_strong_backbone_clean/strong_backbone/checkpoints/model_best.pth
STRONG_RESIDUAL=/hy-tmp/realpde_runs/strong_backbone_clean_20260924_run1/exp1_strong_backbone_clean/residual_22k/model_best.pth
KIT_ROOT=<same-v9-kit-root-used-by-strong-backbone-run>
OUT=/hy-tmp/realpde_runs/spatial_phase_gate_20260925_run1
```

Then:

```bash
python -u -B tools/run_spatial_phase_gate.py preflight \
  --real-root "$REAL_ROOT" \
  --strong-backbone "$STRONG_BACKBONE" \
  --strong-residual "$STRONG_RESIDUAL" \
  --kit-root "$KIT_ROOT" \
  --out-root "$OUT" \
  --workers 4
```

Preflight must pass checkpoint SHA checks and record `holdout_accessed=false`.

## 5. Execute exactly two arms

If two approved GPUs are available, Control and Candidate may run concurrently in separate detached sessions. Otherwise run serially. Do not alter science to obtain concurrency.

Control:

```bash
CUDA_VISIBLE_DEVICES=<GPU> python -u -B tools/run_spatial_phase_gate.py control \
  --real-root "$REAL_ROOT" \
  --strong-backbone "$STRONG_BACKBONE" \
  --strong-residual "$STRONG_RESIDUAL" \
  --kit-root "$KIT_ROOT" \
  --out-root "$OUT" \
  --workers 4
```

Candidate:

```bash
CUDA_VISIBLE_DEVICES=<GPU> python -u -B tools/run_spatial_phase_gate.py candidate \
  --real-root "$REAL_ROOT" \
  --strong-backbone "$STRONG_BACKBONE" \
  --strong-residual "$STRONG_RESIDUAL" \
  --kit-root "$KIT_ROOT" \
  --out-root "$OUT" \
  --workers 4
```

Frozen science:

- both resume the same selected Strong Residual state;
- both reset optimizer;
- 5,000 updates, batch 8, LR `1e-5`;
- exact same temporal sampling and clean residual recipe;
- Control P00 only;
- Candidate 50% P00 + 50% balanced P01/P10/P11;
- Seen-Dev evaluation P00 only.

Monitor GPU/host RAM, disk, process and logs. Candidate RAM preload intentionally holds the full-resolution train fields, so RAM usage is higher than Control. If resource limits are exceeded, STOP and report. Do not change preload behavior, batch, workers or science without Sol review.

## 6. Review

After both arm `DONE` markers exist:

If Control and Candidate have already completed successfully and only the review step failed because of review-code logic, **do not rerun either training arm**. Sync the reviewed fix on `main`, rerun the focused tests/compile, and rerun only the review command against the existing `OUT` artifacts. Training artifacts remain authoritative as long as their two arm `DONE` markers and exit-code evidence are intact.


```bash
python -u -B tools/run_spatial_phase_gate.py review \
  --real-root "$REAL_ROOT" \
  --strong-backbone "$STRONG_BACKBONE" \
  --strong-residual "$STRONG_RESIDUAL" \
  --kit-root "$KIT_ROOT" \
  --out-root "$OUT" \
  --workers 4
```

Review must verify exact initialization parity, identical temporal sampling SHA, and expected phase exposure before emitting GO/NO_GO.

Review-code recovery must not modify or regenerate either arm's checkpoints, training logs, sampling audits, Seen-Dev predictions or metrics. If any required arm artifact is missing or corrupt, STOP and report `BLOCKED` instead of silently rerunning training.

Do not execute AoA10 after GO. This gate stops at Seen-Dev.

## 7. Archive lightweight evidence only

Create a fresh destination, for example:

```bash
DEST=docs/spatial_phase_gate/results/20260925_run1
mkdir -p "$DEST/control" "$DEST/candidate"
```

Copy only lightweight review evidence:

```bash
cp "$OUT/preflight.json" "$OUT/train_dev_manifest.json" \
   "$OUT/summary.json" "$OUT/SPATIAL_PHASE_GATE_REVIEW.md" "$DEST/"

for ARM in control candidate; do
  cp "$OUT/$ARM/arm_manifest.json" "$DEST/$ARM/"
  cp "$OUT/$ARM/train/run_config.json" \
     "$OUT/$ARM/train/sampling_audit.json" \
     "$OUT/$ARM/train/summary.json" \
     "$DEST/$ARM/"
  mkdir -p "$DEST/$ARM/seen_dev_best_p00"
  cp "$OUT/$ARM/seen_dev_best_p00/final_primary_metrics.json" \
     "$OUT/$ARM/seen_dev_best_p00/by_horizon.csv" \
     "$OUT/$ARM/seen_dev_best_p00/by_trajectory.csv" \
     "$OUT/$ARM/seen_dev_best_p00/by_trajectory_horizon.csv" \
     "$OUT/$ARM/seen_dev_best_p00/manifest.json" \
     "$DEST/$ARM/seen_dev_best_p00/"
done
```

Do **not** copy checkpoint binaries, raw H5, full console logs, prediction arrays or other large artifacts into Git.

Inspect archive size and content before staging.

## 8. Commit and push evidence

```bash
git status --short
# Pre-existing unrelated untracked files may still be listed here; do not touch them.
git add "$DEST"
git diff --cached --check
# Verify the index contains only files beneath $DEST.
git diff --cached --name-only
git commit -m "Archive spatial phase gate evidence"
git pull --rebase origin main
git push origin main
git rev-parse HEAD
git rev-parse origin/main
```

Delivery requires remote `main` verification.

## 9. Final report and hard stop

Return exactly the execution facts needed for Sol review:

```text
REALPDE_SPATIAL_PHASE_GATE_V1
Status: REVIEW_REQUIRED / BLOCKED
Execution commit: ...
Tests: PASS / FAIL
Control: COMPLETE / BLOCKED
Candidate: COMPLETE / BLOCKED
Init parity: ...
Temporal sampling parity: ...
Control phase exposure: ...
Candidate phase exposure: ...
P00 Seen-Dev Control: Rel / TKE / MVPE / point
P00 Seen-Dev Candidate: Rel / TKE / MVPE / point
Gate: GO / NO_GO / BLOCKED
Locked-final/private accessed: NO
AoA10 holdout accessed: NO
Codabench accessed: NO
Full-data refit started: NO
Submission packaging started: NO
Automatic follow-up started: NO
Results commit: ...
Remote main verified: YES / NO
```

Then STOP. A GO does not authorize another experiment or final long training.
