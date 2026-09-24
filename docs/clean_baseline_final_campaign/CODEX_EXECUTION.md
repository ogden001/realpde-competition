# CODEX EXECUTION - CLEAN_BASELINE_FINAL_CAMPAIGN

Status: `IMPLEMENTED_BY_SOL / EXECUTE_ONLY / REVIEW_REQUIRED`

Codex role: environment adaptation, tests, execution, monitoring, evidence archive, commit/push. Do not redesign the experiment.

## 1. Sync and write-permission gate

```bash
git checkout main
git fetch origin
git pull --rebase origin main
git status --short
git rev-parse HEAD
git rev-parse origin/main
git push --dry-run origin HEAD:main
```

Require clean tree, `HEAD == origin/main`, and dry-run push success. Unknown local changes => STOP. Do not stash/reset/clean them.

## 2. Resolve approved assets

Locate released real-PIV root; official `sim_real_cno.pth` SHA256 `82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61`; RealPDEBench/model root; v9 starting-kit root; Clean Stage1 best SHA256 `6aec4edb5e42746080e19baaafa2d6b719bb57befb1ff9035ece7889ec280036`; Clean Stage2 best SHA256 `1536fe02e11fc0dc991a38cb1489d0be0752ec8149be7b9f1542a4cb81c2f975`.

Do not search locked-final/private paths. Expected old run path may be `/hy-tmp/runs/realpde_clean_baseline_v1_20260924`, but trust SHA, not path.

## 3. Tests and compile

```bash
python -m pytest -q \
  tests/test_clean_baseline_final_campaign.py \
  tests/test_clean_baseline_v1.py \
  tests/test_colleague_v2_campaign.py \
  tests/test_sota_v2_adaptive.py

python -m py_compile \
  tools/clean_baseline_final_campaign.py \
  tools/train_clean_strong_backbone.py \
  tools/evaluate_clean_residual_checkpoint.py \
  tools/train_clean_residual_aware_sps.py

git diff --check
```

Code/test failure => `BLOCKED`. Pure environment/path fixes are allowed only if model/loss/data/split/sampling/budget/gates are unchanged.

## 4. Preflight

Set a fresh output root, for example:

```bash
OUT=/hy-tmp/realpde_runs/clean_baseline_final_campaign_20260924_v1
```

Run the common preflight with real paths:

```bash
python -u -B tools/clean_baseline_final_campaign.py preflight \
  --real-root "$REAL_ROOT" \
  --sim-pretrain "$SIM_PRETRAIN" \
  --model-root "$MODEL_ROOT" \
  --kit-root "$KIT_ROOT" \
  --clean-stage1 "$CLEAN_STAGE1" \
  --clean-stage2 "$CLEAN_STAGE2" \
  --out-root "$OUT" \
  --workers 4
```

## 5. First two experiments

If two approved GPUs exist, run Exp1 and Exp2 concurrently in separate detached `tmux` sessions, same checkout, separate `CUDA_VISIBLE_DEVICES`. If only one GPU is approved, run Exp1 then Exp2 serially. Never alter batch size, budgets, precision or science to force concurrency.

Use the same common arguments as preflight:

```bash
CUDA_VISIBLE_DEVICES=<GPU> python -u -B tools/clean_baseline_final_campaign.py exp1 ...common args...
CUDA_VISIBLE_DEVICES=<GPU> python -u -B tools/clean_baseline_final_campaign.py exp2 ...common args...
```

Monitor process/session, GPU memory/utilization, disk, logs and artifacts every ~10-15 minutes. Do not early-stop from metric fluctuations.

## 6. Experiment 3

When an approved GPU is free:

```bash
CUDA_VISIBLE_DEVICES=<GPU> python -u -B tools/clean_baseline_final_campaign.py exp3 ...common args...
```

Exp3 uses frozen clean-baseline Stage2 best and does not require an intermediate Sol winner selection.

## 7. Review and archive

After all three `DONE` markers exist:

```bash
python -u -B tools/clean_baseline_final_campaign.py review ...common args...

DEST=docs/clean_baseline_final_campaign/results/20260924_run1
python -u -B tools/clean_baseline_final_campaign.py archive \
  ...common args... \
  --archive-dest "$DEST"
```

If `DEST` exists, use run2/run3. Never overwrite. Check archive size/manifest. Checkpoints, raw H5 and large raw logs must not enter Git.

## 8. Commit and push evidence

```bash
git status --short
git add "$DEST"
git diff --cached --check
git commit -m "Archive clean baseline final campaign evidence"
git pull --rebase origin main
git push origin main
git rev-parse HEAD
git rev-parse origin/main
```

Task is not delivered until the results commit is on remote `main`.

## 9. Runtime planning only

Based on project evidence on RTX 3090 Ti / similar hardware:

- Exp1: strong backbone 35k roughly 6-7h plus clean residual 22k roughly 2h, total about 8-9h.
- Exp2: scalar control 22k roughly 2h; Pareto projection arm expected roughly 2.5-3.5h; sequential about 4.5-5.5h. With two GPUs, wall time is dominated by slower arm.
- Exp3: residual-aware SPS 5k roughly 1-1.5h because point predictor is frozen and only head backpropagates.

Do not change science from these estimates.

## 10. Final report and hard stop

Return only after push:

```text
CLEAN_BASELINE_FINAL_CAMPAIGN
Status: REVIEW_REQUIRED / BLOCKED
Execution commit: ...
Tests: PASS / FAIL
Exp1: COMPLETE / BLOCKED
Exp1 gate: ...
Exp2: COMPLETE / BLOCKED
Exp2 selected beta: ...
Exp3: COMPLETE / BLOCKED
Exp3 gate: ...
Locked-final/private accessed: NO
Codabench accessed: NO
Full-data refit started: NO
Submission packaging started: NO
Results commit: ...
Remote main verified: YES / NO
```

Then STOP. Do not propose or start another experiment. Sol will analyze the GitHub evidence.
