# CODEX TASK — EXP3 CONSTRAINED SPS REVIEW-ONLY RECOVERY

Status: `REVIEW_REQUIRED`

Goal: repair the completed Exp3 review by applying the already-frozen gate correctly:

> maximize Seen-Dev adaptive SPS **subject to** mean interval width <= 1.20x the same-Dev best static width.

This is **not a new experiment**. Do not retrain. Do not redesign SPS. Do not add another calibration family.

The code has already been written by ChatGPT/Sol:
- `tools/train_clean_residual_aware_sps.py`
- `tools/recover_clean_exp3_sps.py`
- `tests/test_clean_baseline_final_campaign.py`
- `docs/clean_baseline_final_campaign/README.md`

Codex's role is execution, verification, evidence collection and reporting only.

## Expected recovery facts

Completed source run:

`/hy-tmp/realpde_runs/pareto_sps_20260924_run1`

Archived head SHA256:

`1cde13a23fe15547dc2ee16cede9edd89fe8c12bc88f8e9ad059641f0a6c90b8`

From the archived grids, the corrected Seen-Dev constrained optimum is expected to be:

- selected step: `4500`
- floor: `0.0025`
- mult_u: `0.75`
- mult_v: `1.5`
- rel: `0.0075`
- adaptive Seen-Dev SPS: about `51.65443487`
- SPS gain vs static: about `+3.23807509`
- width ratio: about `1.19260040`

These numbers are verification targets, not values to hard-code or force. The recovery script must recompute them from the saved grids/checkpoint.

---

# HARD CONSTRAINTS

1. **ZERO TRAINING**
   - 0 optimizer steps.
   - No `backward()`.
   - No continuation from step 4500/5000.
   - No fresh uncertainty-head training.
   - No point-model training.
   - If recovery cannot be completed from the saved head, stop as `BLOCKED`. Never retrain to repair evidence.

2. **SOURCE CHECKPOINT MUST MATCH**
   - Use only `/hy-tmp/realpde_runs/pareto_sps_20260924_run1/head_best.pth`.
   - Its SHA256 must equal:
     `1cde13a23fe15547dc2ee16cede9edd89fe8c12bc88f8e9ad059641f0a6c90b8`.
   - If missing or mismatched: stop `BLOCKED`.

3. **SAVED STEP MUST MATCH CORRECTED OPTIMUM**
   - The corrected width-constrained optimum must resolve to the same step stored in `head_best.pth`.
   - Expected: step 4500.
   - If the constrained optimum needs another checkpoint, stop `BLOCKED`.
   - Do not reconstruct, interpolate, approximate or retrain a missing checkpoint.

4. **FROZEN SCIENTIFIC PROTOCOL**
   - Do not change the 300-combination adaptive calibration family.
   - Do not change static calibration family.
   - Do not change width cap `1.20x`.
   - Do not change minimum SPS gain `+1.0`.
   - Do not change Train51 / Seen-Dev12 / AoA10 Holdout18.
   - Do not change stride/window/sub-sampling semantics.
   - Do not add features, losses, smoothing, clipping, ensemble, interpolation or new heuristics.

5. **AoA10 ACCESS RULE**
   - AoA10 Holdout may be accessed **only after** the corrected Seen-Dev gate is recomputed and returns `GO`.
   - Exactly one frozen evaluation pass.
   - Apply the Seen-Dev-selected calibration unchanged.
   - **No AoA10 recalibration.**
   - **No AoA10-based model/calibration/checkpoint selection.**
   - If Seen-Dev is `NO_GO`, AoA10 must remain untouched.

6. **PRIVATE / FINAL / SUBMISSION BOUNDARY**
   - Do not access locked-final/private data.
   - Do not access Codabench.
   - Do not build or test a submission package.
   - Do not run full-data refit.
   - Do not start any subsequent experiment.

7. **REPOSITORY POLICY**
   - Work only on `main`.
   - First `git pull --ff-only origin main`.
   - Do not create a branch.
   - Do not modify the scientific code written by ChatGPT/Sol.
   - Existing unrelated untracked files must remain untouched.
   - If execution reveals a code defect, stop and report it. Do not independently redesign/fix the implementation.

8. **EVIDENCE IMMUTABILITY**
   - Do not modify or overwrite the original source run.
   - Recovery output must go to a new directory.
   - Do not upload checkpoint binaries to GitHub.
   - Final status remains `REVIEW_REQUIRED` even if the Seen-Dev gate is `GO`.

---

# SOFT CONSTRAINTS

- Reuse the existing Python/CUDA environment; avoid package installation or environment churn.
- Prefer the exact original paths/checkpoints/manifests from the completed clean campaign.
- Resolve paths from the original run/campaign evidence or prior command history. Do not substitute a different checkpoint because it is convenient.
- Keep GPU work inference-only and minimal.
- Keep evidence compact: JSON summaries, audit, hashes, concise report. No large tensors, H5 files or weights.
- Do not refactor unrelated code.
- Do not broaden the task into SPS research.
- Preserve reproducibility: record HEAD, GPU/runtime, source hashes and exact command.

---

# EXECUTION

## 1. Sync and preflight

Run:

```bash
cd <repo>
git checkout main
git pull --ff-only origin main
git status --short
git rev-parse HEAD
```

Record HEAD.

Verify these files exist:

```bash
test -f tools/train_clean_residual_aware_sps.py
test -f tools/recover_clean_exp3_sps.py
test -f tests/test_clean_baseline_final_campaign.py
test -d /hy-tmp/realpde_runs/pareto_sps_20260924_run1
test -f /hy-tmp/realpde_runs/pareto_sps_20260924_run1/head_best.pth
```

Verify head SHA before any evaluation:

```bash
sha256sum /hy-tmp/realpde_runs/pareto_sps_20260924_run1/head_best.pth
```

Expected exactly:

`1cde13a23fe15547dc2ee16cede9edd89fe8c12bc88f8e9ad059641f0a6c90b8`

If not exact: stop `BLOCKED`.

## 2. Code verification

Run:

```bash
python -m py_compile   tools/train_clean_residual_aware_sps.py   tools/recover_clean_exp3_sps.py

python -m pytest tests/test_clean_baseline_final_campaign.py -q

git diff --check
```

Any failure: stop and report. Do not patch the scientific code yourself.

## 3. Resolve frozen input paths

Use the **same** clean Stage2 residual checkpoint, RealPDE data root, model root, Train51/Seen-Dev12 manifest and AoA10 Holdout18 manifest used by the completed clean campaign.

Required residual checkpoint SHA256:

`1536fe02e11fc0dc991a38cb1489d0be0752ec8149be7b9f1542a4cb81c2f975`

Verify it before execution.

If the exact original manifest/path cannot be established unambiguously, stop `BLOCKED`. Do not invent a replacement split.

## 4. Run review-only recovery

Use a **new** output directory, for example:

`/hy-tmp/realpde_runs/pareto_sps_20260924_run1_constrained_recovery`

It must not already exist.

Run:

```bash
python -u -B tools/recover_clean_exp3_sps.py   --source-run /hy-tmp/realpde_runs/pareto_sps_20260924_run1   --real-root "<EXACT_ORIGINAL_REAL_ROOT>"   --train-manifest "<EXACT_ORIGINAL_TRAIN_DEV_MANIFEST>"   --holdout-manifest "<EXACT_ORIGINAL_AOA10_HOLDOUT_MANIFEST>"   --residual-checkpoint "<EXACT_CLEAN_STAGE2_CHECKPOINT>"   --model-root "<EXACT_ORIGINAL_MODEL_ROOT>"   --out-dir /hy-tmp/realpde_runs/pareto_sps_20260924_run1_constrained_recovery   --expected-head-sha256 1cde13a23fe15547dc2ee16cede9edd89fe8c12bc88f8e9ad059641f0a6c90b8   --workers 4   --require-cuda
```

Do not wrap this in a training runner.

## 5. Required review checks

Inspect:

- `selection_audit.json`
- `summary.json`

Confirm all of the following:

- `optimizer_steps == 0`
- `training_performed == false`
- selected step == `4500`
- selected calibration == `floor 0.0025 / mult_u 0.75 / mult_v 1.5 / rel 0.0075`
- point parity <= `1e-7`
- Seen-Dev SPS gain >= `+1.0`
- Seen-Dev width ratio <= `1.20`
- corrected Seen-Dev gate == `GO`
- `holdout_used_for_selection == false`
- `holdout_recalibrated == false`
- `codabench_accessed == false`
- `locked_final_accessed == false`

If the corrected Seen-Dev gate is not `GO`, confirm `holdout_accessed == false` and stop.

If it is `GO`, the script is allowed to perform the single frozen AoA10 evaluation. Report its static and adaptive:
- SPS
- coverage
- interval width
- SPS delta vs static
- coverage delta vs static

Do **not** make the final scientific merge decision.

## 6. Archive evidence to GitHub

Create a compact evidence directory under:

`docs/clean_baseline_final_campaign/results/20260925_exp3_constrained_recovery/`

Include only:
- `summary.json`
- `selection_audit.json`
- `RUN_REPORT.md`
- `SHA256SUMS.txt`
- optional small environment/source-hash JSON if useful

Do **not** include:
- `head_best.pth`
- any model checkpoint
- H5 data
- cached predictions/tensors
- large logs

`RUN_REPORT.md` must explicitly state:

- `Status: REVIEW_REQUIRED`
- this was a review-only recovery
- 0 optimizer steps
- source head SHA
- corrected constrained selection rule
- corrected Seen-Dev metrics/gate
- whether AoA10 was accessed
- AoA10 result if accessed
- no recalibration on AoA10
- locked-final/private not accessed
- Codabench not accessed
- no follow-up experiment started

Then:

```bash
git status --short
git diff --check
git add docs/clean_baseline_final_campaign/results/20260925_exp3_constrained_recovery/
git commit -m "Archive Exp3 constrained SPS recovery"
git push origin main
```

Only evidence files may be committed by this task.

---

# FINAL CODEX REPORT

Return a compact report with:

1. Status: `REVIEW_REQUIRED / SUCCESS` or `REVIEW_REQUIRED / BLOCKED`
2. execution HEAD
3. tests / py_compile / diff-check
4. source head SHA verification
5. optimizer steps
6. corrected Seen-Dev:
   - selected step
   - calibration
   - static SPS
   - adaptive SPS
   - SPS gain
   - width ratio
   - gate
7. AoA10:
   - accessed YES/NO
   - static SPS
   - adaptive SPS
   - delta
   - coverage delta
   - no recalibration confirmation
8. safety:
   - locked-final/private
   - Codabench
   - full-data
   - package/submission
9. evidence Git commit SHA

Stop there. Do not launch another experiment.
