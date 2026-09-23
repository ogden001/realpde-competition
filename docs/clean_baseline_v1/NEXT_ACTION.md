# NEXT_ACTION — Execute REALPDE_CLEAN_BASELINE_V1

Status: REVIEW_REQUIRED

## Goal

Execute the already-implemented clean baseline. Do not redesign the model, loss, split, sampler, budgets, or checkpoint-selection rule.

## Hard protocol

- Branch: main only.
- Pull the latest main before execution.
- Working tree must be clean.
- Do not access locked-final/private data.
- Do not access Codabench.
- Do not package or submit.
- Do not start any all81/all82/full-data refit.
- Do not change code to improve metrics.
- If a code/data/environment error occurs, stop and report REVIEW_REQUIRED with evidence. Do not independently invent a workaround that changes scientific semantics.

## Step 1: update and test

Run:

```bash
git checkout main
git pull --ff-only
python -m pytest -q tests/test_clean_baseline_v1.py tests/test_colleague_incremental_screen.py
```

If tests fail, stop.

Also run a syntax/import smoke check for:

```bash
python -m py_compile   tools/colleague_80pt/train_clean_baseline_cno.py   tools/colleague_80pt/run_clean_baseline_v1.py   tools/colleague_80pt/residual_multi.py
```

## Step 2: resolve existing approved paths only

Locate the already-present released real PIV directory, official sim_real CNO checkpoint, and RealPDEBench/model root on this GPU.

Expected examples from previous environments include:

- real PIV: /data/p0ab_real_h5_20260830 or the corresponding migrated /hy-tmp path
- official init: /data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth
- model root: /third_party

Do not search or read locked-final/private locations.

Verify the official init SHA256 is:

`82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61`

## Step 3: execute once

Choose a new, non-existing output directory, for example:

`/hy-tmp/runs/realpde_clean_baseline_v1_20260924`

Run:

```bash
python -u -B tools/colleague_80pt/run_clean_baseline_v1.py   --real-root <REAL_PIV_ROOT>   --sim-pretrain-checkpoint <SIM_REAL_CNO_PT>   --model-root <REALPDEBENCH_MODEL_ROOT>   --out-root <NEW_OUTPUT_ROOT>   --workers 4
```

Do not launch a second arm or parameter sweep.

## Step 4: report

Return a compact report with:

- Status: REVIEW_REQUIRED / FAILED
- execution commit SHA
- GPU
- canonical split SHA
- split audit: 51 Train / 12 Seen-Dev / 18 AoA10 Holdout
- Stage1 best iteration and Rel-L2/TKE/MVPE/point_score
- Stage1 final metrics
- Stage2 step0 metrics
- Stage2 best iteration and Rel-L2/TKE/MVPE/point_score
- Stage2 final metrics
- AoA10 holdout metrics for Stage2 step0/best/final
- Stage1/Stage2 checkpoint SHA256
- runtime if available
- OOM or other errors
- files modified: NO
- training started: YES only after tests and split gates pass
- Codabench accessed: NO
- locked-final accessed: NO
- full-data refit started: NO
- submission/package started: NO

Do not interpret results beyond a short factual summary. ChatGPT/Sol will review the learning curves and decide the next experiment.
