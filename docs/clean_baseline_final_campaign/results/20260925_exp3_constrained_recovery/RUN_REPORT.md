# Exp3 Constrained SPS Recovery

Status: `REVIEW_REQUIRED`

## Execution

- Execution HEAD: `cbac90a8be0678d2778a2cab69a7b238638ba323`
- Recovery script: `tools/recover_clean_exp3_sps.py` from that HEAD, staged under `/tmp` on the GPU host; original Exp3 source directory remained read-only.
- Tests: `py_compile` PASS; `pytest tests/test_clean_baseline_final_campaign.py -q` PASS (10 passed); `git diff --check` PASS.
- Device: CUDA (`torch 2.4.0+cu121`).
- Original Exp3 checkpoint SHA256: `1cde13a23fe15547dc2ee16cede9edd89fe8c12bc88f8e9ad059641f0a6c90b8` (PASS).
- Clean Stage2 checkpoint SHA256: `1536fe02e11fc0dc991a38cb1489d0be0752ec8149be7b9f1542a4cb81c2f975` (PASS).
- Training: `optimizer_steps = 0`; `training_performed = false`.

## Seen-Dev constrained selection

- Selection was recomputed from the saved static and adaptive calibration grids.
- Selected step: `4500`.
- Calibration: `floor=0.0025`, `mult_u=0.75`, `mult_v=1.5`, `rel=0.0075`.
- Static SPS: `48.4163597846`.
- Adaptive SPS: `51.6544348700`.
- SPS gain: `+3.2380750853`.
- Width ratio: `1.1926004028` (cap `1.20`).
- Point prediction parity max abs: `0` (limit `1e-7`).
- Seen-Dev gate: `GO`.

## One frozen AoA10 evaluation

Seen-Dev GO allowed the single frozen AoA10 evaluation with the selected Seen-Dev calibration. AoA10 was not used for selection and was not recalibrated.

- Static SPS: `45.8426232545`.
- Adaptive SPS: `49.0174144371`.
- SPS delta: `+3.1747911826`.
- Static coverage: `0.7722953367`.
- Adaptive coverage: `0.8370197539`.
- Coverage delta: `+0.0647244172` (about `+6.47` percentage points).
- Adaptive mean interval width (u/v): `0.0151314000`.
- `holdout_used_for_selection = false`; `holdout_recalibrated = false`.

## Scope checks

- Locked-final/private accessed: `NO`.
- Codabench accessed: `NO`.
- Full-data refit or submission/package: `NO`.
- Follow-up experiment started: `NO`.
- Status remains `REVIEW_REQUIRED` pending review.
