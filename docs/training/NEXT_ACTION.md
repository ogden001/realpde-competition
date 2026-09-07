# NEXT_ACTION

## Goal
Record the stopped RW-01 shuffle revalidation control for ChatGPT/Sol review.

## Tasks
1. Preserve the pushed deterministic global-shuffle implementation and its invariant tests.
2. Review the completed RW-CTRL evidence; MVPE exceeded the pre-registered control tolerance.
3. Do not run RW-01 or alter the experiment without ChatGPT/Sol direction.

## Constraints
- Frozen 50 Train / 16 Dev; P0-A CNO, N2, `sim_pretrain/sim_cno.pth`, seed `20260901`, AdamW `1e-5`, batch 8, workers 2.
- Dev remains `start=0, stride=20`; reuse RW-00 without retraining it.
- Only variable is train-window phase randomization. No locked-final, Codabench, full-data, Random Start, stride/phase sweep, or next experiment.
- Final research status is `REVIEW_REQUIRED`; ChatGPT/Sol owns the scientific conclusion.

## Deliverables
- [RW-CTRL review package](../experiments/rw01_shuffle_revalidation_20260907/README_FOR_CHATGPT.md).
- Historical RW-00/RW-01 result remains `INVALID_COMPARISON / SHUFFLE_CONFOUND`.

## Stop
Stop. Await ChatGPT/Sol review; do not design or run another experiment.
