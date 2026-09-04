# A1 Local Branch Audit

Experiment: `T1-ID-HYBRID-CNO-LOCAL-A1-S20260904`

Audit scope: existing A1@3000 checkpoint and saved 16-trajectory Dev artifact only. No retraining, model change, or re-evaluation was performed.

## Verdict flags

- `TRAINING_BRANCH_ACTIVE = NO`
- `EVAL_BRANCH_ACTIVE = NO` — the branch is called by the full forward path, but contributes exactly zero output.
- `ANALYSIS_RESIDUAL_CORRECT = YES`
- `A1_RESULT_VALID = NO`

## Evidence

1. The saved checkpoint contains `local_state_dict`, but its optimizer has 136 parameter tensors (global CNO only); the local branch has 4 parameter tensors and none are represented in the optimizer state/groups. The runner source confirms `AdamW(model.parameters(), ...)` and omits `local.parameters()`.
2. Local branch parameter norms, reconstructed at initialization with the frozen seed/order, are unchanged at update 3000:

   - `input.weight`: init `2.3146324`, final `2.3146324`, delta `0`
   - `input.bias`: init `0.3020560`, final `0.3020560`, delta `0`
   - `output.weight`: init `0`, final `0`, delta `0`
   - `output.bias`: init `0`, final `0`, delta `0`

3. On a real Dev batch `[8,20,32,64,3]`:

   - global output RMS: `0.1199081`
   - local output RMS / max abs / norm: `0 / 0 / 0`
   - full minus global RMS: `0`
   - `full_uv - global_uv == local_uv`: exact; max absolute identity error `0`

4. A train-batch gradient check at the saved zero local state gives nonzero final-projection gradients (`output.weight` norm `0.0352395`, `output.bias` norm `0.0745895`) but zero input-layer gradient, as expected when the zero final projection blocks upstream gradient. This is consistent with the local parameters being omitted from the optimizer, not detached or overwritten.

5. The saved prediction artifact contains `local_residual_uv`; its RMS is `0`. `local_residual_windows.csv` has 659 rows, all residual RMS/norm values are `0`, and the CSV was generated from that real saved local-output array. The residual analysis is therefore correct.

## Separate metrics

Official v9 raw metrics:

| model | Rel-L2 | TKE | MVPE |
|---|---:|---:|---:|
| A1 full model | 0.17646120 | 0.59071702 | 0.15541382 |
| A1 global-only | 0.17646120 | 0.59071702 | 0.15541382 |
| matched Direct | 0.17582881 | 0.59464884 | 0.15163144 |

The A1 full/global equality is exact at the saved artifact level, so the previous A1 architecture comparison cannot be interpreted as evidence for a learned Local branch. Preserve the original report for historical traceability, but mark this experiment `INVALID` for branch-effect claims.

