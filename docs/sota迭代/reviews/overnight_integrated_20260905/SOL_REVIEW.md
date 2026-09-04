# Sol Review — Overnight Integrated Adaptive Probe

Status: **`INVALID_IMPLEMENTATION / REVALIDATION_REQUIRED`**

The run stopped safely before full refit/package/Codabench, but the validation artifacts cannot be used for the pre-registered Gate. The blocker is not only the remote heredoc SyntaxError: the executed implementation drifted from the frozen experiment contract in several material ways.

## Verified implementation issues

1. **Corrector `residual_mse` target is wrong.**
   - Frozen semantics: `MSE(delta_uv, target_uv - backbone_prediction_uv)`.
   - Executed code calls `corrector_loss(base_pred + delta, y, delta)`, while `corrector_loss` computes `delta - (target - prediction)`. This makes the residual target depend on the already-corrected prediction and is not the frozen reference loss.

2. **The so-called base uncertainty head is trained on corrected predictions.**
   - Frozen protocol: base head must model uncertainty of the uncorrected frozen backbone; corrected head is trained only after the corrector Gate passes.
   - Executed `batches_head()` always forms `final = base_pred + delta` and trains the only head against `final`, then saves it as `base_head`.

3. **Initial sigma=0.02 is not implemented.**
   - Frozen reference requires uncertainty output to start at sigma `0.02`.
   - `AdaptiveUncertaintyHead` uses default Conv3d initialization and has no final-layer initialization enforcing `log_std = log(0.02)`.
   - The training log contains an early NLL explosion (hundreds to thousands) before recovery, consistent with an unstable initialization path and requiring correction before interpreting the head.

4. **`--full` currently also refits the uncertainty head.**
   - Frozen protocol explicitly forbids fitting adaptive uncertainty on all-82 in-sample residuals; full mode must refit only the corrector at fixed 3960 updates.
   - `run_training(full=True)` currently executes both corrector and uncertainty-head training.

5. **Packaging path is not yet safe.**
   - The generated `submission.py` template contains literal `+` prefixes in code lines.
   - The package path always instantiates/applies a corrector, so it does not yet implement the required BACKUP semantics (`full@43260 + base adaptive head`, no corrector).

6. **The Gate heredoc error is real but secondary.**
   - `eval_gate.py` failed because an unquoted path was written into Python source. Even after fixing this syntax error, the current validation artifacts must not be gated because items 1–3 invalidate their semantics.

## Evidence that remains useful

- The corrector optimization process itself ran for the intended 2400 updates and its training loss decreased without NaN/Inf.
- The uncertainty head ran for the intended 1400 updates and eventually returned to finite losses, but it is not a valid base-head artifact under the frozen contract.
- The job correctly stopped before full refit, package build, or Codabench, preventing propagation of the invalid validation stage.
- `66 passed` demonstrates code-level tests passed, but current tests do not sufficiently encode the frozen reference semantics above.

## Sol decision

Do **not** run the Gate on the existing validation artifacts. Do **not** start all-82 refit or package build.

Next action is a bounded semantic repair + TDD + validation-only rerun. Only after the corrected validation artifacts and Gate evidence are committed should Sol decide whether full refit/package work is authorized.

`NEXT_ACTION = REPAIR_AND_REVALIDATE_ONLY`
