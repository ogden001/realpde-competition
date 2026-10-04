# REALPDE CLEAN OVERNIGHT CAMPAIGN

**Status: REVIEW_REQUIRED — stopped at the Stage-A gate**

## Stage-A

- Completed update: 60,000 / 60,000
- Trainer PID 6459: exited after reaching target
- Latest Seen-Dev, A@60,000: Rel-L2 0.098830, TKE 0.474816, MVPE 0.073039, point 90.860658
- Seen-Dev champion reported by the run summary, A@57,000: Rel-L2 0.100464, TKE 0.465893, MVPE 0.073000, point 90.934038
- The inspected final train-log tail through update 60,000 contained finite values and no visible NaN/Inf or fatal error
- GPU snapshot after training: RTX 4090, 0% utilization, 1 MiB used

## Review blockers

1. **Holdout protocol ambiguity.** The existing `REALPDE_CLEAN_BASELINE_V1_HOLDOUT_EVAL_ONLY` protocol describes a one-time audit of one Seen-Dev-selected model with `selection_allowed=false`. The overnight campaign asks for a Holdout audit of a frozen multi-candidate set. No existing protocol explicitly authorizing that exact candidate-set audit was identified. Holdout was not accessed and Stage-B was not started.
2. **Checkpoint metadata and candidate freeze are unverified.** The bounded directory metadata read stalled and was interrupted. The 60k Seen-Dev evaluation exists, but the immutable checkpoint was not verified and the frozen candidate table was not completed. No checkpoint SHA was computed.

## Not run

- Stage-A Holdout audit: not run
- Stage-B arms B1–B6: not run
- Final Holdout audit: not run

The run summary reported `locked_final_accessed=false` and `codabench_accessed=false`. Private/locked-final data and Codabench were not accessed. The two-minute monitor automation was paused at this gate.

**Review requested:** clarify whether an existing approved single-model audit protocol should govern this campaign.