# CLEAN_BASELINE_FINAL_CAMPAIGN

Status: `READY_FOR_EXECUTION / REVIEW_REQUIRED`

Deadline target: 2026-09-28 final submission.

This campaign is the final clean re-validation funnel. It contains exactly three experiments and must not grow new research branches.

## Frozen research anchor

Reference: `REALPDE_CLEAN_BASELINE_V1`.

- Train51: AoA 0/5/15/20
- Seen-Dev12: AoA 0/5/15/20, 3 each
- unseen-AoA10 Holdout18: all available AoA=10, holdout-only
- Train stride=1 for point-model training unless an SPS-specific head protocol says otherwise
- Dev/Holdout stride=20, start=0
- Past20 -> Future20, sub_sample=2
- seed=41, batch=8 for point-model training
- Clean Stage1 best SHA256: `6aec4edb5e42746080e19baaafa2d6b719bb57befb1ff9035ece7889ec280036`
- Clean Stage2 best SHA256: `1536fe02e11fc0dc991a38cb1489d0be0752ec8149be7b9f1542a4cb81c2f975`

Clean Seen-Dev best: Rel-L2 `0.078363446`, TKE `0.507731898`, MVPE `0.064407949`.

The 22k matched residual control used for bounded comparisons is frozen evidence at `docs/clean_baseline_v1/results/20260924_run1/evidence/stage2_residual/eval_step_22000.json`.

## Experiment 1 - Strong Backbone Clean

Candidate recipe is frozen as one unit: `Dense-All + P0-A + MF-CNO + N2 + vorticity + Stage-B extra Rel`.

Old all82/full-data strong-backbone weights are forbidden as initialization. Candidate starts from official `sim_real_cno.pth`.

Full historical convergence schedule is retained: Stage A 30,000 updates at LR `1e-5`; Stage B 5,000 updates at LR `3e-6`; effective batch 8; seed 41; point-score checkpoint selection only. After clean strong-backbone selection, train the exact clean-baseline residual recipe for 22,000 updates.

Primary gate versus clean baseline residual @22k: mean percentage change of Rel/TKE/MVPE raw error <= `-1.0%`; at least two metrics improve; no metric worsens > `1.0%`; TKE must not worsen. AoA10 is evaluated once only if Seen-Dev gate passes.

## Experiment 2 - Pareto-TKE Residual Clean

Matched two-arm experiment from the same clean Stage1 best checkpoint. Control uses TKE weight `0.06` with scalar backward. Candidate uses historical Pareto recipe TKE weight `0.12` with `project_tke` backward. Both use fresh zero-init h96/b2/max_delta=0.04, seed 41, same stride1 windows/order, optimizer/LR/batch/eval cadence, and exactly 22,000 updates.

Runner hard-checks identical step-0 model state and identical sampling audit. Selected control/Pareto checkpoints are interpolated without training only at beta `0.50 / 0.65 / 0.80 / 1.00`. No extra beta search.

Per-beta gate: TKE improves >=2.0%; Rel degradation <=0.5%; MVPE degradation <=0.5%; mean relative error change across three metrics not worse than control. Among passing betas select highest point score. AoA10 is evaluated once for the selected beta only.

## Experiment 3 - Residual-aware SPS Clean

This experiment is independent of Exp1/Exp2 so Codex can finish all three without an intermediate human gate. It validates SPS on frozen clean-baseline `base -> residual -> final`.

It reproduces successful colleague uncertainty semantics: head observes Past20 plus pre-residual base prediction features; target is post-residual final error; h64, 2 blocks, dropout0; masked logmae; Train51 stride5; SeenDev12 stride20; 5000 head updates, batch16, AdamW LR1e-3, warmup/cosine; point predictor frozen; original 300-combination calibration family only.

Primary gate: SeenDev adaptive SPS +1.0 over same-Dev best static interval; mean interval width <=1.20x static; point parity max abs <=1e-7. Calibration selection is **constrained optimization**: among the frozen 300 adaptive calibrations that satisfy mean width <=1.20x the same-Dev best static width, select the highest Seen-Dev SPS; then select the checkpoint by that constrained SPS. The unconstrained SPS maximum is diagnostic only and must not cause a post-hoc NO_GO if a width-feasible higher-than-static candidate exists. Selected Dev calibration is applied once to AoA10 without recalibration.

Review-only recovery for the completed 2026-09-24 Exp3 is implemented by `tools/recover_clean_exp3_sps.py`. It may reuse the already-saved head only when the constrained-optimal step matches the saved head step and its SHA256 matches the archived hash. Recovery performs zero optimizer steps, never modifies the source run, and accesses AoA10 once only after the corrected Seen-Dev gate is GO.

### Experiment 3 final reviewed result

Status: **`SCIENTIFIC GO / REVIEW_REQUIRED`**.

The corrected constrained selection chose step `4500`, head SHA256 `1cde13a23fe15547dc2ee16cede9edd89fe8c12bc88f8e9ad059641f0a6c90b8`, with calibration `floor=0.0025, mult_u=0.75, mult_v=1.5, rel=0.0075`.

Seen-Dev12: static SPS `48.416360` -> adaptive SPS `51.654435` (`+3.238075`); coverage `0.788302 -> 0.849634`; width ratio `1.192600`; point parity `0`; gate `GO`.

One frozen unseen-AoA10 Holdout18 evaluation, without recalibration or holdout-based selection: static SPS `45.842623` -> adaptive SPS `49.017414` (`+3.174791`); coverage `0.772295 -> 0.837020` (`+6.47` percentage points); adaptive mean u/v width `0.015131400`.

Conclusion: the SPS improvement transfers almost unchanged from Seen-Dev to fully unseen AoA10. Exp3 is therefore accepted as a validated SOTA-merge method. Do not reopen SPS grid research. Exact head/calibration reuse is valid only when the final point predictor is the same frozen Clean Stage1+Stage2 predictor; if the final point predictor changes, reuse the validated SPS recipe/protocol with a matched frozen-predictor head, not the old head weights by assumption.

Evidence: `docs/clean_baseline_final_campaign/results/20260925_exp3_constrained_recovery/`.

## Hard campaign boundaries

Forbidden: Random Phase/random-start, AoA augmentation, ordinary TKE-weight sweep, late-horizon expansion, end-to-end joint fine-tuning, new hand-crafted features, new backbone-family sweep, full-data refit, submission/package construction, Codabench, locked-final/private, automatic follow-up experiments.

All results remain `REVIEW_REQUIRED`. Codex produces evidence; Sol makes the scientific merge decision.
