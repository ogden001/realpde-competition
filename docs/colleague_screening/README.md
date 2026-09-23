# Colleague 80pt Incremental Screening

Status: `IMPLEMENT_AND_EXECUTE_AUTHORIZED / COMPETITION_ORIENTED`.

This direction screens three bounded changes on the colleague 80-point Track 1
pipeline. It is an `OFFICIAL_WARM_START` competition experiment. The archived
point predictor and all new continuation updates use all 81 usable trajectories;
the Dev16 gate is reconstructed from the same sorted-filename, seed-41 split.
Dev16 is therefore a subset of training and this protocol also includes files
reserved by the separate clean-research split. Its output is only comparable to
the colleague's historical screening evidence, never a clean generalization claim.

The frozen research variables are:

1. Residual TKE loss rebalancing: `0.06` control versus `0.09` and `0.12`.
2. Fixed temporal phase versus one random phase in `[0, 19]` per trajectory and
   sampler epoch, with identical global shuffle behavior.
3. Original uncertainty-head inputs versus the same inputs plus residual
   `delta_u` and `delta_v`.

Residual arms start from one verified residual-corrector checkpoint, reset the
optimizer and scheduler identically, run 5,000 updates, and evaluate at update
0 and every 1,000 updates.  The primary decision point is update 5,000.  The
four arms share one control:

| arm | TKE weight | temporal sampling |
|---|---:|---|
| R0 | 0.06 | fixed phase 0 |
| R1 | 0.09 | fixed phase 0 |
| R2 | 0.12 | fixed phase 0 |
| R3 | 0.06 | random phase per trajectory and epoch |

Only released train-real trajectories are in scope. Private external data,
Codabench, OSS upload, simulator data, checkpoint deletion, and automatic long
training are outside this screening task.


## Default post-train diagnostics

Every residual training arm now writes the standard diagnostic bundle under
diagnostics/ after its final dev evaluation. This includes Future1-Future20,
per-trajectory, trajectory-by-horizon, spatial maps, mean/fluctuation
decomposition, and residual before/after evidence. See
docs/POST_TRAIN_DIAGNOSTICS.md. This is the default for future point-prediction
training experiments, not an optional follow-up.

## 2026-09-23 matched residual trajectory-sampling screen

Status: `REVIEW_REQUIRED`; no automatic GO/NO-GO or long follow-up.

Evidence: `docs/colleague_screening/results/20260923_trajectory_sampling_screen`.
Execution commit: `6b7eab5f0e8410268d5e112ac02a8052921ffa41`.

Both arms used the same all81 / Dev16-overlap data, frozen all81 CNO, exact
bitwise-equal fresh zero-init residuals (seed 41), FP32, and 5,004 updates.
The fixed-stride control reproduced 3,341 stride-20 windows and consumed
3,336 samples per epoch; both arms consumed 40,032 samples over 12 epochs with
identical per-trajectory draw counts. The stratified random-start arm had zero
duplicate-trajectory batches and consumed 40,032 unique windows versus 3,341
for control. No private/locked-final data or Codabench was accessed, and no
follow-up training was started.

Raw Dev16-overlap metrics at update 5,004 (for Sol's review, not a scientific
verdict):

| arm | Rel-L2 | TKE | MVPE |
|---|---:|---:|---:|
| A fixed stride-20 | 0.08404702 | 0.48352758 | 0.07628248 |
| B stratified random-start | 0.08415310 | 0.48843803 | 0.07654543 |

## 2026-09-23 late-horizon backbone campaign

Status: `REVIEW_REQUIRED`; no automatic GO/NO-GO or follow-up was started.

Evidence: `docs/colleague_screening/results/20260923_late_horizon_backbone_campaign`.
Execution commit: `6fa2f13d7db2e81e4c913df35fe379007953d185`.

The same 640 Dev16 stride-20 windows were replayed across the four specified
backbones without training. The matched continuation then ran Control and
Ramp for exactly 5,000 updates each from the same audited @53582 checkpoint,
with the same restored optimizer, Dense-All data order, seed, and Stage-B
objective. Ramp changed only the N2 velocity-MSE horizon weights (normalized
linear F1=1 to F20=2). Both arms have evaluations at 0/1k/2k/3k/4k/5k.

At update 5,000, Control vs Ramp Dev metrics were: Rel-L2 `0.10138615` vs
`0.10130069`, TKE `0.41880146` vs `0.41889709`, MVPE `0.07759760` vs
`0.07762400`, and F20 Rel `0.17830724` vs `0.17716208`. The archive contains
the full horizon, per-trajectory, and matched-comparison evidence for Sol's
review; these figures are not a GO/NO-GO determination. No >5k continuation,
residual or uncertainty training, Codabench, or locked-final/private access
was performed.
