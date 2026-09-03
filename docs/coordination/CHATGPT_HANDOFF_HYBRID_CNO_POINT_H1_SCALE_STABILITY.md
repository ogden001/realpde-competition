# H1 Scale Stability Audit Handoff

Reference: `T1-ID-HYBRID-CNO-POINT-H1-SCALE-STABILITY-S20260903`

## Result

Fixed replay of the saved dev predictions at `alpha=0.5`; no training,
optimizer step, alpha selection, checkpoint update, locked-final access, or
Codabench access. The aggregate replay exactly reproduces the prior H1 Scale
handoff values. Stability label: `H1_SCALE_STABILITY_SUPPORTIVE`.

Rel-L2 and MVPE each win on 16/16 trajectories. TKE wins on 2/16; its median
improvement is negative, so the aggregate TKE error worsening is broad rather
than driven by a small number of trajectories. Rel/MVPE/TKE joint counts are
16/16, 2/16, and 16/16 respectively for the requested conditions.

## Evidence

Small audit outputs are in `artifacts/hybrid_cno_point_h1_scale_stability_s20260903/`:
`trajectory_metrics.csv`, `stability_summary.json`, `bootstrap_summary.json`,
`run_metadata.json`, `report.md`, `README_FOR_CHATGPT.md`, and `status.json`.

Remote source replay: `/home/chyfuture/realpde_runs/hybrid_cno_point_h1_scale_s20260902/artifacts_parent/artifacts/dev_replay.npz`.
The audit ran in `realpde-pytorch-h5py:0831` on `gpu` using the official v9 scorer.

`NEXT_ACTION = REVIEW_REQUIRED`
