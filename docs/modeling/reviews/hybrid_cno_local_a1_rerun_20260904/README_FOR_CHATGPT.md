# A1 Valid Rerun Review Evidence

- Experiment: `hybrid_cno_local_a1_rerun_20260904`
- Experiment ID: `T1-ID-HYBRID-CNO-LOCAL-A1-RERUN-S20260904`
- Final status: `REVIEW_REQUIRED`

This directory contains lightweight evidence for independent Sol review of the valid optimizer-fixed rerun. It includes official Rel-L2/TKE/MVPE metrics, 16-trajectory case evidence, horizon analysis, local residual statistics, convergence, runtime, preflight optimizer audit, representative spatial maps, and provenance.

The rerun used the unchanged matched A1 architecture/protocol after fixing the optimizer wiring: Direct@1500 initialization, P0-A global CNO plus raw Past20 u/v local residual branch, N2 loss, AdamW `1e-5`, batch 8, seed `20260901`, total@3000.

No KEEP/PARK/STOP research conclusion is made here. Sol should independently review the evidence and decide.

Excluded by design: checkpoint files, full Dev predictions, raw tensors, training caches, original data, and large NPZ/NPY files.

