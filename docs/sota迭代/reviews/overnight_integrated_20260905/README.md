# Overnight Integrated Adaptive Probe — V4_INVALID_BASELINE / REVIEW_REQUIRED

Validation training execution commit: `d376d4ebb35ef0c3da95af7c3eb86419ab2c5f2d`.
Gate evaluator snapshot commit: `8797afa368cc5187ee1fd9b4f5fba99d819157eb`.
Corrected-head snapshot commit: `9bca78577dacde739f717c8e007b08ad9102ffef`.
Calibration evaluator snapshot commit: `ba3a822d16e00a72fefd3bf3bc3a3630dfe4a702`.

Status: `REVIEW_REQUIRED`.

The repaired frozen validation-family corrector, base uncertainty head, and
corrected uncertainty head completed with 2400, 1400, and 1400 updates on the
50/16 manifest. The official-v9 Gate passed: Rel-L2 improvement 23.7029%,
MVPE improvement 41.6318%, aggregate TKE improvement 17.3616%, and zero dev
trajectories with TKE degradation above 15%.

The fixed 56-row calibration grid is complete. Best base calibration is
floor=0.0025/mult=1, SPS 37.644685, coverage 0.832950. Best corrected
calibration is floor=0.0025/mult=1, SPS 44.145264, coverage 0.855313.

Complete raw logs, JSON/CSV evidence, and SHA256 provenance are retained in
this directory. The bounded baseline parity audit reproduced the historical
30900 metrics only with the checkpoint's canonical feature spacing, while the
v4 training metadata used doubled `dx/dy`; therefore this v4 corrector is
marked `V4_INVALID_BASELINE` and its gains are not promoted. No all-82 refit,
package, or Codabench submission was run.

Remote OUT_ROOT: `/home/chyfuture/realpde_runs/overnight_integrated_20260905_v4/`

Parity evidence: [BASELINE_PARITY_AUDIT.md](BASELINE_PARITY_AUDIT.md),
`baseline_parity_v2.json`, `trajectory_metrics_v2.csv`, and
`baseline_parity_audit_v2.review.log`.

The implementation and evidence commits were pushed to `main`; this is a
handoff for Sol review, not an automatic submission decision.
