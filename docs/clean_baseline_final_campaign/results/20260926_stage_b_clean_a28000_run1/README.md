# RealPDE Clean A@28k → Stage-B transfer test

Status: REVIEW_REQUIRED
Training: completed at the requested hard stop, B@1000.
Execution commit: b43f9f90a36f3276b6ea716a85b5b29ae780243d
GPU: NVIDIA GeForce RTX 5070 (12 GB)
Training wall time: 924.0 seconds, excluding B@0 preflight.

## Provenance and fixed recipe

Parent checkpoint: Clean Stage-A @28000, transferred from 3090 to 5070. The source and destination SHA-256 matched: e49aa546653007835f31e690bbf91d28a8a0f60807d7847978c22fdabd666b1d. Checkpoint metadata was verified as stage A, scope clean, iteration 28000, with 132 optimizer-state tensors.

The existing Stage-B runner was used without training-code changes. Recipe: P00-only Dense-All, carried AdamW state, LR 3e-6, batch 8, FP32, seed 41; base losses mse 1.0 / tke 0.05 / rel 0.027514 / mvpe 0.009757; vorticity lambda 15.5385751724; extra Stage-B Rel 0.027514. Evaluations were at 0, 500 and 1000. Training stopped at 1000.

Data boundary: Train51 and Seen-Dev12 only. No holdout, locked-final, private data or Codabench access.

## Seen-Dev results

| Update | Rel-L2 | TKE | MVPE | Point | H19 sq. err. | H20 sq. err. | H19+H20 sq. err. |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.104709 | 0.480161 | 0.077965 | 90.637652 | 1050.165 | 1474.419 | 2524.584 |
| 500 | 0.102250 | 0.468020 | 0.074695 | **90.857492** | 1043.183 | 1456.683 | 2499.866 |
| 1000 | **0.102298** | 0.475233 | **0.077239** | 90.738706 | 996.666 | 1358.732 | **2355.398** |

H19 / H20 / H19+H20 squared-error deltas vs B@0:

| Update | H19 Δ | H20 Δ | H19+H20 Δ |
|---:|---:|---:|---:|
| 500 | -6.982 | -17.736 | -24.718 |
| 1000 | -53.499 | -115.687 | -169.186 |

Best point checkpoint: B@500, SHA-256 ff169727fb37f88915d00ee6a3bb307a9d27a8d8dd701a376b934eb479cbb2cc.
Best tail checkpoint (minimum Seen-Dev H19+H20 squared-error sum among B@0/500/1000): B@1000, SHA-256 36ad1f0ea392b30a8c453db9a8a29fd20bc42b902ec4fbf744ac139ccc556b1e.

Checkpoint binaries remain on 5070 at /hy-tmp/realpde_runs/sota_merge_clean_stage_b_a28000_20260925_run1/checkpoints/ and are not included here.

## Evidence files

- run_config.json and summary.json
- aggregate_metrics.csv
- stage_b_curve.csv (logged training losses plus fixed LR and sampler position)
- stage_b_horizon_curve.csv (horizon squared errors and deltas)
- eval_000000, eval_000500, eval_001000
- checkpoint_provenance.json (checkpoint paths, sizes and hashes)
- raw_training.log
- post_train_diagnostics/horizon_delta_vs_b0.csv
- sota_merge_clean_stage_b_a15000_20260925_run1_handoff.json (execution handoff and safety record)
