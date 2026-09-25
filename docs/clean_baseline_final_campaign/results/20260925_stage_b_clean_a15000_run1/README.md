# RealPDE Clean Stage-B A@15k — Run 1

**Status:** REVIEW_REQUIRED
**Training status:** completed at update 2000; hard stop reached.
**Review state:** local checkpoint artifacts remain on the 5070 host; checkpoints are intentionally excluded from Git. See checkpoint_provenance.json for paths and SHA-256 hashes.

## Protocol

- Execution commit: 386920e532e54ba1cbf42c93eb33c5ca165c0cd6
- GPU: NVIDIA GeForce RTX 5070 (12 GB)
- Source: clean Stage-A update 15000; SHA-256 77843313b411e70ef1471a59854f741ed7eddebb68c7d090286a39da78777dcc
- Optimizer: AdamW state carried from Stage A; Stage-B LR 3e-6
- Sampling: P00-only Dense-All; batch size 8; FP32; seed 41
- Loss: mse 1.0, tke 0.05, rel 0.027514, mvpe 0.009757; vorticity lambda 15.5385751724; extra Stage-B Rel 0.027514 (effective Rel 0.055028)
- Budget: updates 0–2000; evaluations at 0/500/1000/1500/2000; checkpoints at 500/1000/1500/2000
- Data: Train51 and Seen-Dev12 only. No holdout, locked-final, private data, or Codabench access.
- Wall time: 1802.6 seconds (training process, excluding B@0 preflight)

The runner expects a JSON train/dev manifest. The manifest used on 5070 was generated from the extracted Train51/Seen-Dev12 data_manifest.tsv; SHA-256: b167dba4ab4d9afa6b908d7575ea93576c64b09635bfa5d4e54f28767864c223.

## Seen-Dev results

| Update | Rel-L2 | TKE | MVPE | Point |
|---:|---:|---:|---:|---:|
| 0 | 0.1112000644 | 0.4829474986 | 0.0823766142 | 90.442128 |
| 500 | 0.1080543175 | 0.4775302410 | 0.0806659684 | **90.574280** |
| 1000 | **0.1072569862** | 0.4814852774 | 0.0816950053 | 90.527520 |
| 1500 | 0.1078407615 | 0.4813242257 | 0.0810019672 | 90.531167 |
| 2000 | 0.1066798046 | 0.4832986891 | **0.0793942735** | 90.552003 |

B@0 reproduced the historical A@15k metrics (Rel-L2 0.111199, TKE 0.482945, MVPE 0.082377, point 90.442157) within a negligible difference, so training proceeded.

### Tail squared-error change vs B@0

| Update | H19 Δ | H20 Δ | H19+H20 Δ |
|---:|---:|---:|---:|
| 500 | +17.549 | -70.505 | -52.956 |
| 1000 | -30.741 | -170.242 | -200.983 |
| 1500 | +0.297 | -159.667 | -159.371 |
| 2000 | +30.590 | -129.119 | -98.529 |

Values are Seen-Dev squared-error sums; negative means lower error than B@0. Full horizon 1–20 values and deltas are in stage_b_horizon_curve.csv.

Best point checkpoint: update 500, point 90.5742796424, SHA-256 3c49d1779529050983096d332c39be8f522a7184a274d9310f57c0b6089b224c. All model checkpoints remain at /hy-tmp/realpde_runs/sota_merge_clean_stage_b_a15000_20260925_run1/checkpoints/ on the 5070 host.

## Files

- run_config.json: fixed run configuration and execution provenance
- aggregate_metrics.csv: official aggregate metrics at all five evaluation points
- stage_b_curve.csv: every 100-update training log record, with LR and derived sampler epoch/offset
- stage_b_horizon_curve.csv: aggregate metrics, per-horizon squared errors, and deltas vs B@0
- eval_*/: evaluation summaries and horizon summaries
- checkpoint_provenance.json: source and saved checkpoint hashes; checkpoint files are not in Git
- raw_training.log: full training stdout/stderr
- post_train_diagnostics/horizon_delta_vs_b0.csv: H19/H20 tail comparison
- sota_merge_clean_stage_b_a15000_20260925_run1_handoff.json: complete execution handoff and safety record

The training-time GitHub write-auth dry-run was unavailable. After completion, the review bundle was pushed to origin/main in commit dbd609f; no checkpoint binaries or datasets are included.
