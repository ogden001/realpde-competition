# Point LOCAL3 loss-gradient diagnostic handoff

Reference: `T1-ID-POINT-LOSS-GRAD-DIAG-20260902`
Execution state: **COMPLETED / TRAIN-ONLY LOSS DIAGNOSTIC**

## Protocol

- Existing LOCAL3 `last@1500.pt`; no retraining and no optimizer step.
- B3_PACKED, batch 8, seed `20260901`, fixed seeded-shuffled train-window order, 32 train batches.
- For each batch, MSE and weighted TKE (`0.05*TKE`) gradients were computed separately with `autograd.grad` over trainable parameters only. Gradients were flattened globally; no optimizer state or buffers were included.
- Dev, locked-final, scorer and Codabench were not accessed.

## Results (32 train batches)

| quantity | mean | median | p25 | p75 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| `grad_ratio = ||g_tke||/||g_mse||` | 57.949 | 44.764 | 33.222 | 59.379 | 19.821 | 189.385 |
| cosine(`g_mse`,`g_tke`) | -0.0853 | -0.0731 | -0.7305 | 0.5293 | -0.8963 | 0.8692 |
| raw MSE | 0.0004545 | 0.0004183 | 0.0003367 | 0.0005477 | 0.0001419 | 0.0009874 |
| raw TKE | 0.793055 | 0.799197 | 0.770361 | 0.825342 | 0.683439 | 0.847909 |
| weighted TKE (`0.05*TKE`) | 0.0396527 | 0.0399599 | 0.0385180 | 0.0412671 | 0.0341720 | 0.0423954 |

The mean of the per-batch scalar ratios `(0.05*TKE)/MSE` was `103.472×` (the ratio of the means is `87.25×`). This is separate from gradient dominance but points in the same direction. Cosine is mixed: median is mildly negative, with both strong conflicts and aligned batches; cosine is not used as a gate.

## Diagnostic label

`TKE_GRADIENT_STRONGLY_DOMINANT` because median gradient ratio is `44.764 >= 10`.

This supports the narrow hypothesis that the frozen `0.05*TKE` term can dominate LOCAL3 optimization despite its small nominal coefficient, and may help explain improved TKE alongside worse Rel-L2/MVPE. It does not establish causality or authorize changing the loss.

## Evidence

- Remote run: `/home/chyfuture/realpde_runs/point_loss_grad_diag_s20260901`
- Artifacts: `gradient_metrics.csv`, `summary.json`, `run_metadata.json`, `report.md`, `README_FOR_CHATGPT.md`
- Checkpoint SHA-256: `c1cc1995202a9f8d9a553eb177508c7f1ea95d048baf2663ce6094675f2ffa9b`
- Manifest SHA-256: `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`
- Diagnostic script SHA-256: `9d7cb89d41daabd859848af0c14e6a6d86b38fc0139f8fb1f176da6c303a30ea`

No balanced-loss training, LR/loss sweep, 7500 continuation, LOCAL5, dev evaluation, locked-final access or Codabench was performed.
