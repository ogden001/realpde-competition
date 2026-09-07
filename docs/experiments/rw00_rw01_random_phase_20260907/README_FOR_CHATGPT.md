# RW-00 / RW-01 Random Phase Review

Status: `INVALID_COMPARISON / SHUFFLE_CONFOUND`.

Provenance: core implementation `b61706c28ef778f9524ab337860807953e14d509`; launcher fix `aaa5e71baa1ac8bbb0ca37191a20fe3f5b27e8c1`; old results `28a235d9dacb60ab7f66297c73a2598bebe19c25`.

Protocol: frozen 50 Train / 16 Dev, P0-A CNO, N2, seed `20260901`, AdamW `1e-5`, batch 8, workers 2, shared `sim_pretrain/sim_cno.pth`, milestone evaluations at 1000/2000/3000/5000/7500. Dev stayed fixed `start=0, stride=20`.

## Metrics

| update | RW-00 Rel-L2 | RW-01 Rel-L2 | improvement | RW-00 TKE | RW-01 TKE | improvement | RW-00 MVPE | RW-01 MVPE | improvement |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.200521 | 0.311556 | -55.38% | 0.658004 | 1.551016 | -135.71% | 0.166719 | 0.290040 | -73.96% |
| 2000 | 0.169085 | 0.269374 | -59.31% | 0.586997 | 1.101652 | -87.68% | 0.137881 | 0.192667 | -39.74% |
| 3000 | 0.160694 | 0.287604 | -78.98% | 0.559630 | 1.051449 | -87.88% | 0.134048 | 0.256457 | -91.32% |
| 5000 | 0.152474 | — | — | 0.546525 | — | — | 0.119312 | — | — |
| 7500 | 0.149218 | — | — | 0.528597 | — | — | 0.118325 | — | — |

RW-01 was stopped at @3000 because Rel-L2 and MVPE degradation both exceeded 3%; TKE also degraded strongly. @7500 RW-01 was therefore intentionally not run. These numbers are retained as historical execution evidence only: RW-00 used global `shuffle=True`, while old RW-01 emitted sampler indices in trajectory/time blocks. They cannot support an A/B conclusion about phase randomization.

Trajectory wins at the formal available point @3000: Rel-L2 `0/16`, TKE `0/16`, MVPE `0/16`.

Window audit: 50 trajectories × 12 epochs; phase counts are recorded in `RW-01_random_phase/window_audit_summary.json`; `invalid_window_count=0`, `non_stride20_count=0`, window counts per trajectory `12–42`.

The requested trend cannot be assessed from this run. A replacement with a deterministic global shuffle after phase selection is required before review.

The replacement control is recorded in [RW-01 shuffle revalidation](../rw01_shuffle_revalidation_20260907/README_FOR_CHATGPT.md). Its matched phase-0 control triggered `STOP_REVIEW_REQUIRED` on MVPE, so it did not proceed to a new RW-01 comparison.
