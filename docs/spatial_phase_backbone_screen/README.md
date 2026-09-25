# REALPDE Spatial Phase Backbone Screen V1

Status: `READY_FOR_EXECUTION / REVIEW_REQUIRED`

## Purpose

This experiment answers one narrow question:

> Starting from the same historical Clean Baseline Stage-1 CNO initialization, does using alternate spatial downsampling phases during backbone training improve the official P00 Seen-Dev result?

This is a clean Stage-1 backbone experiment. It is **not** the earlier residual-continuation screen.

## Historical control

Control is the completed Clean Baseline V1 Stage-1 run:

- execution commit: `65e4f0f1d029eea47c6ba96364ce889e22d437d8`
- same official `sim_real_cno.pth` initialization
- Train51 / Seen-Dev12
- batch 8
- LR `1e-4`
- cosine scheduler
- seed 41
- 8,723 updates
- original P00-only 64x128 -> 32x64 sampling

Frozen matched control metrics:

| Update | Rel-L2 | TKE | MVPE |
|---:|---:|---:|---:|
| 8,000 | 0.099606201 | 0.778031290 | 0.080289602 |
| 8,723 | 0.099932298 | 0.792903721 | 0.080299616 |

Source: `docs/clean_baseline_v1/results/20260924_run1/RUN_REPORT.md`.

The control is **not rerun** in this screen.

## Candidate

The candidate reuses the existing trainer:

`tools/colleague_80pt/train_clean_baseline_cno.py`

The trainer architecture, loss, optimizer, scheduler, data split, temporal windows, batch size, LR, seed and training budget remain unchanged.

The only scientific change is training spatial sampling:

- P00: 50%
- P01/P10/P11: the remaining 50%, balanced
- phase assignment seed: `20260925`
- the same phase is used for Past20 and Future20
- Seen-Dev evaluation remains P00 only

Budget:

- updates: `8723`
- eval interval: `1000`
- batch: `8`
- LR: `1e-4`
- seed: `41`

## Decision

Primary matched comparison:

`Candidate @8000` vs `Historical Control @8000`

GO only if all are true:

- mean relative raw-error change across Rel-L2/TKE/MVPE <= `-0.50%`
- at least 2 of 3 raw metrics improve
- no single raw metric degrades by more than `1.00%`
- point score improves

The 8,723 comparison is diagnostic support only.

Candidate best checkpoint is recorded but is **not** used for the gate. This avoids repeating the previous best-vs-best / step-0 ambiguity.

## Hard boundary

This screen does not train Residual, does not change the Strong Backbone recipe, does not use AoA augmentation, temporal random-phase changes, loss changes, SPS, Holdout, locked-final/private, Codabench, full-data refit, packaging or submission.

A GO authorizes only a Sol review. It does not authorize automatic long training.
