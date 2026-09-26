# Serial Challenger V1 — review package

Status: `REVIEW_REQUIRED`
Run: `/hy-tmp/realpde_runs/serial_challenger_v1_20260926`
Protocol: `REALPDE_SOTA_MERGE_SERIAL_UNATTENDED_V1`

This directory contains compact run evidence for independent review. The run completed with state `DONE`. It records metric outputs and checkpoint identities; checkpoint files and training data are intentionally not included.

## Protocol and provenance

- Starting point: clean Stage-A update 57,000; SHA256 `1bdde38d902c2bec516570cadb8703797457c7db9c3fda67eb9a911021d1b873`.
- Execution source commit: `87d25f4e322b50e49c97ce1840e6b91a6c6d0566` (GitHub `main` at run review time).
- Verified source asset bundle: `realpde_serial_assets_20260926.tar`, SHA256 `83dc5e8c016e5ab78c9a64e73faa82a6737f168b6f51c4ae27b8423287b457da`.
- Stage-B: 6,000 updates, P00 only, learning rate `3e-6`, carried Stage-A optimizer, Seen-Dev12 selection, evaluations/saves every 500 updates.
- Residual: selected Stage-B backbone frozen; hidden width 96, 2 blocks, max delta 0.04, AdamW (`lr=2e-4`, `weight_decay=1e-5`), 22,000 updates, alpha 1.0, existing Serial Strong-Backbone loss, Seen-Dev12 selection.
- Split: Train51 / Seen-Dev12 / AoA10 18 trajectories. AoA10 is audit-only; both selection records state `aoa_used_for_selection: false`.
- Scope: no SPS, full-data run, locked-final/private access, or Codabench access. No submission was made.
- Final state: `DONE`, status `REVIEW_REQUIRED`; no downstream run was started.

## Selected checkpoints and final metrics

| Evaluation | Selected update | SHA256 | Rel-L2 | TKE | MVPE | point score |
|---|---:|---|---:|---:|---:|---:|
| Stage-B, Seen-Dev12 | 6,000 | `d158fe3c426ae42c9488853378432646aec57f3cbd9f9b2a0bbe140ab66df06d` | 0.09523930 | 0.46676627 | 0.06840461 | 91.07505934 |
| Residual, Seen-Dev12 | 18,000 | `574220df3b50bd023690911930111f0893f3ee6aa86b88661824595a83f1eeb5` | 0.07868838 | 0.49113336 | 0.06343374 | 91.14169227 |
| Residual, AoA10 audit | 18,000 | same selected residual checkpoint | 0.08650679 | 0.51539314 | 0.09658930 | 90.25248346 |

Final primary metrics include 491 Seen-Dev windows / 12 trajectories and 756 AoA10 windows / 18 trajectories. The AoA10 point score is reported as an audit metric only, not a selection signal. The official run report retains the selected aggregate values and provenance.

## Evidence files

- `FINAL_REPORT.json`: final state, selected checkpoints, hashes, final Seen/AoA metrics, and scope flags.
- `status.json`: terminal run status.
- `stage_b/aggregate_metrics.csv`, `stage_b/summary.json`: Stage-B Seen-Dev milestone curve and selection summary.
- `stage_b_aoa_curve/curve.csv`: Stage-B AoA10 audit milestones.
- `residual/summary.json`: Residual Seen-Dev milestone history, selection rule, and selected update.
- `residual_aoa_curve/curve.csv`: Residual AoA10 audit milestones. It contains raw metrics and does not include a point-score column.
- `final_seen_selected/` and `final_aoa_selected/`: final primary metrics and by-horizon summaries for selected residual checkpoint.
- `selected_stage_b.json`, `selected_residual.json`: checkpoint selection split and SHA256 records.

The metric/evidence files were copied from the completed run without recomputing metrics; CSV line endings were normalized to LF for the repository. They include remote run paths as recorded by the runner. Model checkpoints, H5 data, logs, per-trajectory prediction arrays, and large trajectory-level tables are omitted.

## Engineering compatibility change

`tools/colleague_80pt/residual_multi.py` now passes `weights_only=False` to `torch.load` for the trusted local recovery checkpoint consumed by the residual evaluator. This addresses PyTorch 2.9's changed default for trusted checkpoint loading. It does not change model structure, training data, optimizer, loss, or scientific parameters.

## Review state

This is evidence for Sol's review, not a scientific approval or promotion. `FINAL_REPORT.json` marks the result `REVIEW_REQUIRED` and requests a manual comparison with Joint V1 before any further action.
