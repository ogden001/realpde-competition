# Clean held-out AoA=10° benchmark — results analysis

## Current progress

- The remote campaign completed both arms: CNO 8,723 updates and residual 20,000 updates.
- Inner Dev checkpoint comparison completed before the final Heldout10 evaluation.
- The Heldout10 evaluation used 14 real PIV trajectories at AoA=10° only; no post-heldout tuning or rerun was performed.
- The whitelist archive contains the split manifests, bridge audit, training review logs, residual evaluations at steps 2,500 through 20,000, and unified diagnostics for all six checkpoints.

## Interpretation

The frozen protocol is satisfied, but this experiment does not show a benefit from the 5°↔15° same-Re mean-field bridge augmentation. Using the final 20,000-update checkpoints, Aug is slightly worse than Control on every listed primary diagnostic in both Inner Dev and Heldout10. The Heldout10 comparison is the relevant clean generalization result:

| Metric | Control final | Aug final | Aug vs Control |
|---|---:|---:|---:|
| Rel-L2 raw | 0.084960483 | 0.085416079 | +0.5362% |
| TKE raw | 0.523943663 | 0.535470963 | +2.2001% |
| MVPE raw | 0.098180927 | 0.098620839 | +0.4481% |
| Mean-field Rel-L2 | 0.051576575 | 0.051754942 | +0.3458% |
| Fluctuation Rel-L2 | 0.785081260 | 0.789568637 | +0.5716% |

The Inner Dev final-checkpoint deltas are smaller but have the same direction: Rel-L2 +0.2936%, TKE +0.5276%, MVPE +0.2469%, mean-field Rel-L2 +0.4261%, and fluctuation Rel-L2 +0.2331% for Aug relative to Control. Both arms selected their 17,500-update checkpoints as `best` on Inner Dev; the Heldout10 result was evaluated only after that selection.

This is an observation of this frozen experiment, not a basis for changing parameters or rerunning. It should not be interpreted as a leaderboard submission result.

## Evidence

- Campaign and protocol gates: `campaign_manifest.json`, `heldout_aoa_split_audit.json`
- Train/Inner Dev and Heldout10 manifests: `train_innerdev_manifest.json`, `heldout_eval_manifest.json`
- Inner Dev comparison: `inner_dev_comparison/comparison_summary.csv`
- Heldout10 comparison: `heldout10_comparison/comparison_summary.csv`
- Augmentation audit: `aug/cno_stage1.aoa_audit.json`, `aug/residual/aoa_augmentation_audit.json`
- 5k matched residual evidence: `control/residual/eval_step_05000.json`, `aug/residual/eval_step_05000.json`
