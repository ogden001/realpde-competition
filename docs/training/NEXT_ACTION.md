# NEXT_ACTION

## Goal
Run the matched Random Phase A/B: RW-MA forced phase 0 versus RW-MB independently random phase, both with the same sampler/global-shuffle path.

## Tasks
1. Run RW-MA for 3000 updates with evaluations at 1000/2000/3000.
2. Run RW-MB from the same initial checkpoint and execution commit, applying only the registered @1000 severe-negative gate.
3. Produce matched metrics, case-level wins, phase/window audits, review logs, artifact manifests and a review README.

## Constraints
- Frozen 50 Train / 16 Dev; P0-A CNO, N2, `sim_pretrain/sim_cno.pth`, seed `20260901`, AdamW `1e-5`, batch 8, workers 2.
- Dev is fixed `start=0, stride=20`; official scorer is unchanged.
- Only variable: per-trajectory train-window phase. No legacy RW-00 gate, locked-final, Codabench, full-data, Random Start, stride/phase sweep, mixed modes or further experiment.
- At @1000, stop RW-MB only for Rel-L2 and MVPE both >10% worse, all three >10% worse, or TKE >20% worse.

## Deliverables
- Implementation/execution commit; A/B metadata, audit, update and trajectory comparisons, review logs and manifests.
- `REVIEW_REQUIRED` result package for ChatGPT/Sol.

## Stop
After the 3000-update matched A/B or the registered severe early stop. Do not run 7500 automatically.
