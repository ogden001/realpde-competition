# RW-MA / RW-MB Matched Random Phase Review

Status: `RUN_PENDING`; final scientific status will remain `REVIEW_REQUIRED` pending ChatGPT/Sol review.

## Frozen A/B

- RW-MA: Random Phase sampler/global-shuffle path, every trajectory forced to phase 0.
- RW-MB: identical runner/sampler/global-shuffle path, independently sampled phase 0–19 per trajectory and epoch.
- Shared: 50 Train / 16 Dev, P0-A CNO, N2, `sim_pretrain/sim_cno.pth`, seed `20260901`, AdamW `1e-5`, batch 8, workers 2, 3000 updates, evaluations at 1000/2000/3000.
- Dev remains fixed `start=0, stride=20`; official scorer unchanged.

RW-MB may stop only at @1000 for the registered severe-negative gate: Rel-L2 and MVPE both >10% worse, all three metrics >10% worse, or TKE >20% worse. No legacy RW-00 metric is used for this gate.

The @3000 TKE “obvious compensation” language has no numerical threshold. The report will preserve its raw deltas and label it `REVIEW_REQUIRED` rather than inventing a quantitative interpretation.
