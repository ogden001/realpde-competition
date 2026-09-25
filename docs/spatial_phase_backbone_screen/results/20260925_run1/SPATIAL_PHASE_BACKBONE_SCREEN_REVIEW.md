# REALPDE Spatial Phase Backbone Screen V1

Status: `REVIEW_REQUIRED`

Historical Clean Baseline Stage-1 is the frozen control. The candidate
reuses the original Stage-1 trainer and changes only training spatial sampling phase.

- Gate: **GO**
- Baseline @8000: `{'rel_l2_raw': 0.099606201, 'tke_raw': 0.77803129, 'mvpe_raw': 0.080289602}`
- Candidate @8000: `{'rel_l2_raw': 0.09793073683977127, 'tke_raw': 0.7783641815185547, 'mvpe_raw': 0.07919468730688095}`
- @8000 relative deltas: `{'rel_l2_raw': -1.6820882067660974, 'tke_raw': 0.04278639211986519, 'mvpe_raw': -1.3637067140014603}`
- @8000 mean raw-error delta: `-1.0010%`
- @8000 point-score delta: `0.039359`
- Baseline @8723: `{'rel_l2_raw': 0.099932298, 'tke_raw': 0.792903721, 'mvpe_raw': 0.080299616}`
- Candidate @8723: `{'rel_l2_raw': 0.09793958067893982, 'tke_raw': 0.7701005935668945, 'mvpe_raw': 0.07866805791854858}`
- @8723 relative deltas: `{'rel_l2_raw': -1.9940673445337795, 'tke_raw': -2.8759011755357142, 'mvpe_raw': -2.031837962277952}`
- Candidate assigned spatial phase: `{'P00': 0.5000121015562601, 'P01': 0.16666263281457996, 'P10': 0.16666263281457996, 'P11': 0.16666263281457996}`
- Candidate best iteration (diagnostic only): `8723`

Decision is based on matched update 8000, not best-vs-best checkpoint selection.
STOP here. No residual or follow-up training is authorized.
