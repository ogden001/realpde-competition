# Arm A Pareto-TKE continuation — 2026-09-23

## Run record

- Status: completed; 6,000 additional updates, evaluated every 1,000 updates.
- Resume: A@12000 `model_best.pth`, SHA256 `b690838b7e87647ae9667e84f446969730d4592936b10b9f70f42abf525194b4`.
- Frozen backbone: Stage-1 CNO, SHA256 `ff28aaf0114d57e320e5ea9cf9945d874a7482f22e3f0aeaed5ebbe286573f8a`.
- Configuration: batch 8; AdamW initial LR `5e-5`, weight decay `1e-5`; cosine schedule over this 6,000-update continuation; eval interval 1,000; hidden 96; blocks 2; max delta 0.04; train alpha 1.0; TKE weight 0.12; `project_tke`.
- Sampling/data: seed 41; fixed phase sampling; stride 20; all 81 train trajectories / 3,260 windows; same 16-trajectory, 640-window Dev split and overlap policy as original A. Other losses, bounds, and runtime evaluation setting were retained.
- Execution checkout: `5582c65f50134a1249d7dd5bab4f217a1481e988`; trainer `tools/colleague_80pt/residual_multi.py` SHA256 `3bfebd4f61dc97ce295095a98d6273b35e1f401622c9a894070d67cb1ae78cab`.
- Run directory: `/hy-tmp/realpde_runs/colleague80_v2_arm_a_pareto_tke_continuation_20260923/A_pareto_tke`.
- Validation used 640 windows at every listed milestone. Values below are the selected alpha=1.0 result.

## Milestones

| Continuation updates | Rel-L2 | TKE | MVPE | local `final_est` |
|---:|---:|---:|---:|---:|
| A@12000 baseline | 0.0811836861 | 0.4265229106 | 0.0709700336 | 81.99803773 |
| 1,000 | 0.0814745264 | 0.4301182583 | 0.0716462281 | 81.93643237 |
| 2,000 | 0.0816919165 | 0.4272869371 | 0.0715942890 | 81.95099508 |
| 3,000 | 0.0815549901 | 0.4242315024 | 0.0713481862 | 81.99265938 |
| 4,000 | 0.0812169243 | 0.4236618467 | 0.0708568213 | 82.01605707 |
| 5,000 | 0.0810724087 | 0.4238787174 | 0.0707507215 | 82.02485685 |
| 6,000 | 0.0810459096 | 0.4235429764 | 0.0707221480 | 82.02973404 |

## Best checkpoint and comparisons

- Best step: continuation step 6,000 (cumulative A step 18,000); best `final_est` 82.02973404.
- Best checkpoint: `/hy-tmp/realpde_runs/colleague80_v2_arm_a_pareto_tke_continuation_20260923/A_pareto_tke/model_best.pth`.
- Best checkpoint SHA256: `852b41cda7d15ce9b205aecb32e206712e7a036b18981a62a9afc4e9a0b9a238`.
- Relative to A@12000: Rel-L2 −0.170%, TKE −0.699%, MVPE −0.349%; local `final_est` +0.031696 points.
- Relative to current80 raw Dev baseline (Rel-L2 0.0804204196, TKE 0.4491582513, MVPE 0.0711169168): Rel-L2 +0.778%, TKE −5.703%, MVPE −0.555%.
- Target Rel-L2 ≤0.08082: **not reached** (best/final 0.08104591).
- Mechanical gate against current80: **FAIL**. TKE improvement ≥3%: pass; Rel-L2 degradation ≤0.5%: fail; MVPE degradation ≤0.3%: pass.

Only this lightweight evidence is versioned; no checkpoint binary or training data is included.
