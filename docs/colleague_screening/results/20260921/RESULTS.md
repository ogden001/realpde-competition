# Colleague 80pt incremental screening results

Run ID: `full_ef9c54f`<br>
Execution commit: `ef9c54f621efcb82703cb2de40ce979c40df3c6a`<br>
Status: completed on the cloud GPU; no long training or Codabench submission was started.

This is the colleague-aligned competition protocol: all 81 usable real trajectories for training, the deterministic seed-41 Dev16 split, and the archived colleague residual checkpoint. Dev16 overlaps the training pool by design, so these numbers are comparable to the colleague screen but are not clean generalization estimates.

## Residual screen

| Arm | Change | Rel-L2 raw | TKE raw | MVPE raw | Fixed-time estimate | Gate |
|---|---|---:|---:|---:|---:|---|
| R0 | TKE 0.06, fixed phase | 0.080241 | 0.450366 | 0.071028 | 81.8716 | control |
| R1 | TKE 0.09, fixed phase | 0.081179 | 0.435177 | 0.071261 | 81.9333 | FAIL |
| R2 | TKE 0.12, fixed phase | 0.082059 | 0.425257 | 0.071522 | 81.9420 | FAIL |
| R3 | TKE 0.06, random phase | 0.080721 | 0.463794 | 0.072026 | 81.7376 | FAIL |

Relative changes and fixed-time estimate deltas are computed by the runner against R0:

| Arm | ΔRel-L2 | ΔTKE | ΔMVPE | Δestimate |
|---|---:|---:|---:|---:|
| R1 | +1.17% | -3.37% | +0.33% | +0.0617 |
| R2 | +2.27% | -5.58% | +0.70% | +0.0703 |
| R3 | +0.60% | +2.98% | +1.40% | -0.1341 |

Interpretation: increasing the TKE weight improved TKE, but the Rel-L2 guard failed (R1: +1.17%; R2: +2.27%). R2 has the highest fixed-time proxy, but it does not satisfy the pre-registered promotion gate. Random phase failed all three R3 gates at the primary step.

## Uncertainty-head screen

| Head | Inputs | SPS | Coverage | Gate |
|---|---|---:|---:|---|
| H0 | Original features | 51.6022 | 0.8896 | control |
| H1 | Original features + residual delta_u/delta_v | 51.6310 | 0.8904 | FAIL |

H1 SPS delta is **+0.0288**, below the +0.5 promotion threshold. The head screen used one frozen cache from the archived predictor for architecture screening; it is not the final head for a newly long-trained predictor.

## Reproduction/provenance

- GPU: `NVIDIA GeForce RTX 5070`; PyTorch `2.9.1+cu128`; Python `3.11.15`.
- Data manifest SHA-256: `3612185c939f6c4cd4890afd3968d160a44e88d3e6f7733b8cb9ce71501d532c`.
- Dev split manifest SHA-256: `d127e851f3f5eefb011313b1b0b1d79e65db6d99c0754d264119ff12046a83a2`.
- Base CNO checkpoint SHA-256: `ff28aaf0114d57e320e5ea9cf9945d874a7482f22e3f0aeaed5ebbe286573f8a`.
- Residual start checkpoint SHA-256: `909fdc7f8a6a42335e4ea4ce7471fc9a507135e9a29a4927bc7b7318be9c85b2`.
- Raw runner output is preserved in `screen_report.json`, `run_manifest.json`, `commands.jsonl`, per-arm summaries, and the split/data manifests in this directory.

## Decision for the next stage

No residual arm or uncertainty-head variant met its pre-registered gate. Do not start a long training run solely from this screen. Any follow-up should first explain whether the small proxy gains from TKE reweighting justify relaxing the Rel-L2 gate or testing a different loss formulation.
