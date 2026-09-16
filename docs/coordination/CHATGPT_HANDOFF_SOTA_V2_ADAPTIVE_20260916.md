# ChatGPT handoff — SOTA-V2 adaptive — 2026-09-16

Status: **ONLINE_KEEP / NEW_ONLINE_SOTA**

- Required source commit: `8b2e74a8dc4bde1ca7f7f8b755b3d902d7a740d9`; SHA guard source is fixed.
- Gate: `ADAPTIVE_GO`; adaptive Dev SPS `45.07008160038756` vs static `42.12489194711354`.
- Full checkpoint: update `53582`, SHA256 `f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce`.
- Adaptive head: `1400` updates, SHA256 `01c1fc06f806ca00a370cf971df0f093d363ab7f3a4c4e473f1727441f3d4d17`.
- Submitted package: `/home/chyfuture/realpde_runs/sota_v2_adaptive_20260916/package_clean/submission.zip`, SHA256 `9cfc055c6232d2b0aef9f88f1cb3de2aae883b7cff7f02659ed8b9cf85b3ed55`, bytes `30191330`.
- Clean-room smoke A/B: both `PASS`; prediction parity `0.0` and deterministic diff `0.0`.
- Runtime first/steady: A `0.3303255550/0.0307885470 s`; B `0.3408626430/0.0308223500 s`; peak CUDA `142737920` bytes.
- Clean rebuild used no monkeypatch or in-process SHA override.

## Codabench result

- Final: `77.314446`
- Rel-L2: `93.816645`
- TKE: `79.164203`
- MVPE: `93.411176`
- Time: `86.898836`
- SPS: `30.319572`

Relative to the previous online SOTA `76.694784 / 93.434384 / 77.588799 / 92.519563 / 87.066646 / 29.519724`:

- Final `+0.619662`
- Rel-L2 `+0.382261`
- TKE `+1.575404`
- MVPE `+0.891613`
- Time `-0.167810`
- SPS `+0.799848`

Conclusion: `KEEP / NEW_ONLINE_SOTA`. The SOTA-V2 backbone transferred positively online across all three physical prediction metrics, while the fresh adaptive uncertainty component also raised SPS. The slight Time decrease does not offset the aggregate gain.

Review evidence: `docs/sota迭代/reviews/sota_v2_adaptive_20260916/`.
