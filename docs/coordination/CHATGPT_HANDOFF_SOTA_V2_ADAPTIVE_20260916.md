# ChatGPT handoff — SOTA-V2 adaptive — 2026-09-16

Clean package rebuild is complete and requires review only.

- Required source commit: `8b2e74a8dc4bde1ca7f7f8b755b3d902d7a740d9`; SHA guard source is fixed.
- Gate: `ADAPTIVE_GO`; adaptive SPS `45.07008160038756` vs static `42.12489194711354`.
- Full checkpoint: update `53582`, SHA256 `f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce`.
- Adaptive head: `1400` updates, SHA256 `01c1fc06f806ca00a370cf971df0f093d363ab7f3a4c4e473f1727441f3d4d17`.
- Final candidate package: `/home/chyfuture/realpde_runs/sota_v2_adaptive_20260916/package_clean/submission.zip`, SHA256 `9cfc055c6232d2b0aef9f88f1cb3de2aae883b7cff7f02659ed8b9cf85b3ed55`, bytes `30191330`.
- Clean-room smoke A/B: both `PASS`; prediction parity `0.0` and deterministic diff `0.0`.
- Runtime first/steady: A `0.3303255550/0.0307885470 s`; B `0.3408626430/0.0308223500 s`; peak CUDA `142737920` bytes.
- Clean rebuild used no monkeypatch or in-process SHA override.
- Codabench submission: not performed.
- Review evidence: `code/docs/sota迭代/reviews/sota_v2_adaptive_20260916/`.

The previous package evidence remains historical; the final candidate points to `package_clean/submission.zip`. Repository algorithm/source files remain unchanged.
