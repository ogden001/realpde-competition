# ChatGPT Handoff — Matched Direct CNO + Local Residual A1

- Experiment: `T1-ID-HYBRID-CNO-LOCAL-A1-S20260904`
- Git implementation commit: `d05c983`
- Protocol: frozen 50/16/16 manifest, seed `20260901`, P0-A Direct CNO, N2, AdamW `1e-5`, batch 8; Direct@1500 → A1@3000.
- Architecture: `prediction = global_cno(x) + local_residual(x)`; local branch uses only Past20 raw `u/v`; zero-init final projection; 914 local parameters.
- Official scorer SHA: `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`.

## Dev result

Matched Direct / A1 raw errors:

| metric | Direct | A1 | relative delta | trajectory wins |
|---|---:|---:|---:|---:|
| Rel-L2 | 0.17582881 | 0.17646120 | -0.360% | 2/16 |
| TKE | 0.59464884 | 0.59071702 | +0.661% | 7/16 |
| MVPE | 0.15163144 | 0.15541382 | -2.494% | 1/16 |

Architecture verdict: `WEAK_SIGNAL_PARKED`. Level 2 trigger: not activated.

## Review artifact

The lightweight review evidence is now committed under:

[`docs/modeling/reviews/hybrid_cno_local_a1_20260904/`](../modeling/reviews/hybrid_cno_local_a1_20260904/)

This directory contains the final report, summary, 16-trajectory case table,
horizon metrics, update curve, runtime/provenance metadata, residual statistics,
and representative good/bad spatial maps.

The original lightweight ChatGPT bundle is also kept outside Git under:

`artifacts/hybrid_cno_local_a1_20260904_chatgpt_review.zip`

SHA-256: `dce578db93076e25caf64136dcaf5ffc4f6933a76d6105a82a0a4329c285c5a2`

It contains the report, case table, horizon metrics, maps, residual statistics, update curve, runtime and provenance. The full Dev prediction artifact and checkpoint remain on the GPU host and are intentionally not committed.
