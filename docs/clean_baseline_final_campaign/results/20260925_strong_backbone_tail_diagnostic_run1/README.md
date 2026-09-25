# Exp1 Strong Backbone F18–F20 zero-training diagnostic

Status: `REVIEW_REQUIRED`

## Scope and protocol

This is an inference-only diagnostic on the frozen Clean Seen-Dev12 set: 51 Train / 12 Seen-Dev trajectories, 491 evaluation windows. It compares the exact Exp1 Strong Backbone output and Strong Backbone + Residual output against the same targets at F1–F20. No optimizer step or training was performed. AoA10 fields, locked-final/private, and Codabench were not accessed.

The two checkpoint SHA256 values were verified before execution:

- Strong Backbone: `d340effd68031aba1cd2bc610c676878fb9105dd7f81b9ce15767280e665a5c0`
- Residual: `d2b4ddd0df6064d053ec918c2671a93b7d9c2a40a15afe854bb3a5badf1438fc`

Code was synchronized from `main` at `1a7d2a306b0dddd149b24efaf2a6a96b27450e55`, which contains requested commit `c800877464193c3afd499a851cedbc97fce1a557`. Test command result: `5 passed`; py_compile and `git diff --check` passed. The local PyTorch emitted a NumPy compatibility warning during test import, but the tests passed. Remote diagnostic runner exit code: `0`.

## F18–F20 results

| Horizon | Backbone Rel-L2 | Final Rel-L2 | Correction gain | Delta-target cosine | alpha* | Help fraction |
|---|---:|---:|---:|---:|---:|---:|
| F18 | 0.112266 | 0.092574 | 17.540% | 0.540736 | 1.067458 | 98.778% |
| F19 | 0.133289 | 0.098094 | 26.405% | 0.650055 | 1.151403 | 100.000% |
| F20 | 0.174276 | 0.108411 | 37.793% | 0.783148 | 1.447733 | 100.000% |

### F18 → F20 changes

- Backbone Rel-L2 growth: **+55.235%**.
- Final Rel-L2 growth after correction: **+17.107%**.
- Correction-gain drop (`F18 gain - F20 gain`): **-20.253 percentage points**, meaning correction gain rose by 20.253 points at F20.
- Cosine drop (`F18 cosine - F20 cosine`): **-0.242411**, meaning alignment cosine rose from 0.541 to 0.783.

Diagnostic flags from the frozen script thresholds:

- `backbone_tail_growth_present = true`
- `residual_effectiveness_drop_present = false`
- `residual_alignment_drop_present = false`

The measured pattern is consistent with sharp Backbone tail-error growth while the Residual continues to help and its direction alignment improves. At F20, `alpha* = 1.448` and delta/target RMS ratio is `0.541`, indicating a possible moderate under-correction along the learned direction. This is diagnostic evidence only; it does not trigger another training run.

## Runtime and artifacts

- GPU: one RTX 3090 Ti; inference-only batch size 16 (script default), workers 2.
- Remote output: `/hy-tmp/realpde_runs/strong_backbone_tail_diagnostic_20260925_run1`.
- Source commit: `1a7d2a306b0dddd149b24efaf2a6a96b27450e55`.
- The diagnostic briefly overlapped the independent running SPS job after explicit user authorization; no OOM or diagnostic failure occurred.
- The archive contains only the script’s CSV/JSON evidence, run manifest, completion record, and checksums. It contains no checkpoint weights, HDF5 data, or full console log.
