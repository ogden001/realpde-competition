# REALPDE Tail Fluctuation Coherence

**Status:** `REVIEW_REQUIRED`
**Code:** `main` at `cbac90a8be0678d2778a2cab69a7b238638ba323`
**Remote run:** `tail_fluctuation_coherence_20260925_run1`

## Scope and verification

- Clean Train51 / Seen-Dev12 only: 12 trajectories and 491 windows.
- Backbone SHA-256: `d340effd68031aba1cd2bc610c676878fb9105dd7f81b9ce15767280e665a5c0`.
- Residual SHA-256: `d2b4ddd0df6064d053ec918c2671a93b7d9c2a40a15afe854bb3a5badf1438fc`.
- Tests: **PASS**, 20 passed on the local workspace and remote GPU venv. `py_compile` passed locally and remotely; local `git diff --check` passed.
- Optimizer steps: **0**. No training, checkpoint selection, parameter tuning, AoA10 field access, locked-final/private access, Codabench, full-data refit, or submission packaging.
- GPU: NVIDIA RTX 3090 Ti. Preflight and run-time checks showed the GPU idle; no Exp3 process was interrupted.

## Finding

The dominant F19/F20 issue is **fluctuation coherence loss**, not a pure amplitude error or a consistent one-step phase lag. The Strong Backbone F20 fluctuation amplitude is 1.65× the target, but its global fluctuation cosine is only 0.122. More decisively, the ordinary residual brings the final predictor F20 amplitude ratio to 0.942 while its global cosine remains just 0.276 (window-mean cosine 0.420). The F20 global phase matrix's best match is only one horizon earlier and improves cosine by less than 0.01; the per-window-mean matrix remains diagonal. That is low coherence without a strong, shared temporal shift.

Mean-field norms are close to target (0.989 backbone, 0.996 final); mean-field Rel-L2 is 5.41% and 4.71%. This is a modest background error and does not explain the much larger fluctuation error. The previous amplitude-only explanation is rejected as the primary mechanism. Amplitude mismatch is present in the backbone, but does not explain the final predictor's low coherence.

## Strong Backbone

**Future20 mean-field Rel-L2:** 0.05407 global (0.05407 window mean; 0.04943 median). **Mean-field norm ratio:** 0.989 global (0.991 window mean).

| Horizon | Pred / GT fluctuation RMS | Amplitude ratio | `scale*` | Global cosine | Window mean / median cosine |
|---|---:|---:|---:|---:|---:|
| F18 | 0.011305 / 0.013125 | 0.861 | 0.367 | 0.316 | 0.323 / 0.335 |
| F19 | 0.015321 / 0.013755 | 1.114 | 0.173 | 0.193 | 0.236 / 0.246 |
| F20 | 0.023987 / 0.014518 | 1.652 | 0.074 | 0.122 | 0.166 / 0.164 |

## Backbone + Residual

**Future20 mean-field Rel-L2:** 0.04712 global (0.04398 window mean; 0.04107 median). **Mean-field norm ratio:** 0.996 global (0.998 window mean).

| Horizon | Pred / GT fluctuation RMS | Amplitude ratio | `scale*` | Global cosine | Window mean / median cosine |
|---|---:|---:|---:|---:|---:|
| F18 | 0.009117 / 0.013125 | 0.695 | 0.699 | 0.486 | 0.503 / 0.575 |
| F19 | 0.010425 / 0.013755 | 0.758 | 0.517 | 0.392 | 0.463 / 0.529 |
| F20 | 0.013679 / 0.014518 | 0.942 | 0.293 | 0.276 | 0.420 / 0.486 |

## Phase matrix: F18–F20 rows

Values are global aggregate cosine. The full 20×20 global and window-mean matrices are preserved in the CSV files.

| Predictor | Pred | Diagonal cosine | Best GT horizon | Offset | Best cosine | Gain vs diagonal |
|---|---:|---:|---:|---:|---:|---:|
| Backbone | F18 | 0.3162 | F18 | 0 | 0.3162 | 0.0000 |
| Backbone | F19 | 0.1928 | F18 | −1 | 0.1989 | 0.0061 |
| Backbone | F20 | 0.1225 | F19 | −1 | 0.1294 | 0.0069 |
| Final | F18 | 0.4856 | F18 | 0 | 0.4856 | 0.0000 |
| Final | F19 | 0.3917 | F18 | −1 | 0.3947 | 0.0030 |
| Final | F20 | 0.2764 | F19 | −1 | 0.2858 | 0.0094 |

For F18–F20, the best window-mean match stays on the diagonal for both predictors, with zero gain over the diagonal. The small global off-diagonal gains do not indicate a coherent phase correction.

## Trajectory consistency and assessment

- F20 trajectories with global fluctuation cosine below 0.5: **Backbone 12/12; Final 6/12**.
- The existing per-trajectory output has no best-phase offset. No new per-trajectory phase search was added.
- Amplitude-only explanation: **REJECTED** as the main cause.
- Fluctuation coherence loss: **SUPPORTED**.
- Temporal phase drift: **WEAK**; global best offsets are −1 but gains are tiny, and the window-mean best matches are diagonal.
- Mean-field failure: **WEAK**; field norms are near one and mean-field Rel-L2 is around 5%.

Interpretation: the backbone overshoots F20 fluctuation energy, but the final predictor has near-correct fluctuation energy and still poor phase-specific similarity. Lost fluctuation coherence is the stronger explanation. This is Seen-Dev mechanism evidence only; no repair direction is selected here.

## Reproduction and archived files

Remote output: `/hy-tmp/realpde_runs/tail_fluctuation_coherence_20260925_run1`
Split manifest: `/hy-tmp/realpde_runs/strong_backbone_clean_20260924_run1/train_dev_manifest.json`
Data root: `/hy-tmp/realpde_data/train_real`
The source commit, split checksum, checkpoint paths, and runtime are recorded in [`run_manifest.json`](run_manifest.json). Evidence checksums are in [`SHA256SUMS.txt`](SHA256SUMS.txt).

This archive contains only summaries, CSVs, manifests, checksums, and `DONE`. It contains no checkpoint, H5, prediction array, or console log.
