# REALPDE F18–F20 Error Anatomy

**Status:** `REVIEW_REQUIRED`
**Code:** `main` at `1fee054ff022576fb634e08e7445af459583a1d6`
**Run:** `strong_backbone_tail_error_anatomy_20260925_run1`

## Scope and verification

- Clean Train51 / Seen-Dev12 only; 491 Seen-Dev windows from 12 trajectories.
- Backbone SHA-256: `d340effd68031aba1cd2bc610c676878fb9105dd7f81b9ce15767280e665a5c0`.
- Residual SHA-256: `d2b4ddd0df6064d053ec918c2671a93b7d9c2a40a15afe854bb3a5badf1438fc`.
- Tests: **PASS**, 13 passed on the local workspace and remote GPU venv. `py_compile` passed on both.
- Optimizer steps: **0**. No training, AoA10 field access, locked-final/private access, Codabench access, full-data run, or submission packaging.
- GPU: NVIDIA RTX 3090 Ti. The GPU was idle in the preflight snapshot; no Experiment 3 process was present or interrupted.

## Finding

The strongest zero-training signal is a severe **tail fluctuation-amplitude mismatch** in the backbone. F20 least-squares scaling reduces tail-frame SSE by 65.56%, but its whole-Future20 Rel-L2 improvement comes with 20.87% worse TKE and 3.42% worse MVPE. On the ordinary residual predictor, the corresponding oracle changes whole-Future20 Rel-L2 by −0.80%, TKE by +7.65%, and MVPE by −0.01%. This is diagnostic headroom, not a safe correction.

Spatial shifts give small backbone-only F20 gains and no global-shift gain on the final predictor. Temporal target offsets favor slightly earlier truth frames, but the gains are small; every allowed extrapolation grid selects `gamma=0`. Residual tail alpha improves tail SSE while making all three whole-Future20 metrics slightly worse. These results do not support residual under-correction or a temporal extrapolation fix as the main opportunity. They also do not establish a genuine backbone representation limitation as the primary cause while the amplitude oracle has such a large tail-frame gain.

## Spatial phase

| Horizon | Global best `(dx,dy)` | Global SSE gain | Per-window oracle gain | Mode shift / fraction | Nonzero fraction |
|---|---:|---:|---:|---:|---:|
| F18 | `(0,0)` | 0.00% | 1.76% | `(0,0)` / 82.28% | 17.72% |
| F19 | `(-1,0)` | 3.32% | 5.14% | `(0,0)` / 52.95% | 47.05% |
| F20 | `(-1,0)` | 3.82% | 4.51% | `(-1,0)` / 54.38% | 59.27% |

The direction becomes more common late, but the gains remain modest and the per-window choices are mixed. The final predictor independently selects `(0,0)` at all three horizons; its F20 shift changes whole-Future20 Rel/TKE/MVPE by **0.00% / 0.00% / 0.00%**. Assessment: **weak coherent spatial-drift evidence**.

## Temporal phase

| Predictor | Horizon | Best GT horizon (offset) | Lag-probe SSE gain | Best `gamma` | Extrapolation SSE gain |
|---|---:|---:|---:|---:|---:|
| Backbone | F18 | F17 (−1) | 1.46% | 0.00 | 0.00% |
| Backbone | F19 | F17 (−2) | 1.72% | 0.00 | 0.00% |
| Backbone | F20 | F18 (−2) | 1.28% | 0.00 | 0.00% |
| Final predictor | F18 | F17 (−1) | 0.39% | 0.00 | 0.00% |
| Final predictor | F19 | F18 (−1) | 1.50% | 0.00 | 0.00% |
| Final predictor | F20 | F19 (−1) | 1.45% | 0.00 | 0.00% |

Predicted F20 is 0.88% closer in SSE to GT F19 than GT F20, but GT F18 is closer still. Positive extrapolation worsens every backbone and final-predictor tail frame. Assessment: **weak lag evidence; no support for forward extrapolation**. The final predictor's F20 extrapolation changes whole-Future20 Rel/TKE/MVPE by **0.00% / 0.00% / 0.00%**.

## Fluctuation amplitude

| Predictor | Horizon | `scale*` | Tail-frame SSE gain |
|---|---:|---:|---:|
| Backbone | F18 | 0.3511 | 18.64% |
| Backbone | F19 | 0.1570 | 39.74% |
| Backbone | F20 | 0.0675 | 65.56% |
| Final predictor | F18 | 0.6940 | 3.80% |
| Final predictor | F19 | 0.4971 | 11.20% |
| Final predictor | F20 | 0.2654 | 29.72% |

Whole-Future20 metric change under the independently fitted F18–F20 amplitude oracle:

| Predictor | Rel-L2 change | TKE change | MVPE change |
|---|---:|---:|---:|
| Backbone | −6.55% | +20.87% | +3.42% |
| Final predictor | −0.80% | +7.65% | −0.01% |

The amplitude mismatch is clear in tail SSE. Its whole-sequence tradeoff, especially the TKE increase, makes it **oracle evidence rather than a low-risk correction**.

## Residual tail alpha

| Horizon | `alpha*` | Existing residual tail SSE gain | Oracle tail SSE gain |
|---|---:|---:|---:|
| F18 | 1.0675 | 29.12% | 29.24% |
| F19 | 1.1514 | 41.53% | 42.26% |
| F20 | 1.4477 | 55.47% | 61.33% |

Current Exp1 final whole-Future20 metrics are **Rel-L2 0.079537 / TKE 0.496013 / MVPE 0.064418**. Applying the Seen-Dev tail-alpha oracle yields **0.079831 / 0.498733 / 0.064570**, respectively: **+0.37% / +0.55% / +0.24%** versus current final. The residual direction is useful at the tail, but increasing its magnitude is not a whole-sequence opportunity on these metrics.

## Mechanism ranking and assessment

F20 backbone tail-frame SSE gain, ranked:

1. Amplitude scaling: **65.56%**
2. Global spatial shift: **3.82%**
3. Temporal extrapolation: **0.00%**

- Spatial drift coherent: **WEAK**
- Temporal lag: **WEAK** (small negative offsets; no positive extrapolation benefit)
- Amplitude mismatch: **YES** (large tail SSE gain; whole-sequence TKE tradeoff)
- Residual magnitude opportunity: **NO** (tail alpha worsens whole-Future20 Rel/TKE/MVPE)
- Primary diagnosis: amplitude mismatch is the strongest measured zero-training mechanism. This run does not justify selecting a repair or starting another experiment; all oracle values are Seen-Dev only.

## Reproduction

Remote output: `/hy-tmp/realpde_runs/strong_backbone_tail_error_anatomy_20260925_run1`
Data root: `/hy-tmp/realpde_data/train_real`
Split manifest: `/hy-tmp/realpde_runs/strong_backbone_clean_20260924_run1/train_dev_manifest.json`
The exact code commit, split-manifest checksum, checkpoint paths, and environment are recorded in [`run_manifest.json`](run_manifest.json). File checksums are in [`SHA256SUMS.txt`](SHA256SUMS.txt).

```bash
python -u -B tools/diagnose_tail_error_anatomy.py \
  --real-root "$REAL_ROOT" \
  --split-manifest "/hy-tmp/realpde_runs/strong_backbone_clean_20260924_run1/train_dev_manifest.json" \
  --backbone-checkpoint "/hy-tmp/realpde_runs/strong_backbone_clean_20260924_run1/exp1_strong_backbone_clean/strong_backbone/checkpoints/model_best.pth" \
  --residual-checkpoint "/hy-tmp/realpde_runs/strong_backbone_clean_20260924_run1/exp1_strong_backbone_clean/residual_22k/model_best.pth" \
  --model-root "$KIT_ROOT" \
  --out-dir "/hy-tmp/realpde_runs/strong_backbone_tail_error_anatomy_20260925_run1" \
  --workers 2 --require-cuda
```

No checkpoints, H5 files, prediction arrays, or console logs are archived here.
