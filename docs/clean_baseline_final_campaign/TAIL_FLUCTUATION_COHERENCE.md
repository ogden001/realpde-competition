# Exp1 Tail Fluctuation-Coherence Diagnostic

Status: `REVIEW_REQUIRED`

This is the final zero-training mechanism diagnostic for the Exp1 F19/F20 tail cliff.

It does **not** search for a submission parameter. It separates each Future20
prediction and target into:

```text
Future20 field = temporal mean field + zero-mean fluctuation
```

and asks whether the tail failure is mainly:

1. mean-field error;
2. fluctuation-amplitude error;
3. fluctuation coherence / phase error.

## Frozen scientific scope

Use the exact Exp1 Strong Backbone and ordinary Residual checkpoints:

- Backbone SHA256:
  `d340effd68031aba1cd2bc610c676878fb9105dd7f81b9ce15767280e665a5c0`
- Residual SHA256:
  `d2b4ddd0df6064d053ec918c2671a93b7d9c2a40a15afe854bb3a5badf1438fc`

Evaluate only:

- Clean Train51 / Seen-Dev12 manifest;
- 491 Seen-Dev windows;
- no AoA10 field access.

## Measurements

For both:

- Strong Backbone;
- Strong Backbone + ordinary Residual;

the script emits:

### Mean field

- global Future20 temporal-mean Rel-L2;
- per-window mean / median mean-field Rel-L2;
- prediction/target mean-field norm ratio.

### Fluctuation by horizon F1..F20

- predicted fluctuation RMS;
- target fluctuation RMS;
- amplitude ratio;
- fluctuation cosine, global and per-window;
- least-squares fluctuation scale after means are correctly separated;
- fluctuation Rel-L2.

This is the key distinction from the previous amplitude oracle. The previous
oracle scaled a frame around the prediction mean toward the absolute target.
This diagnostic compares true prediction fluctuation to true target
fluctuation after each sequence's own temporal mean is removed.

### 20x20 phase-similarity matrix

For every predicted fluctuation F_i and GT fluctuation F_j:

```text
cos(pred_fluctuation_i, gt_fluctuation_j)
```

is computed using:

- one global aggregate cosine;
- the mean of per-window cosines.

For every predicted horizon, the script reports:

- diagonal cosine;
- best matching GT horizon;
- best phase offset;
- improvement of the best match over the diagonal match.

### Trajectory consistency

For every Seen-Dev trajectory at F18/F19/F20:

- fluctuation cosine;
- amplitude ratio.

This distinguishes a systematic mechanism from a few difficult trajectories.

## Interpretation

### Coherence / phase failure

Strong evidence if:

- F20 amplitude ratio remains roughly order-1;
- F20 diagonal fluctuation cosine collapses;
- the 20x20 matrix shows an off-diagonal best match or broad phase ambiguity;
- the same trend is present across many trajectories.

Do **not** solve this by shrinking amplitude. A phase/coherence correction or
tail-focused backbone change is the relevant family.

### Amplitude failure

Strong evidence if:

- fluctuation cosine remains high;
- amplitude ratio is far from 1;
- least-squares fluctuation scale is meaningfully away from 1.

A constrained amplitude correction may then be justified.

### Mean-field failure

If Future20 mean-field error itself is large, the problem is broader than
tail fluctuation phase. A short backbone Stage-C is more plausible than a tiny
phase-only head.

## Hard constraints

- Seen-Dev12 only.
- Exact checkpoint SHA match required.
- Exactly 491 evaluation windows.
- Zero optimizer steps.
- No training.
- No parameter search beyond the defined diagnostic calculations.
- No AoA10 field access.
- No locked-final/private.
- No Codabench.
- No full-data refit.
- No submission packaging.
- Do not launch a follow-up experiment automatically.
- Do not reinterpret oracle/phase offsets as submission parameters.

## Soft / operational constraints

The following may be changed without blocking the task, provided scientific
inputs and outputs remain identical:

- local checkout / Git-bundle synchronization method;
- `CUDA_VISIBLE_DEVICES` / actual GPU index;
- Python executable / venv activation;
- output directory name;
- log path / tmux session name;
- DataLoader workers;
- evaluation batch size if needed for memory;
- CPU affinity / prefetch / pin-memory operational settings;
- temporary directories;
- handling of unrelated pre-existing untracked files.

Do not modify model architecture, checkpoint state, data split, sampling,
prediction values, metric definitions or scientific calculations under the
label of an operational fix.

## Outputs

- `backbone_summary.json`
- `final_summary.json`
- `backbone_fluctuation_by_horizon.csv`
- `final_fluctuation_by_horizon.csv`
- `backbone_phase_alignment.csv`
- `final_phase_alignment.csv`
- `backbone_phase_similarity_global.csv`
- `backbone_phase_similarity_window_mean.csv`
- `final_phase_similarity_global.csv`
- `final_phase_similarity_window_mean.csv`
- `backbone_tail_by_trajectory.csv`
- `final_tail_by_trajectory.csv`
- `comparison_summary.json`
- `run_manifest.json`
- `DONE`
