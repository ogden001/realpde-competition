# Direct-CNO 7.5k Matched Control

Status: `REVIEW_REQUIRED`

## Goal

Test one causal hypothesis behind the Strong Backbone F19/F20 cliff:

> Does the cliff come from the MF mean/fluctuation output parameterization, or does it remain when the exact same Strong recipe uses a direct CNO output?

This is a **matched 7,500-update control**, not a new optimization campaign.

## Single scientific variable

Reference at update 7,500:

`Dense-All + P0-A + MF-CNO + N2 + Vorticity`

Control:

`Dense-All + P0-A + Direct-CNO + N2 + Vorticity`

Everything else is frozen.

## Frozen protocol

- Clean Train51 / Seen-Dev12.
- 41,317 dense Train windows.
- 491 fixed Seen-Dev windows.
- Official `sim_real_cno.pth` initialization.
- Init SHA256:
  `82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61`
- P0-A feature builder unchanged.
- Seed = 41.
- Batch = 8.
- AdamW semantics identical to `train_clean_strong_backbone.py`.
- LR = `1e-5`.
- N2 weights unchanged.
- Vorticity weight unchanged.
- 7,500 optimizer updates.
- Stage-B is not entered.
- No AoA10 / locked-final / Codabench / full-data / SPS selection.

Milestone evaluations:
- 0
- 2,500
- 5,000
- 7,500

The primary causal comparison is **Direct @7,500 vs MF @7,500**.

## Matched MF reference

Frozen MF @7,500 evidence from the tail-origin diagnostic:

- Rel-L2: `0.1216189489`
- TKE: `0.5056437850`
- MVPE: `0.0957256854`
- F18 Rel: `0.1258108965`
- F19 Rel: `0.1462923875`
- F20 Rel: `0.1956690987`
- F18->F20 growth: `55.5264%`

Evidence source:

`docs/clean_baseline_final_campaign/results/20260925_strong_backbone_tail_origin_run1/checkpoint_curve.csv`

## Required diagnostics

For Direct CNO at every milestone:

- overall Rel-L2 / TKE / MVPE;
- point score;
- by-horizon F1..F20;
- by-trajectory;
- trajectory × horizon;
- Future20 mean-field metrics;
- fluctuation amplitude / cosine / Rel-L2 by horizon;
- F18/F19/F20 tail summary.

No additional oracle or parameter sweep.

## Interpretation

### Evidence supporting MF representation as the main cliff source

Direct @7,500 should show a substantially flatter F18-F20 curve than the frozen MF @7,500 reference.

The main evidence is:
- Direct F18->F20 growth;
- Direct F19 and F20 Rel;
- Direct fluctuation coherence at F18/F19/F20.

Overall Rel/TKE/MVPE are secondary for the causal question, but still required because a flatter tail obtained by collapsing the useful dynamics would not be a meaningful solution.

### Evidence against MF as the main source

If Direct @7,500 still shows a similar F19/F20 cliff, the cause lies upstream of the MF output parameterization, most plausibly in the P0-A / N2 / vorticity training recipe or their interaction.

Sol makes the final research conclusion. Codex does not launch another experiment from this result.

## GPU resource contract

This experiment is explicitly authorized to run concurrently with other approved jobs on the same RTX 3090-class GPU.

Existing GPU utilization is **not** a blocking condition.

Hard resource constraints:

- this process must use **<=12 GiB peak CUDA reserved memory**;
- training batch remains exactly 8;
- do not reduce micro-batch or add gradient accumulation to fit memory;
- do not switch FP32/BF16/FP16;
- do not stop, pause or modify existing approved GPU jobs.

The runner records and enforces its own CUDA peak reserved memory.

If the exact batch-8 control exceeds 12 GiB or OOMs while sharing the GPU:
- do not modify scientific training semantics;
- leave existing jobs untouched;
- report resource conflict / BLOCKED, or execute later when enough memory is available.

Because the GPU is shared:
- wall-clock time / throughput / latency are not valid comparison metrics;
- point-prediction metrics remain valid;
- runtime fields must be marked invalid for performance comparison.

## HARD CONSTRAINTS

- The only scientific variable is MF output -> Direct output.
- Batch=8 is frozen.
- Seed=41 is frozen.
- 7,500 updates exactly.
- LR=1e-5 exactly.
- Same Clean split and sampling.
- Same P0-A.
- Same N2 and vorticity objective.
- Same official initialization.
- No Stage-B.
- No AoA10.
- No locked-final/private.
- No Codabench.
- No full-data.
- No SPS selection.
- No checkpoint interpolation.
- No extra experiment or hyperparameter sweep.
- No automatic next experiment.
- Experiment process peak CUDA reserved memory <=12 GiB.

## SOFT / OPERATIONAL CONSTRAINTS

May adapt without changing science:

- checkout / Git-bundle mechanism;
- Python / venv path;
- CUDA device index;
- workers;
- eval batch size;
- pin-memory / prefetch;
- output/log/tmux paths;
- temporary directories;
- coexistence with unrelated untracked files.

Do not reinterpret training micro-batch as a soft parameter. It is scientifically frozen to 8 for this control.

## Outputs

Top level:

- `run_config.json`
- `preflight.json`
- `aggregate_metrics.csv`
- `runtime.json`
- `summary.json`
- `DONE`

Each eval milestone:

- `summary.json`
- `by_horizon.csv`
- `by_trajectory.csv`
- `by_trajectory_horizon.csv`
- `fluctuation_by_horizon.csv`

Checkpoints may remain on the GPU host. Only lightweight evidence goes to Git.
