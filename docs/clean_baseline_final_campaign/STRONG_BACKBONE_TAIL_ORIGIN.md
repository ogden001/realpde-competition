# Strong Backbone F19/F20 Tail-Origin Diagnostic

Status: `REVIEW_REQUIRED`

This is a zero-training diagnostic for one question:

> Why does the Strong Backbone, unlike the Clean colleague backbone, develop a sharp F19/F20 cliff?

The diagnostic combines:

1. **Checkpoint evolution** from update 0 through 35k.
2. **Raw MF head anatomy** before and after temporal zero-mean reconstruction.

No new model is trained or selected.

## Frozen scientific inputs

### Clean split

- Train51 / Seen-Dev12 only.
- Seen-Dev = 12 trajectories / 491 fixed windows.
- No AoA10 field access.

### Official initialization

Official sim-real CNO SHA256:

`82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61`

Update 0 is reconstructed using the exact Strong Backbone initialization path:

`official direct CNO -> P0-A input expansion -> MF five-channel projection initialization`.

### Required Strong Backbone milestones

All eight exact checkpoint files must exist:

- 7,500
- 15,000
- 20,000
- 25,000
- 30,000
- 31,000
- 32,500
- 35,000

The script validates the payload `iteration` against the filename.

Do not substitute `model_best.pth`, `model_latest.pth`, or a different training run for a missing milestone.

## Diagnostic 1: checkpoint evolution

For update:

`0, 7500, 15000, 20000, 25000, 30000, 31000, 32500, 35000`

evaluate the exact same 491 Seen-Dev windows and record:

- whole-sequence Rel-L2 / TKE / MVPE;
- F18 / F19 / F20 Rel-L2;
- F18->F19, F19->F20, F18->F20 growth;
- raw fluctuation amplitude at F18/F19/F20;
- reconstructed fluctuation amplitude at F18/F19/F20;
- fluctuation cosine;
- F19+F20 share of total fluctuation energy.

Interpretation:

- cliff already large at update 0 -> architecture / temporal-boundary initialization is plausible;
- cliff small at update 0 but grows with training -> training objective / optimization-induced energy allocation is plausible;
- especially strong evidence if TKE improves while F19/F20 energy share and cliff grow together.

## Diagnostic 2: raw MF anatomy

MF output before reconstruction is:

`[mean_u, mean_v, fluct_u, fluct_v, p]`.

The deployed prediction uses:

```text
mean_field = temporal_mean(mean_raw)
centered_fluct = fluct_raw - temporal_mean(fluct_raw)
prediction = mean_field + centered_fluct
```

For every milestone and every horizon F1..F20 record:

- raw fluctuation RMS;
- centered fluctuation RMS;
- target fluctuation RMS;
- raw amplitude ratio;
- centered amplitude ratio;
- raw / centered fluctuation cosine;
- raw / centered fluctuation Rel-L2;
- raw / centered / target energy share;
- pseudo pre-centering frame Rel-L2;
- deployed reconstructed frame Rel-L2;
- raw mean-head deviation from its Future20 temporal mean.

Interpretation:

- raw F20 already abnormal -> anomaly exists in learned CNO/MF raw head before zero-mean reconstruction;
- raw is smooth but reconstruction creates the cliff -> MF centering is implicated;
- both grow together during training -> learned raw-head temporal energy allocation is primary, with centering only secondary.

The pseudo pre-centering prediction is diagnostic only. It is not a valid submission candidate.

## HARD CONSTRAINTS

These are scientific constraints and may not be changed:

- Clean Train51 / Seen-Dev12 only.
- Exactly 491 Seen-Dev windows.
- Official init SHA must match.
- Exact milestone iterations listed above.
- Same P0-A feature builder and MF architecture as Exp1.
- Same fixed Seen-Dev sampling.
- Zero optimizer steps.
- No training or fine-tuning.
- No checkpoint selection.
- No interpolation.
- No parameter sweep.
- No new oracle transform.
- No AoA10 field access.
- No locked-final/private.
- No Codabench.
- No full-data refit.
- No submission packaging.
- No automatic follow-up experiment.
- Do not replace a missing checkpoint with another checkpoint.
- Do not change raw-head/reconstruction definitions or diagnostic metric formulas.

If any scientific input is missing or inconsistent and cannot be restored without changing the protocol, return `BLOCKED`.

## SOFT / OPERATIONAL CONSTRAINTS

The following may be adapted without blocking, provided scientific results are unchanged:

- GitHub vs verified Git-bundle synchronization.
- Independent worktree / checkout.
- Python executable / venv.
- `CUDA_VISIBLE_DEVICES` and physical GPU index.
- Output path and log path.
- tmux/nohup use.
- evaluation batch size.
- DataLoader workers.
- pin_memory / prefetch / CPU-affinity settings.
- temporary directories.
- unrelated pre-existing untracked files.

If the GPU is occupied by another approved experiment, do not kill it. Wait or use another scientifically equivalent execution slot.

Principle: **lock science, not operations**.

## Outputs

Top-level:

- `checkpoint_curve.csv`
- `evolution_summary.json`
- `checkpoint_sha256.json`
- `run_manifest.json`
- `DONE`

For every update:

- `update_XXXXX/by_horizon.csv`
- `update_XXXXX/raw_mf_by_horizon.csv`
- `update_XXXXX/summary.json`

Archive only lightweight CSV/JSON/MD evidence. Do not commit checkpoints, H5 data, prediction arrays, or full console logs.
