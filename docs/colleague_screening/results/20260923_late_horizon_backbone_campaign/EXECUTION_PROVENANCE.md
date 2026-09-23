# Late-Horizon Backbone Campaign — execution provenance

Status: `REVIEW_REQUIRED`. This report records execution evidence only; it
does not make a scientific GO/NO-GO judgment.

## Source and preflight

- Actual execution HEAD: `6fa2f13d7db2e81e4c913df35fe379007953d185` on `main`.
  It includes the requested ancestry `6d23868d9a47733c9b6f0b8294ebbdc2fab93c05`.
- A pure compatibility fix was required because the official v9 kit exports
  `CNO3d` from `rpde_baselines.model.cno`, while the helper first looked for the
  older `rpde_baselines.cno` location. The fallback and regression test are in
  the execution commit; no loss, data, checkpoint, budget, optimizer, sampling,
  or evaluation semantics were changed.
- The originally specified manifest path was absent on the GPU host. The
  manifest copy used was byte-identical to the repository copy, SHA256
  `d127e851f3f5eefb011313b1b0d1d79e65db6d99c0754d264119ff12046a83a2`.
  It identifies `colleague_dev16_seed41_all81`, 16 Dev trajectories, and the
  documented exclusion. Scorer SHA256:
  `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`.
- Prescribed tests: `22 passed` for
  `tests/test_late_horizon_backbone_campaign.py`,
  `tests/test_post_train_diagnostics.py`, and
  `tests/test_colleague_v2_campaign.py`.
- Prescribed `python -m py_compile tools/late_horizon_backbone_campaign.py`:
  passed. The official CNO checkpoint also loaded on CPU with all 232 state
  entries matched and no missing or unexpected keys.

## Verified checkpoint identities

| Checkpoint | SHA256 |
|---|---|
| Colleague backbone | `ff28aaf0114d57e320e5ea9cf9945d874a7482f22e3f0aeaed5ebbe286573f8a` |
| Current80 residual | `909fdc7f8a6a42335e4ea4ce7471fc9a507135e9a29a4927bc7b7318be9c85b2` |
| Strong backbone @53582 | `cc732555859cb00b3ea1af31415632c087af1fc60abd4f4056263e6f02ffa307` |
| Arm-B final @38400 | `fde7f3c20e1e42f93f616741a3c6e1a68d4e38c4dc002f3f4875f9f2c5b18762` |

The campaign's checkpoint-hash record is also archived in
`experiment0_fingerprint/checkpoint_sha256.json`.

## Execution bounds

- Experiment 0: 640 shared windows from 16 Dev trajectories, all four models,
  no training.
- Experiment 1: both arms start at iteration 53,582; 82 released PIV
  trajectories; Dense-All sampling; micro-batch 4, accumulation 2, effective
  batch 8; seed `20260901`; restored AdamW state; LR `3e-6`; FP32; clip 1.0;
  exactly 5,000 updates per arm; evaluation at 0/1k/2k/3k/4k/5k.
- Control uses the historical Stage-B objective. Ramp changes only the N2
  velocity-MSE weighting from uniform Future20 to mean-normalized linear
  F1=1.0 through F20=2.0. The archived run configs and logs preserve both
  arms' objective and sampler provenance.
- The runner wrote `DONE`; all requested per-step summaries and checkpoints
  through update 5,000 are present. No training beyond 5k, residual training,
  uncertainty-head training, parameter sweep, submission/package, Codabench,
  or locked-final/private access was started.

The outer `ARCHIVE_MANIFEST.json` hashes each archived evidence file. Model
checkpoints are deliberately excluded from this Git evidence archive.
