# TMR-01 — SOTA-V2 × Teammate Residual Corrector

Status: `REVIEW_REQUIRED`.

This is evidence only. Scientific interpretation and GO/NO-GO decisions remain with ChatGPT/Sol.

## Provenance

- Execution commit used for the two 2400-update trainings: `a2d2dc09e2507c3ad4cc8b6a3282a3797020fb24`.
- Evaluation recovery commit: `97d78e0ee8708aee306125cb38c15739ed5f017f`.
- Frozen manifest SHA256: `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`.
- Frozen checkpoint SHA256:
  - `@30000`: `d57539d97e94ae193997a59a04d02ca61f68f1e8837599a24df2e152ed10059f`
  - `@32500`: `6926722895611c38dee79cf16ce43b87c7de3d4a1305e9db431dadd4c1a7bf47`
- Trained corrector SHA256:
  - `@30000`: `68fe6af062a05a7e5b55b06fc23365604cce1369237d3c8825a9a6e1350f9dca`
  - `@32500`: `fa1c1b46f71a16da7b08c2f36a4de0f735dd9880b007c636d1b774ccbe084dc0`
- Official v9 scorer SHA256: `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`.
- Runtime: `realpde-pytorch-h5py:0831`, PyTorch `2.2.2+cu121`, NumPy `1.26.4`, h5py `3.16.0`, RTX 3090.
- Training used 50 Train Dense-All windows; Dev evaluation used 16 trajectories / 659 canonical windows.

The first runner invocation completed both corrector trainings, then stopped at the first Dev scorer call because of a missing helper name. The fix only loads the supplied official scorer and passes `measured_channels`; the two existing corrector checkpoints were reused, with no retraining.

## Physical metrics

Relative improvements are `1 - candidate / alpha0`; negative values mean a worse error.

| Backbone | Alpha | Rel-L2 | Rel improvement | TKE | TKE improvement | MVPE | MVPE improvement |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30000 | 0.0 | 0.105989 | 0.000% | 0.474373 | 0.000% | 0.090337 | 0.000% |
| 30000 | 0.5 | 0.095604 | +9.799% | 0.490215 | -3.340% | 0.079936 | +11.514% |
| 30000 | 1.0 | 0.090948 | +14.191% | 0.504334 | -6.316% | 0.075162 | +16.798% |
| 32500 | 0.0 | 0.099935 | 0.000% | 0.469293 | 0.000% | 0.075778 | 0.000% |
| 32500 | 0.5 | 0.092959 | +6.980% | 0.485677 | -3.491% | 0.073913 | +2.462% |
| 32500 | 1.0 | 0.089722 | +10.219% | 0.498592 | -6.243% | 0.073321 | +3.242% |

## Trajectory and horizon evidence

Trajectory counts are relative to each backbone's alpha-0 row, over 16 Dev trajectories:

| Backbone | Alpha | Rel-L2 improved | TKE improved | MVPE improved |
|---|---:|---:|---:|---:|
| 30000 | 0.5 | 16/16 | 0/16 | 16/16 |
| 30000 | 1.0 | 16/16 | 1/16 | 16/16 |
| 32500 | 0.5 | 16/16 | 0/16 | 14/16 |
| 32500 | 1.0 | 16/16 | 3/16 | 14/16 |

The h19 / h20 / h19+h20 squared-error fractions were:

| Backbone | Alpha | h19 | h20 | h19+h20 |
|---|---:|---:|---:|---:|
| 30000 | 0.0 | 8.672% | 16.434% | 25.105% |
| 30000 | 0.5 | 8.014% | 13.887% | 21.901% |
| 30000 | 1.0 | 7.713% | 11.251% | 18.965% |
| 32500 | 0.0 | 7.872% | 15.903% | 23.775% |
| 32500 | 0.5 | 7.766% | 13.622% | 21.388% |
| 32500 | 1.0 | 7.822% | 11.402% | 19.223% |

Full-correction fluctuation-energy ratios were `0.749` (`@30000`) and `0.760` (`@32500`) versus target energy. Full-correction RMS deltas were `0.006109` and `0.005195`, corresponding to `4.458%` and `3.738%` of backbone RMS respectively.

## Scope confirmations

- SPS: not accessed.
- Uncertainty / interval optimization: not accessed.
- Full-data refit: not accessed.
- Locked-final/private data: not accessed.
- Package build: not performed.
- Codabench: not accessed.
- No automatic GO/NO-GO and no next experiment started.

Supporting raw lightweight evidence is in `summary.json`, `physical_metrics.csv`, `trajectory_metrics_long_30000.csv`, `trajectory_metrics_long_32500.csv`, `horizon/`, and the training/provenance JSON files.
