# Idle-GPU Overnight Queue — 2026-09-06

Status: `A2_COMPLETE_NO_GO / HCUR_PREFLIGHT_FAILED`

## Execution provenance

- Execution commit: `4f12367d45a4730e0caf00b67dbe0a311f99d868`
- Frozen manifest SHA-256: `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`
- Direct@1500 SHA-256: `5499e60a3b8146bf095070dc76d03c85eae57b5f1ec794444276bab362458ec4`
- Direct@3000 SHA-256: `9fa52c905c2603179da39e90da7689a1d65005d5c00d0df2d9fe2e310cf32aeb`
- Official sim_pretrain CNO SHA-256: `af85374bfd06c0e386ec803d777396c21484978392213025697c5a7470106b6b`
- Official scorer SHA-256: `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`
- Split: frozen 50 train / 16 dev; locked-final not accessed
- Codabench: not accessed
- Remote artifact namespace: `realpde_runs/idle_gpu_overnight_20260906/`

The queue ran in the official `realpde-pytorch-h5py:0831` GPU container. The
repository was mounted from an exact detached worktree at the execution
commit; research code and checkpoints were not modified.

## Verification

- `tests/test_arch_a2_multiscale.py`: 4 passed
- `tests/test_horizon_curriculum.py`: 4 passed
- `py_compile`: passed using a temporary `PYTHONPYCACHEPREFIX` because the
  mounted execution worktree was read-only
- A2 preflight: passed; zero-init parity, pressure preservation, UV-only
  branch behavior, optimizer audit, gradient and checkpoint roundtrip passed

## A2 Multi-scale result

Configuration was Direct@1500 + 1500 updates, seed `20260901`, AdamW
`1e-5`, batch `8`, N2 loss, and the frozen 2x coarse Past20-u/v residual
branch. The final checkpoint was absolute update 3000.

| Metric | Direct@3000 reference | A2@3000 | Change | Gate |
|---|---:|---:|---:|---|
| Rel-L2 | `0.1758288145` | `0.1770425439` | `-0.6903%` | fail |
| TKE | `0.5946488380` | `0.5946809649` | `-0.0054%` | protected |
| MVPE | `0.1516314447` | `0.1546803266` | `-2.0107%` | fail |

Fixed gate: `NO_GO` (required Rel-L2 and MVPE improvement ≥ 3%; TKE
degradation limit −2%). Trajectory wins were Rel-L2 `2/16`, TKE `13/16`,
and MVPE `2/16`. The intermediate A2@2500 row was better than the final row,
but no checkpoint was selected from that intermediate result; the frozen gate
uses the final absolute update 3000.

Key artifacts:

- `a2_multiscale/preflight/preflight.json`
- `a2_multiscale/run/run_metadata.json`
- `a2_multiscale/run/update_curve.csv`
- `a2_multiscale/run/trajectory_comparison.csv`
- `a2_multiscale/run/gate_result.json`
- `a2_multiscale/run/summary.json`
- `a2_multiscale/run/model_update_03000.pth` (remote-only checkpoint; not in Git)

## Horizon Curriculum result

HCUR did not enter formal training. Its preflight stopped with:

```text
RuntimeError: P0-A adapted initialization drifted from raw CNO:
0.0002474784851074219
```

The HCUR output contains only its execution commit and an empty preflight
directory; there is no valid HCUR checkpoint, metric, gate, or summary. This
is recorded as `PREFLIGHT_FAILED / REVIEW_REQUIRED`; no fix or rerun was
performed in this queue.

The complete stdout/stderr is retained in the remote queue log
`realpde_runs/idle_gpu_overnight_20260906.log`.

## Decision

A2 is a reproducible negative result and should not be promoted. HCUR is
blocked pending an explicit review of the initialization-parity contract.
No Codabench package, locked-final audit, or third experiment was produced.
