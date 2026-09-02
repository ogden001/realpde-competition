# P0-A + N2 training handoff — 2026-09-02

## Scope and decision

This note records the two completed training lines used to choose the current
Track 1 competition candidate. It deliberately excludes private-test access,
Codabench uploads, data files, checkpoints, and generated submission archives.
All reported errors use the official Track 1 v9 `scoring.py` raw metrics.

**Current model decision:** 20-channel causal P0-A CNO with the frozen N2 loss,
initialized from the official `sim_real_ft/sim_real_cno.pth` checkpoint. The
method beat the 3-channel N2 reference on the fixed dev split in all three raw
errors at the matched 4,100-update budget.

P0-A takes the three raw CNO channels and appends 17 history-only features:
speed, four spatial velocity derivatives, vorticity and absolute vorticity,
strain magnitude, temporal u/v deltas, history u/v mean and standard deviation,
u/v fluctuations, and a history TKE proxy. It is causal and requires no
trajectory metadata at inference.

The N2 objective is:

`MSE + 0.05 * TKE + 0.027514 * Relative-L2 + 0.009757 * MVPE`.

## Process A — fixed 50/16 validation training and continuation

**Purpose:** decide whether P0-A remains useful when compared on the same
official warm-start, fixed ID split, loss, seed, and update budget as the raw
3-channel N2 reference.

**Data/protocol:** 50 train trajectories / 2,052 training windows; 16 dev
trajectories; seed `20260901`; effective batch 8; official v9 dev replay every
820 updates. This is an `OFFICIAL_WARM_START` competition check, not a clean
offline causal feature-engineering claim because the initialization is official
real-PIV fine-tuned.

### Matched-budget result (4,100 updates)

| Raw error | Raw 3-channel N2 | P0-A + N2 | Relative improvement |
|---|---:|---:|---:|
| Rel-L2 | 0.145684 | 0.141550 | 2.84% |
| TKE | 0.576014 | 0.546986 | 5.04% |
| MVPE | 0.122189 | 0.120323 | 1.53% |

This is the primary GO evidence for P0-A + N2. It improves every raw error
under the matched protocol.

### Continuation evidence

The same validation run restored model and AdamW state at 4,100, then at 6,800,
using a deterministic fresh data shuffle because the older runner had not saved
the DataLoader generator state. Subsequent runs write `train_metrics.jsonl` at
every 100 updates; per-batch loss values are diagnostic only, while the table
below is the model-selection signal.

| Update | Rel-L2 | TKE | MVPE |
|---:|---:|---:|---:|
| 4,100 | 0.141550 | 0.546986 | 0.120323 |
| 5,740 | 0.140899 | 0.534357 | 0.112906 |
| 6,800 | 0.134755 | 0.521813 | 0.108504 |
| 7,380 | 0.133431 | 0.510232 | 0.104979 |
| 8,200 | 0.130229 | 0.542222 | 0.101615 |
| 9,020 | 0.130865 | 0.517756 | 0.106612 |
| 9,840 | 0.125992 | **0.507496** | 0.105560 |
| 10,300 | **0.123734** | 0.515453 | **0.096836** |

**Conclusion:** continued training through 10,300 is beneficial. The 10,300
point trades 1.57% higher TKE than the 9,840 minimum for substantially better
Rel-L2 and MVPE. With the official composite formula unavailable, it is the
preferred fixed-last continuation point rather than a claim of a known global
score optimum.

## Process B — 82-trajectory full-data refit and continuation

**Purpose:** fit the selected P0-A + N2 competition method on all 82 released
real PIV trajectories after method selection. No dev, locked-final, private
test, or Codabench labels are used in this process.

**Initial refit:** 3,383 training windows, 6,800 optimizer updates, effective
batch 8, approximately 16.08 data traversals. It completed normally in 5,384
seconds with peak allocated CUDA memory 5,065,201,152 bytes (4.72 GiB).

**Continuation milestones:**

- The 6,800 checkpoint was retained intact.
- A continuation from 6,800 began toward 15,300 updates.
- A distinct `model_update_10300.pth` checkpoint was saved and verified to
  contain iteration 10,300, so it cannot be overwritten by the final save.
- A later time-bounded continuation was safely stopped at 12,481 updates; its
  model and AdamW state were saved before restart.
- The active final continuation resumes from 12,481 toward 15,300 with a
  hard 23:39 Asia/Shanghai stop to reserve time for packaging and an authorized
  manual Codabench submission.

The full-data run has no held-out validation by design. Its per-batch N2 loss
is logged in `train_metrics.jsonl`, but it must not be used as a substitute for
the fixed 50/16 dev evidence above.

## Runtime and memory observations

- Default protocol: micro-batch 4 with accumulation 2 (effective batch 8),
  about 4.72 GiB peak allocated memory and the best observed update throughput.
- A user-authorized high-memory probe used micro-batch 16 / accumulation 1.
  It allocated 18.46 GiB but took 2.35 seconds per update, versus roughly 0.65
  seconds per update for the normal protocol. Larger batch therefore reduced
  throughput on this CNO workload and was not used for the deadline run.
- All training was run in the official-compatible PyTorch CUDA container with
  explicit CLI paths and recorded checkpoint/scorer hashes in metadata.

## Submission status and next action

No Codabench package or submission has been produced in these processes. Once
the time-bounded full continuation finishes, choose one of the available
full-data checkpoints (10,300 fallback or final last checkpoint), build a
submission with the exact P0-A implementation, run the official smoke test,
measure runtime/SPS bounds, then submit only after explicit confirmation.

