# Structured Temporal Dynamics R2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce CPU-only R1 evidence summaries and execute the fixed R2 C0/V1/A1/A2 matched experiment with auditable preflight, artifacts, and `REVIEW_REQUIRED` handoff.

**Architecture:** Extend the existing runner with a canonical batch-8 path, latent temporal modules inserted immediately before CNO `project`, and optimizer restoration that loads the backbone state before adding fresh temporal parameters. Add a CPU summary tool that consumes existing per-arm CSVs and emits only factual deltas/wins.

**Tech Stack:** Python, PyTorch, NumPy, CSV/JSON, SSH remote GPU execution, official v9 scorer.

**Spec:** User request for Structured Temporal Dynamics R2 in the current task.

## Global Constraints

- Fixed 50 Train / 16 Dev, P0-A, N2, stride=20, seed `20260901`, AdamW `1e-5`, physical batch=8, no gradient accumulation.
- Restore Direct@1500 model and optimizer state; add temporal parameters after restore.
- Evaluate absolute updates `2000/2500/3000`; never access locked-final/full-data/SPS/Codabench.
- Remote root is `/home/chyfuture/realpde_runs/structured_temporal_dynamics_round2_20260908/`.
- Final status is `REVIEW_REQUIRED`; do not emit KEEP/NO-GO/PROMISING conclusions.

### Task 1: R1 CPU evidence summary

**Files:** Create `code/tools/realpde_structured_temporal_summary.py` and its tests.

- [ ] Read R1 aggregate/horizon/trajectory CSVs from an explicit `--r1-root`.
- [ ] Emit `r1_horizon_delta.csv`, `r1_trajectory_summary.csv`, and `r1_horizon_band_summary.csv` relative to C0.
- [ ] Test exact improvement/delta/win formulas on a tiny fixture.

### Task 2: R2 runner and modules

**Files:** Modify `code/tools/realpde_structured_temporal_dynamics.py` and tests.

- [ ] Add latent inspection and project-boundary forward path.
- [ ] Implement V1 frozen deterministic vorticity calibration and A1 temporal Conv3d/A2 temporal attention with zero-init residual projection.
- [ ] Enforce physical batch 8 and zero accumulation for R2.
- [ ] Restore backbone optimizer state first, then add a fresh temporal param group.
- [ ] Record preflight, adapter norms, gradients, parameter counts, and checkpoint round-trip evidence.

### Task 3: Smoke and remote execution

- [ ] Run failing-then-passing focused tests and bounded local preflight/smoke.
- [ ] Copy only code/fixtures needed to remote, run R1 summary and R2 arms sequentially on idle GPU, and preserve logs/artifacts.
- [ ] Run R2 CPU matched summaries from remote CSVs.

### Task 4: Handoff and verification

- [ ] Create `code/docs/coordination/CHATGPT_HANDOFF_STRUCTURED_TEMPORAL_DYNAMICS_R2.md` with execution facts and tables only.
- [ ] Verify artifact completeness, finite outputs, status, git diff, tests, commit SHA, and push `main`.
