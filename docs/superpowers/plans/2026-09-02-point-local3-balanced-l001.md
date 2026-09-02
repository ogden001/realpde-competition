# POINT-LOCAL3-BALANCED-L001 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement and launch one bounded random-init LOCAL3 experiment changing only the TKE loss weight from 0.05 to 0.001.

**Architecture:** Add a standalone runner that imports the frozen LOCAL3 model and B3_PACKED dataset helpers, records atomic status and train diagnostics, evaluates dev only after the 1500-step gate, and continues the same optimizer state to 7500 only when the preregistered gate passes.

**Tech Stack:** Python, PyTorch, NumPy, HDF5, official Track 1 v9 scorer, Docker on the remote RTX 3090 host.

**Spec:** User-provided POINT-LOCAL3-BALANCED-L001 task in `/Users/oukairi/.codex/attachments/a2b517b7-7bb2-42a6-aa28-cb01ca997c44/pasted-text.txt`.

## Global Constraints

- Use manifest seed `20260901`, frozen 50/16/16 split, B3_PACKED, batch 8, AdamW `1e-4`, LOCAL3 `362→256→256→256→128→40`, GELU, raw residual u/v, replicate padding.
- Start from random initialization; do not load old LOCAL3 checkpoints.
- Train `MSE + 0.001*TKE`; no loss/LR/architecture sweep and no optimizer step during gradient snapshots.
- Train 1500 updates, evaluate dev once, and continue the same state to 7500 only if the strict screening gate passes.
- Never access locked-final or Codabench; do not auto-run LOCAL5 or other follow-up experiments.
- Long jobs run detached with host/container RAM hard limit 48 GiB; Codex does not poll continuously.

### Task 1: Register experiment and define runner contract

**Files:**
- Modify: `code/docs/track1_experiment_registry.md`
- Modify: `code/docs/coordination/STATUS.md`
- Create: `code/tools/realpde_point_local3_balanced_runner.py`
- Test: `code/tests/test_point_local3_balanced_runner.py`

**Interfaces:**
- Runner CLI accepts explicit `--data-root`, `--manifest`, `--kit-root`, `--out-dir`, `--seed`, `--device`, and optional `--smoke`.
- Runner writes atomic `status.json`, training CSV, gradient snapshots, screening/full evaluation artifacts, and final README/report/summary metadata.

- [ ] Write tests for lambda, model input/output shape, deterministic order, and screening gate boundary behavior.
- [ ] Run tests and confirm they fail because the new runner module is absent.
- [ ] Implement the smallest standalone runner reusing `PackedDataset`, `fixed_order`, `loader`, `Local3MLP`, `loss_parts`, official scorer, and trajectory helpers.
- [ ] Run tests, then pycompile and smoke-test with a bounded local/remote data subset.
- [ ] Commit only the new runner/tests and protocol docs, preserving unrelated dirty files.

### Task 2: Launch bounded remote job

**Files:**
- Create: remote run directory under `/home/chyfuture/realpde_runs/point_local3_balanced_l001_s20260902`
- Create: remote detached log and status artifacts

- [ ] Copy the runner, frozen manifest, and required source/kit mounts without copying datasets or checkpoints into Git.
- [ ] Launch Docker with `--memory=48g --memory-swap=48g --shm-size=8g` and the explicit CLI.
- [ ] Confirm the container/PID and initial status exactly once; return host, PID/container ID, run directory, log/status paths, and iTerm monitoring commands.
- [ ] Stop Codex-side polling; recover and summarize only on a later user request.
