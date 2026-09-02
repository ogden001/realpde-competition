# H1 Frozen CLEAN CNO + LOCAL3 Point Residual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (recommended) to implement this plan task-by-task.

**Goal:** Run one bounded H1 experiment that freezes the registered CLEAN FE-00 CNO and trains only a zero-initialized LOCAL3 point residual head.

**Architecture:** A standalone runner loads the FE-00 CNO checkpoint, verifies frozen parameters and zero-init exact equivalence, precomputes train-side CNO predictions in RAM, trains a 402-input LOCAL3 head with uv MSE only, and performs the preregistered 1500-step dev gate before any optional continuation.

**Tech Stack:** Python, PyTorch, NumPy, HDF5, official Track 1 v9 scorer, Docker on remote RTX 3090.

**Spec:** User-provided H1 task in `/Users/oukairi/.codex/attachments/4991025e-cc19-4f14-bdf5-88aba985b9c1/pasted-text.txt`.

## Global Constraints

- CLEAN FE-00 CNO-only checkpoint; no `sim_real_ft`, no CNO retraining, no joint training.
- Frozen manifest SHA `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`, 50 train / 16 dev / 16 locked-final untouched, seed `20260901`, B3_PACKED, batch 8.
- Point head `402→256→256→256→128→40` GELU; 3×3 replicate-padded raw u/v history plus CNO future uv and normalized x/y; output corrects uv only and preserves CNO pressure.
- Loss is absolute corrected-field uv MSE only; AdamW `lr=1e-4`, `weight_decay=0`, 1500-step screen, gate-controlled continuation to 7500.
- No locked-final, Codabench, LOCAL5, extra features, loss/LR sweep, or automatic H2.
- Detached Docker uses `--memory=48g --memory-swap=48g --shm-size=8g`; Codex does not babysit.

### Task 1: Registry and tests

**Files:** `code/docs/track1_experiment_registry.md`, `code/docs/coordination/STATUS.md`, `code/tests/test_hybrid_cno_point_h1_runner.py`.

- [ ] Append H1 `IN_PROGRESS` entry with checkpoint provenance and exact gate.
- [ ] Write failing tests for zero-init equivalence, CNO parameter isolation, 402-input/40-output head shape, and gate sign semantics.
- [ ] Run tests and observe failure before implementation.

### Task 2: Standalone runner

**Files:** `code/tools/realpde_hybrid_cno_point_h1_runner.py`.

- [ ] Reuse official CNO loading and B3_PACKED window semantics; load and hash the FE-00 checkpoint.
- [ ] Implement frozen CNO prediction cache, H1 head, train-only MSE loop, atomic status, screening/full evaluation, diagnostics, and reports.
- [ ] Ensure optimizer contains only Point Head parameters and CNO gradients remain `None`.

### Task 3: Verification and launch

- [ ] Run pycompile, unit tests, and bounded smoke with checkpoint/zero-init/shape/isolation checks.
- [ ] Copy only source/manifest to the remote run directory; mount data/checkpoint/kit read-only.
- [ ] Launch detached H1 with the 48 GiB hard limit, confirm PID/container and initial status once, then stop polling.
