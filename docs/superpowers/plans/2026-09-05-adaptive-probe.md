# Adaptive Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with review checkpoints.

**Goal:** Implement the frozen Residual Corrector + Adaptive Uncertainty probe, validate it on the frozen 50/16 protocol, conditionally refit on all 82 released trajectories, and package clean-room-smoke-tested PRIMARY/BACKUP artifacts.

**Architecture:** Add one bounded runner containing causal tensor-only flow features, small 3D residual heads, frozen-backbone training, fixed-budget validation/full stages, official-v9 scoring, Gate evaluation, and self-contained package generation. Keep raw execution logs remote and commit only review evidence, reports, and review logs.

**Tech Stack:** Python 3, PyTorch, HDF5, NumPy, pytest, official Track 1 v9 scorer, vendored CNO.

**Spec:** `docs/sota迭代/NEXT_ACTION.md` and `docs/sota迭代/TEAMMATE_ADAPTIVE_PROBE_REFERENCE_20260905.md`

## Global Constraints

- Frozen backbone: P0-A + N2 validation update `30900`, full update `43260`.
- Validation: frozen 50/16 manifest, 2400 corrector updates, 1400 head updates, batch 8, eval corrector at 600/1200/1800/2400.
- Full corrector: all 82 released trajectories / 3383 windows, fixed 3960 updates, no checkpoint selection.
- Runtime inputs use only Past20 `[u,v,p]`, tensor shape, and normalized tensor indices; no metadata, grid, locked-final, private-test, or Codabench.
- Pressure prediction and pressure bounds are exactly zero.

---

### Task 1: Lock tensor semantics with unit tests

**Files:**
- Create: `tests/test_adaptive_probe.py`
- Create: `tools/realpde_adaptive_probe.py`

- [ ] **Step 1: Write failing tests** for feature shape/causality, pressure-zero correction, sigma clamp, frozen backbone, loss finiteness, and Gate thresholds.
- [ ] **Step 2: Run `pytest -q tests/test_adaptive_probe.py`** and confirm import/implementation failures.
- [ ] **Step 3: Implement the minimal pure-PyTorch helpers and modules.**
- [ ] **Step 4: Re-run the focused tests and then the full registered test set.**

### Task 2: Add fixed-budget validation/full runner and package smoke

**Files:**
- Modify: `tools/realpde_adaptive_probe.py`
- Create: `tests/test_adaptive_probe_runner.py`

- [ ] **Step 1: Add CLI validation, full-refit, package, and smoke subcommands with explicit paths.**
- [ ] **Step 2: Add official-v9 evaluation, trajectory TKE degradation counting, fixed calibration grid, provenance, and review-log generation.**
- [ ] **Step 3: Add package wrapper and deterministic direct-vs-package smoke.**
- [ ] **Step 4: Run unit, focused runner, and bounded synthetic smoke tests.**

### Task 3: Execute the preregistered remote experiment

**Files:**
- Modify: `docs/sota迭代/README.md`
- Create: `docs/sota迭代/reviews/overnight_integrated_20260905/README.md`, metrics/provenance files, and complete `*.review.log` files.

- [ ] **Step 1:** Resolve runtime paths and run runtime snapshot/preflight without locked-final/private-test access.
- [ ] **Step 2:** Run validation corrector and base uncertainty head; evaluate fixed 2400/1400 checkpoints.
- [ ] **Step 3:** Apply the fixed Gate; run full corrector only if it passes, otherwise skip it and retain base BACKUP.
- [ ] **Step 4:** Run only the registered fixed floor×mult grid with the official scorer; build and smoke PRIMARY/BACKUP as applicable.
- [ ] **Step 5:** Commit review evidence with `git add -f` for `*.review.log`, sync `main`, commit, push, and stop at `REVIEW_REQUIRED`.
