# Colleague Incremental Screening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch a reproducible short screen for residual TKE weighting, random temporal phases, and residual-aware uncertainty.

**Architecture:** Extend the handed-off colleague pipeline without changing its model family or scoring definitions.  A continuation runner loads the full residual checkpoint, uses either fixed or deterministic random-phase windows, and emits matched final evidence; the head runner optionally appends residual velocity correction channels from the existing frozen cache.

**Tech Stack:** Python 3, PyTorch, NumPy, h5py, pytest, shell, SSH, OSS CLI.

**Spec:** `docs/colleague_screening/README.md`

## Global Constraints

- Residual arms use 5,000 updates and evaluate every 1,000 updates.
- R0/R1/R2/R3 differ only in TKE weight or temporal phase as registered.
- H0/H1 differ only by adding `delta_u` and `delta_v` to head inputs.
- New optimization uses the colleague's historical all81 train split and the
  same seed-41 Dev16 subset; the direct overlap is recorded as a
  competition-oriented screening limitation.
- Do not access private external data, Codabench, simulation data, or upload OSS artifacts.
- Every remote run records commit, data manifest, checkpoint SHA, command, log, and output path.

## Review Focus

- A continuation checkpoint must load both the frozen base and corrector exactly.
- Fixed and forced-zero random samplers must select the same windows and use global shuffle.
- Random phase must be deterministic by seed/epoch and preserve valid contiguous Past20/Future20 windows.
- Adding residual channels must change only the head input dimension and preserve zero-delta equivalence.
- Remote directories must be unique and refuse overwrite.

---

### Task 1: Residual continuation and sampler

**Files:**
- Modify: `tools/colleague_80pt/realpde_h5_feature_adapter_train.py`
- Modify: `tools/colleague_80pt/residual_multi.py`
- Test: `tests/test_colleague_incremental_screen.py`

**Interfaces:**
- Consumes: colleague `H5WindowDataset`, `ResidualCorrectionModel`, and archived full-model checkpoint.
- Produces: `RandomPhaseWindowSampler`, continuation checkpoint loading, deterministic final-update evaluation artifacts.

- [ ] Write tests for fixed/random phase parity, deterministic phase changes, full-model checkpoint restoration, and registered arm validation.
- [ ] Run the focused tests and confirm they fail because the interfaces are absent.
- [ ] Implement the minimum sampler and continuation behavior.
- [ ] Run focused tests and the full suite.
- [ ] Commit the task.

### Task 2: Residual-aware uncertainty head

**Files:**
- Modify: `tools/colleague_80pt/train_head_fast.py`
- Test: `tests/test_colleague_incremental_screen.py`

**Interfaces:**
- Consumes: cached `base` and corrected `prediction` arrays.
- Produces: `HeadConfig.include_delta` and head inputs with optional `prediction[..., :2] - base[..., :2]`.

- [ ] Write tests for input-channel count, zero-delta equivalence, and nonzero-delta sensitivity.
- [ ] Run the focused tests and confirm they fail for the missing option.
- [ ] Implement the minimum optional delta channels.
- [ ] Run focused tests and the full suite.
- [ ] Commit the task.

### Task 3: Campaign launcher and evidence contract

**Files:**
- Create: `tools/colleague_80pt/run_incremental_screen.py`
- Create: `scripts/run_colleague_incremental_screen.sh`
- Test: `tests/test_colleague_incremental_screen.py`

**Interfaces:**
- Consumes: verified paths for real data, archived checkpoint, code, and output root.
- Produces: four matched residual arm commands, H0/H1 commands, status/provenance files, and refusal to overwrite.

- [ ] Write tests for exact registered arms, command parity, gates, and unique output enforcement.
- [ ] Run the focused tests and confirm the launcher interfaces are absent.
- [ ] Implement dry-run/preflight/launch composition and lightweight gate reporting.
- [ ] Run focused tests, compilation, and the full suite.
- [ ] Commit the task.

### Task 4: Cloud preflight and detached launch

**Files:**
- Modify: `docs/colleague_screening/README.md` after explicit result collection only.

**Interfaces:**
- Consumes: execution commit and verified cloud assets.
- Produces: detached runner PID, log, output path, runtime snapshot, and data manifest.

- [ ] Verify GitHub write access, execution commit, cloud disk/GPU/Python, archive SHA and file manifest.
- [ ] Download and hash released-real data, extract it under `/hy-tmp`, and generate a fresh H5 manifest.
- [ ] Install missing runtime dependencies in an isolated environment and run a bounded smoke test.
- [ ] Start the registered campaign detached and confirm `RUNNING`.
- [ ] Report the PID, log, output path, commit, manifest SHA, and monitoring commands without polling.
