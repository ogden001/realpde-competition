# MF Direction Closeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the single matched Direct continuation from the verified MF-01 Control @1500 checkpoint, score it at @2000/@2500/@3000, and close the Mean/Fluctuation direction using existing MF@1500 and MF@3000 evidence.

**Architecture:** Reuse the frozen MF-01 feature/model/loss/scorer path. Add only a Direct continuation wrapper that loads the exact Direct @1500 state, performs 1500 matched updates, and records absolute update numbers. Generate the closeout report from the official v9 raw metrics and trajectory CSVs.

**Tech Stack:** Python, PyTorch, official Track 1 starting-kit v9 scorer, remote RTX 3090 runner, Git.

**Spec:** User request “MF Direction Closeout: Direct@3000 vs MF@3000”.

## Global Constraints

- Fixed CLEAN 50 Train / 16 Dev, P0-A, N2, seed `20260901`, AdamW `1e-5`, batch/effective batch protocol matched to MF-01.
- Use only `T1-ID-MF01-CONTROL-S20260904` Direct @1500; no warm-start/full-data/locked-final/Codabench/SPS.
- Train exactly one model for +1500 updates and do only official v9 scoring plus lightweight comparison.

### Task 1: Direct continuation runner

**Files:**
- Create: `code/tools/realpde_mf_direction_closeout.py`
- Test: existing MF-01 output smoke test and runner preflight.

- [ ] Load the verified Direct @1500 checkpoint, preserve the Direct output interface and P0-A/N2 protocol, run +1500 updates, and evaluate at absolute updates 2000/2500/3000.
- [ ] Record checkpoint and scorer SHA-256 values and reject non-matching protocol/checkpoint metadata.

### Task 2: Execute and report

**Files:**
- Create: `code/docs/coordination/CHATGPT_HANDOFF_MF_DIRECTION_CLOSEOUT.md`
- Modify: `code/docs/modeling/modeling概要.md`
- Modify: `code/docs/track1_experiment_registry.md`

- [ ] Run the one continuation on the remote GPU with the official kit and frozen manifest.
- [ ] Verify four-model official raw metrics, relative deltas, trajectory win counts, checkpoint curve, and final rating.
- [ ] Commit and push `main` after fresh verification.
