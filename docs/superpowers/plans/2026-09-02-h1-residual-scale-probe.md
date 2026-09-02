# H1 Residual Scale Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate train-selected scalar scaling of the existing H1 Point correction without any neural-network training.

**Architecture:** Load the frozen FE-00 CNO and H1 `last@1500` Point head, replay train/dev windows once to obtain CNO predictions, H1 predictions, and targets, then score a fixed alpha grid using the official v9 scorer. Select alpha only from train, and score dev only for CNO, alpha=1, and the frozen alpha-star.

**Tech Stack:** Python, PyTorch, NumPy, HDF5, official Track 1 v9 scorer, Docker on remote RTX 3090.

**Spec:** `/Users/oukairi/.codex/attachments/770607ea-1994-46ce-81ab-00569be3f3b2/pasted-text.txt`

## Global Constraints

- No optimizer step, model update, loss/LR/feature/architecture change, H2, LOCAL5, locked-final, or Codabench.
- Fixed manifest SHA `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`, seed `20260901`, B3_PACKED, 50 train / 16 dev / 16 locked-final untouched.
- Alpha grid is exactly `[0.0,0.1,...,1.0]`; alpha selection uses train only and TKE degradation `<=5%`, then maximum Rel-L2 improvement with specified tie-breaks.

### Task 1: Tests and standalone probe

**Files:**
- Create: `code/tools/realpde_hybrid_cno_point_h1_scale_probe.py`
- Test: `code/tests/test_hybrid_cno_point_h1_scale_probe.py`

- [ ] Write failing tests for alpha replay equivalence, train-only selection/tie-break, and gate sign semantics.
- [ ] Implement independent probe reusing the frozen H1 loader/head and official scorer.
- [ ] Run unit tests, pycompile, and `git diff --check`.

### Task 2: Remote bounded execution and handoff

- [ ] Run bounded smoke/replay sanity.
- [ ] Execute the fixed train scan and one dev evaluation only if `alpha_star > 0`.
- [ ] Generate required CSV/JSON/report artifacts, append registry/handoff, commit and push only source/docs.
