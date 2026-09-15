# Vorticity Supervision Long Final Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 从 R2 的 C0@3000 与 V1@3000 checkpoint 继续进行 matched long-range validation 至 absolute update 15000，并依据预注册 late-checkpoint gate 机械生成唯一 `MERGE_CANDIDATE` 或 `PARK` 结论。

**Architecture:** 新增长训 runner 复用 R2 的 CNO forward、N2 loss 与 vorticity 定义，但把初始化、optimizer state、absolute iteration、固定 lambda、DataLoader seed 和 milestone 保存明确写入 provenance。独立 summary/replay 工具只消费公开 Dev 评估产物，生成 matched aggregate、trajectory stability、late gate 和最终 handoff 所需事实，不访问 locked-final、full-data、SPS 或 Codabench。

**Tech Stack:** Python, PyTorch, NumPy, CSV/JSON, official Track 1 starting kit v9 scorer, SSH remote RTX 3090.

**Spec:** User request for Vorticity Supervision long final validation dated 2026-09-14.

## Global Constraints

- Fixed 50 Train / 16 Dev manifest, P0-A, N2, stride=20, seed `20260901`, AdamW `lr=1e-5`, physical batch=8, no gradient accumulation.
- Start exactly from R2 `C0@3000` and `V1@3000`, restoring model and optimizer state with absolute iteration 3000.
- V1 uses fixed `lambda_vort=15.5385751724`; no recalibration, loss, LR, architecture, or early stopping changes.
- Evaluate and checkpoint absolute updates `3000, 4500, 6000, 9000, 12000, 15000`; final gate uses only 9000/12000/15000.
- Do not access locked-final, full-data, SPS, or Codabench. Use the target RTX 3090 only if physical batch 8 can run.
- Final registry/modeling updates must record only the mechanical gate result and `Structured Temporal Dynamics exploration CLOSED.`

### Task 1: Long-final runner and tests

**Files:**
- Create: `tools/realpde_vorticity_long_final.py`
- Modify: `tests/test_structured_temporal_dynamics.py`

- [ ] Add explicit protocol validation for batch=8, accumulation=1, fixed lambda, seed and absolute iteration.
- [ ] Restore R2 model and optimizer state, verify optimizer state entries and replay at update 3000 against R2 aggregate references before training.
- [ ] Recreate the same seeded train DataLoader separately for each arm, preserve physical batch and window protocol, train to absolute 15000, and save every requested milestone checkpoint/evaluation.
- [ ] Record preflight, run metadata, training curves, runtime, checkpoint round-trip, finite predictions, zero pressure and prohibited-access flags.
- [ ] Add focused tests for fixed lambda/no recalibration, absolute milestone mapping, protocol rejection, and gate formula helpers.

### Task 2: Matched summary and final replay

**Files:**
- Create: `tools/realpde_vorticity_long_final_summary.py`
- Create: `tools/realpde_vorticity_long_final_replay.py`

- [ ] Consume only C0/V1 Dev outputs to write `matched_metrics.csv` with raw errors and `(C0-V1)/C0*100` improvements.
- [ ] Preserve per-trajectory metrics and write `trajectory_stability.csv` with all required win and protected-count columns.
- [ ] Compute late medians and the exact two-part gate from 9000/12000/15000 into `late_checkpoint_gate.json`, with no alternative threshold.
- [ ] Replay C0@15000 and V1@15000 independently with the official v9 scorer and compare replay values against first-evaluation values within an explicit floating-point tolerance.

### Task 3: Smoke test and remote execution

- [ ] Run focused local tests, Python compilation, and a bounded runner/preflight smoke without touching locked-final/full-data/SPS/Codabench.
- [ ] Verify remote container, GPU availability, R2 checkpoint paths and SHA-256 values, and ensure no competing process occupies the target 3090.
- [ ] Run C0-LONG then V1-LONG with the exact frozen command and remote root `/home/chyfuture/realpde_runs/vorticity_long_final_20260914/`.
- [ ] Run CPU summary and independent final replay; check all required files, six milestone directories, two final `predictions.npz`, finite values, metadata flags and gate JSON.

### Task 4: Handoff, closeout and GitHub

**Files:**
- Create: `docs/coordination/CHATGPT_HANDOFF_VORTICITY_LONG_FINAL.md`
- Modify: `docs/track1_experiment_registry.md`
- Modify: `docs/modeling/modeling概要.md`

- [ ] Write a short facts-only handoff containing protocol/provenance, milestone table, trajectory stability, late medians, gate checks, final replay, runtime, artifacts, tests and commit SHA.
- [ ] Set the single registered/modeling conclusion to the computed `FINAL_GATE` and note `Structured Temporal Dynamics exploration CLOSED.` without additional research interpretation.
- [ ] Run verification before commit: tests, compile, diff check, artifact completeness, clean Git state and final gate provenance.
- [ ] Commit and push `main`; report `FINAL_GATE`, late table, medians, trajectory stability, remote artifact path and commit SHA.
