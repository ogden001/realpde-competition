# Vorticity Low-Memory Paired Final Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with verification checkpoints.

**Goal:** Re-run matched C0 and V1 long validation under one shared-GPU low-memory protocol and mechanically produce the final Vorticity Supervision gate.

**Architecture:** Stop only the prior project-owned Vorticity launcher/watcher, preserve the old batch-8 C0 evidence, and add a low-memory runner/summary/replay path that uses identical micro-batch and accumulation settings for both arms. Restore both R2@3000 model and optimizer states, train each arm to absolute update 12000, compare only 6000/9000/12000, replay 12000 independently, then write the final gate and closeout documents.

**Tech Stack:** Python, PyTorch/CUDA, official v9 scorer, remote RTX 3090 container, pytest, Git.

**Spec:** `/Users/oukairi/.codex/attachments/72ad16be-5f19-4e1a-8c34-e0ea741d2ca8/pasted-text.txt`

## Global Constraints

- Stop only project-owned old Vorticity task processes; never kill unrelated GPU/Python processes.
- Preserve `/home/chyfuture/realpde_runs/vorticity_long_final_20260914/` and do not use it for the new gate.
- Use the same protocol for C0-LOWMEM and V1-LOWMEM: effective batch 8, selected micro-batch/accumulation pair, seed `20260901`, P0-A, N2, stride 20, AdamW `1e-5`, pressure 0.
- Try `micro_batch=4, accumulation_steps=2` first with a per-process reserved-memory cap of 12 GiB; if OOM, restart both arms with `micro_batch=2, accumulation_steps=4`.
- Train from R2 C0@3000 and V1@3000 to absolute update 12000; do not cherry-pick or early-stop.
- Final gate uses only 6000/9000/12000 and the pre-registered thresholds; no subjective override.
- Do not access locked-final, full-data, SPS, or Codabench; do not start later experiments or SOTA merge.
- Do not commit checkpoints, predictions, or other large artifacts.

---

### Task 1: Stop old task and verify environment

**Files:**
- Read: `MEMORY.md`, `code/AGENTS.md`, requested handoff/source/modeling/registry files.
- Create: remote `/home/chyfuture/realpde_runs/vorticity_lowmem_final_20260915/` only after old-task verification.

- [ ] Inspect remote `ps`, `pgrep`, `tmux`, `screen`, Docker exec commands, and command lines for the exact old-task identifiers.
- [ ] Terminate only confirmed old project-owned launchers/watchers, record PID/command/action in `preflight/old_task_stop.json`, and verify no matching old auto-launch command remains.
- [ ] Verify target R2 checkpoints, manifest, scorer, official container, GPU model, and SHA-256 values without touching unrelated processes.

Verification:

```bash
ssh gpu 'nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv'
ssh gpu 'pgrep -af "vorticity_long_final_20260914|realpde_vorticity_long_final.py|V1-LONG" || true'
```

Expected: `old_task_stopped=true`, `old_v1_auto_launch_remaining=false`, old C0 directory preserved.

### Task 2: Implement paired low-memory runner and preflight

**Files:**
- Create/modify: `code/tools/realpde_vorticity_lowmem_final.py`
- Modify: `code/tests/test_structured_temporal_dynamics.py`

- [ ] Add CLI fields for `--micro-batch`, `--accumulation-steps`, `--effective-batch`, `--memory-cap-gib`, `--final-update 12000`, and fixed milestone list.
- [ ] Enforce `micro_batch * accumulation_steps == 8`, identical protocol metadata, and one shared memory configuration for both arms.
- [ ] Restore model state, optimizer state, and absolute iteration 3000 from the R2 checkpoint; create the training DataLoader after all preflight probes so RNG progression is untouched.
- [ ] Implement accumulation with optimizer stepping every configured accumulation interval and record effective batch 8.
- [ ] Apply CUDA allocator cap/equivalent before model allocation and record allocated/reserved peak memory plus GPU model.
- [ ] Keep C0 N2-only and V1 N2 plus exact `15.5385751724 * vorticity_MSE`; save all required milestone checkpoints/evals and final predictions.
- [ ] Add tests for effective-batch validation, fallback pair equality, restored iteration/optimizer, fixed lambda, memory-cap metadata, and old-task matching that excludes unrelated commands.

Verification:

```bash
PYTHONPATH=code/tools python3 -m pytest code/tests/test_structured_temporal_dynamics.py -q -k 'lowmem or long_final or r2_rejects'
python3 -m py_compile code/tools/realpde_vorticity_lowmem_final.py
```

### Task 3: Run paired preflight and select one configuration

**Files:**
- Remote: `/home/chyfuture/realpde_runs/vorticity_lowmem_final_20260915/preflight/`

- [ ] Run C0 and V1 replay/preflight independently with micro-batch 4 and accumulation 2, checking R2 metric parity, model/optimizer restoration, iteration 3000, lambda, DataLoader ordering, pressure, finite outputs, round-trip checkpoint, and reserved memory <=12 GiB.
- [ ] If either arm OOMs, discard both preflight results for the first pair and rerun both with micro-batch 2 and accumulation 4; never mix configurations.
- [ ] If the final pair cannot stay within the 12 GiB cap, stop and report without training.

Expected preflight artifact includes `selected_micro_batch`, `selected_accumulation_steps`, `effective_batch=8`, `old_task_stopped=true`, and both arm checks passing.

### Task 4: Train C0-LOWMEM then V1-LOWMEM

**Files:**
- Remote: `/home/chyfuture/realpde_runs/vorticity_lowmem_final_20260915/C0-LOWMEM/`
- Remote: `/home/chyfuture/realpde_runs/vorticity_lowmem_final_20260915/V1-LOWMEM/`
- Remote: `/home/chyfuture/realpde_runs/vorticity_lowmem_final_20260915/logs/`

- [ ] Run C0 from R2 C0@3000 to 12000 with the selected pair and wait for all 3000/4500/6000/9000/12000 evaluations.
- [ ] Validate C0 completion and metadata before starting V1; do not change configuration based on C0 metrics.
- [ ] Run V1 from R2 V1@3000 to 12000 with the identical pair, fixed lambda, data ordering, and milestones.
- [ ] Verify each arm has checkpoints, official v9 raw scores, trajectory metrics, training curve, runtime, metadata, and `eval_12000/predictions.npz`.

### Task 5: Generate matched summary and replay

**Files:**
- Create/modify: `code/tools/realpde_vorticity_lowmem_final_summary.py`
- Create/modify: `code/tools/realpde_vorticity_lowmem_final_replay.py`
- Remote: `/home/chyfuture/realpde_runs/vorticity_lowmem_final_20260915/summary/`
- Remote: `/home/chyfuture/realpde_runs/vorticity_lowmem_final_20260915/replay/`

- [ ] Generate root/summary `matched_metrics.csv`, `trajectory_stability.csv`, `late_checkpoint_gate.json`, paired training curves, combined runtime, and combined metadata.
- [ ] Compute improvement exactly as `(C0 - V1) / C0 * 100` and use only aggregate official raw metrics for the gate.
- [ ] Run independent 12000 checkpoint-load → Dev inference → official v9 scorer replay for both arms and record maximum differences against original `eval_12000`.
- [ ] Mechanically compute medians at 6000/9000/12000, the three threshold checks, the 2-of-3 check, and exactly `MERGE_CANDIDATE` or `PARK`.

### Task 6: Write closeout docs and verify

**Files:**
- Create: `code/docs/coordination/CHATGPT_HANDOFF_VORTICITY_LOWMEM_FINAL.md`
- Modify: `code/docs/track1_experiment_registry.md`
- Modify: `code/docs/modeling/modeling概要.md`

- [ ] Write protocol/provenance, old-task stop evidence, matched table, late median, trajectory stability, replay consistency, gate checks, runtime/memory, artifact paths, tests, and commit SHA.
- [ ] Record only the mechanical Vorticity result and `Structured Temporal Dynamics exploration = CLOSED`; retain historical old C0 as non-gate convergence evidence.
- [ ] Confirm no `REVIEW_REQUIRED` remains as the final scientific conclusion and do not start SOTA merge.
- [ ] Run focused tests, `py_compile`, `git diff --check`, remote artifact completeness, metadata prohibition checks, and clean-worktree verification.

### Task 7: Commit and push

**Files:**
- Git: intended code, tests, handoff, registry, and overview only.

- [ ] Review `git diff --stat` and ensure no checkpoints/predictions/large artifacts are staged.
- [ ] Commit with a message describing the low-memory paired final.
- [ ] Push `main` and verify `HEAD == origin/main`.
- [ ] Report only final gate, selected low-memory protocol, late matched improvements, median, trajectory stability, memory, replay consistency, artifact root, and commit SHA.
