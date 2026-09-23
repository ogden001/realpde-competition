# NEXT_ACTION — CURRENT80 RESIDUAL DENSE-CONTINUATION SCREEN

Status: `READY_FOR_EXECUTION / REVIEW_REQUIRED_AFTER_RUN`

## 1. Scientific question

验证：

> 当前成熟的 colleague 80pt residual corrector 已经在 stride=20 上长训收敛后，再用真实 PIV 的 dense temporal starts（stride=1）继续训练，是否比同预算的 sparse stride=20 continuation 更有效？

这是一个 **continuation curriculum（续训课程）** 实验，不是 dense-from-scratch 实验。

### Existing evidence that MUST be respected

同事 handoff 已明确记录：

- **Residual stride=1 from scratch：-2.9%，NO-GO**
- 历史上 dense 只有以 continuation 形式出现过正信号
- 当前 80pt residual 的正式训练仍是 stride=20、38,400 updates、best@34,000

因此本任务禁止：

- 从 CNO 上重新初始化一个 stride=1 residual；
- 重跑已经失败的 dense-from-scratch；
- 把本任务解释成“stride=1 residual 从零训练”。

本任务唯一的新问题是：

```text
mature current80 residual
        ↓
short dense stride=1 continuation
```

---

## 2. Frozen baseline

### Stage-1 frozen CNO

SHA256：

`ff28aaf0114d57e320e5ea9cf9945d874a7482f22e3f0aeaed5ebbe286573f8a`

### Current 80pt residual start checkpoint

SHA256：

`909fdc7f8a6a42335e4ea4ce7471fc9a507135e9a29a4927bc7b7318be9c85b2`

Unified offline current80:

- Rel-L2: `0.0804204195737838`
- TKE: `0.4491582512855530`
- MVPE: `0.0711169168353080`

---

## 3. Historical matched sparse control — DO NOT RERUN

We already have an exact same-budget sparse continuation control from 2026-09-21 R0.

Evidence:

`docs/colleague_screening/results/20260921`

Historical execution commit:

`ef9c54f621efcb82703cb2de40ce979c40df3c6a`

R0 semantics:

- same base CNO
- same current80 residual start checkpoint
- updates = `5000`
- eval interval = `1000`
- batch = `8`
- lr = `2e-4`
- weight decay = `1e-5`
- hidden = `96`
- blocks = `2`
- max_delta = `0.04`
- TKE weight = `0.06`
- seed = `41`
- optimizer/scheduler reset for continuation
- train window mode = fixed
- **stride = 20**

R0 @5k:

- Rel-L2 = `0.0802410691976547`
- TKE = `0.4503658413887024`
- MVPE = `0.0710276663303375`

This historical R0 is the matched control. Do not waste GPU reproducing it unless its provenance validation fails.

---

## 4. New candidate — Dense continuation

Only one scientific change relative to historical R0:

```text
stride: 20 → 1
```

Everything else remains matched.

Fixed:

- resume from current80 residual SHA above
- frozen all81 CNO SHA above
- updates = `5000`
- eval every = `1000`
- batch = `8`
- lr = `2e-4`
- weight decay = `1e-5`
- hidden = `96`
- blocks = `2`
- max_delta = `0.04`
- TKE = `0.06`
- point/MSE/temporal/grad/residual/delta losses unchanged
- train alpha = `1.0`
- fixed window mode
- stride = **1**
- seed = `41`
- no AoA augmentation
- no rotation augmentation
- no random-phase sampler semantics beyond stride=1 dense starts
- no architecture change
- no loss change

The experiment is intentionally short. Do not automatically extend beyond 5k.

---

## 5. Data protocol

Use the exact colleague competition-oriented all81 / Dev16-overlap protocol used by historical R0.

Expected split manifest SHA256:

`d127e851f3f5eefb011313b1b0b1d79e65db6d99c0754d264119ff12046a83a2`

Expected data manifest SHA256:

`3612185c939f6c4cd4890afd3968d160a44e88d3e6f7733b8cb9ce71501d532c`

Semantics:

- train = all 81 usable released PIV trajectories
- Dev16 overlaps train by the frozen colleague protocol
- this is a **competition-oriented matched screen**, not a clean generalization estimate
- locked-final/private data forbidden

Do not substitute another split.

---

## 6. Why this experiment is worth running

Current Stage 2 sparse residual:

- ~3,341 stride=20 windows
- 38.4k long training updates
- repeatedly revisits a relatively small set of temporal phases

Stage 1 CNO already saw dense stride=1 starts.

The candidate asks whether, after the residual has already learned a stable correction from sparse windows, a short dense continuation can expose it to more **real temporal phases** without forcing it to learn the residual from scratch.

This is not synthetic augmentation. Every dense window is a real PIV Past20→Future20 pair.

---

## 7. Required runner

Use:

`tools/colleague_80pt/run_dense_residual_continuation_screen.py`

The runner must:

1. verify base CNO SHA;
2. verify current80 residual SHA;
3. verify exact historical split/data-manifest SHA;
4. verify all81 data;
5. run one stride=1 continuation arm for 5k;
6. eval at 1k/2k/3k/4k/5k;
7. build deterministic training review log;
8. build `training_progress.csv`;
9. build `comparison_at_5k.csv` containing:
   - frozen current80
   - historical sparse-5k R0
   - new dense-5k
10. replay:
   - current80
   - dense_best
   - dense_final
11. write standard horizon/trajectory/mean/fluctuation/energy diagnostics;
12. stop at REVIEW_REQUIRED.

No automatic scientific verdict is allowed in the runner.

---

## 8. Review criteria for Sol

Primary comparison:

```text
dense-5k vs historical sparse-5k R0
```

Secondary comparison:

```text
dense-5k vs frozen current80
```

Interpretation guidance:

### Strong positive signal

Dense continuation is worth a longer follow-up only if the improvement is not a one-metric tradeoff. Prefer:

- at least 2/3 of Rel-L2, TKE, MVPE improve versus sparse R0;
- no primary metric has a material regression;
- horizon/trajectory evidence is broad rather than driven by a few cases.

### Weak / no signal

If gains are tiny, mixed, or mostly a TKE↔Rel tradeoff, stop. Do not start 20k/38.4k automatically.

Sol makes the final decision after Git evidence acceptance.

---

## 9. Environment autonomy

Scientific semantics are hard constraints; runtime environment is soft.

Codex may autonomously repair:

- Git tracking/detached checkout
- Python/venv/CUDA
- paths
- DataLoader workers
- launcher/PID/log paths
- output-root versioning
- whitelist archive
- file permissions
- non-semantic compatibility issues

Codex may NOT change:

- base/start checkpoint identities
- split/data manifest identities
- 5k budget
- stride=1 candidate semantics
- lr/batch/seed
- architecture
- feature set
- loss or loss weights
- optimizer/scheduler semantics
- locked-final/private/Codabench boundary

---

## 10. Preflight

Pull latest main and record actual execution HEAD.

Run:

```bash
python -m pytest -q \
  tests/test_dense_residual_continuation_screen.py \
  tests/test_colleague_incremental_screen.py \
  tests/test_post_train_diagnostics.py
```

Pure environment/integration failures may be repaired and committed.

Any required scientific-semantic change must BLOCK.

---

## 11. Execute

Reference command:

```bash
python -u -B tools/colleague_80pt/run_dense_residual_continuation_screen.py \
  --real-root /hy-tmp/realpde_data/train_real \
  --base-checkpoint /hy-tmp/realpde_assets/colleague-80pt/20260921/extracted/handoff_colleague_80pt_archive_20260921/checkpoints/cno_final_all81.pt \
  --start-checkpoint /hy-tmp/realpde_assets/colleague-80pt/20260921/extracted/handoff_colleague_80pt_archive_20260921/checkpoints/residual_model_best.pth \
  --model-root tools/colleague_80pt/submission \
  --data-manifest /hy-tmp/realpde_data/data_manifest.tsv \
  --split-manifest /hy-tmp/realpde_runs/colleague_incremental_screen_20260921/colleague_dev16_manifest.json \
  --out-root /hy-tmp/realpde_runs/dense_residual_continuation_20260923_v1 \
  --workers 4
```

If output root exists, use v2/v3. Do not delete prior runs.

Expected runtime is roughly one 5k residual continuation plus diagnostics, not a 20k or 38.4k campaign.

---

## 12. Required evidence

Must include:

### Provenance

- campaign manifest
- exact execution commit
- base/start checkpoint SHA
- split/data manifest SHA and copies
- GPU/runtime identity
- command log

### Training

- step 0
- step 1000
- step 2000
- step 3000
- step 4000
- step 5000
- learning rate
- loss components
- training review log
- best iteration

### Matched comparison

`dense_stride1_5k/comparison_at_5k.csv`

with:

- current80
- historical sparse5k
- dense5k
- percentage deltas versus current80
- dense percentage deltas versus sparse5k

### Diagnostics

For:

- current80
- dense_best
- dense_final

Must include:

- overall Rel/TKE/MVPE
- Future1..20
- by trajectory
- trajectory × horizon
- mean field
- fluctuation
- TKE energy ratio
- correction help/hurt
- spatial maps

---

## 13. Archive

After completion:

```bash
python -u -B tools/colleague_80pt/archive_dense_residual_continuation_screen.py \
  --run-root <ACTUAL_RUN_ROOT> \
  --dest docs/colleague_screening/results/20260923_dense_residual_continuation
```

If destination exists, append v2/v3. Do not overwrite.

Then:

```bash
git diff --check
git add docs/colleague_screening/results/20260923_dense_residual_continuation*
git commit -m "Archive dense residual continuation screen"
git pull --rebase origin main
git push origin main
```

Do not commit:

- checkpoints
- H5
- raw full training log
- prediction caches

---

## 14. Stop conditions

After archive + push, stop.

Do NOT:

- automatically extend training
- rerun with another stride
- tune LR
- tune loss
- add EMA
- add checkpoint averaging
- full refit
- package
- Codabench submit
- access locked-final/private

EMA / late-checkpoint averaging is deliberately deferred. First isolate whether dense continuation itself has value.

Final response format:

```text
REALPDE DENSE RESIDUAL CONTINUATION SCREEN

Status:
REVIEW_REQUIRED / BLOCKED

Execution commit:
...

Results commit:
...

Historical sparse control provenance:
PASS / FAIL

Base/start checkpoint SHA:
PASS / FAIL

Split/data manifest:
PASS / FAIL

Dense stride1 5k:
COMPLETE / BLOCKED

Training review log:
PASS / FAIL

Matched 5k comparison:
PASS / FAIL

Standard diagnostics:
PASS / FAIL

Best dense iteration:
...

Long follow-up started:
NO

Codabench accessed:
NO

Locked-final accessed:
NO

Missing items:
NONE / ...
```
