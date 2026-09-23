# Colleague 65/16 Baseline Provenance Audit

Status: `GITHUB_EVIDENCE_COMPLETE / GPU_PROVENANCE_REQUIRED`

Date: 2026-09-24

## 1. Purpose

This document separates the colleague's **offline R&D baseline** from the later
**all81 full-train submission baseline**.

The main question is:

> What was the colleague's ordinary 65-train / 16-validation baseline used to
> screen modeling ideas before full-data retraining?

This document records only evidence recoverable from Git/GitHub.  Anything not
directly established is marked UNKNOWN and must be checked on the internal GPU
server before it is treated as a frozen baseline.

---

## 2. High-level conclusion

GitHub evidence is sufficient to establish that the colleague did have a
trajectory-disjoint **65 train / 16 validation** offline development protocol.

The strongest clearly documented early offline candidate is:

```text
E3 CNO
+ ResidualCorrector3D
  hidden=64
  blocks=2
  max_delta=0.04
  alpha=1.0
  updates=2400
```

On the fixed 65/16 HDF5 validation protocol its documented metrics were:

```text
Rel-L2 = 0.091532
TKE    = 0.490739
MVPE   = 0.082907
SPS    = 45.055720
local proxy final = 80.203498
```

The corresponding package name was:

`submission_cnoE3_residualcorr_h64_b2_u2400_alpha100_abs0075_rel0075_20260901.zip`

The later handoff history identifies the online **9-1 champion** as:

`CNO E3 + residual h64/b2/u2400`

with online final:

`78.077620`

This makes the E3+h64/u2400 line the strongest currently documented candidate
for the colleague's early offline R&D champion.

However, GitHub alone does **not** yet prove the complete training ancestry of
the E3 CNO checkpoint itself.  Therefore this candidate must not yet be called
an end-to-end clean baseline.

---

## 3. Confirmed data split

Source code:

`tools/realpde_h5_feature_adapter_train.py`

uses:

```python
paths = list_h5(real_root, BAD_TRAIN_FILES)
train_paths, val_paths = split_paths(paths, val_fraction=0.2, seed=41)
```

with:

`BAD_TRAIN_FILES = {"7575_0.h5"}`

The colleague submission log explicitly records:

```text
usable h5 files: 81 after excluding 7575_0.h5
train trajectories: 65
validation trajectories: 16
train windows: 2701
validation windows: 640
```

Window protocol:

```text
Past20 -> Future20
stride = 20
spatial subsample = 2
seed = 41
validation fraction = 0.2
```

The split is trajectory-level, not window-level.

Therefore the residual-stage 65/16 development split itself is confirmed.

---

## 4. Critical Git-history evidence: all-data training was added later

At commit:

`a9283550a075a27233b0f751e554276910ea8059`

dated 2026-09-01 16:27, message:

`Make residual context optional and support all-data fit`

the residual trainer was modified to add:

```python
parser.add_argument("--train-on-all", action="store_true")
...
fit_paths = paths if args.train_on_all else train_paths
```

Before this commit, the residual trainer used:

```python
train_dataset = H5WindowDataset(train_paths, ...)
```

with no all-data option.

This is strong provenance evidence that residual experiments produced before
that all-data modification used the 65-train / 16-validation split by default.

The documented E3 residual h32/h48/h64 experiments were recorded earlier on
2026-09-01, including:

- commit `ac41dede...`: E3 residual u2400
- commit `4e1ac6a8...`: E3 residual h48
- commit `f5b2e924...`: E3 residual h64

Thus the documented E3 residual h64/b2/u2400 result belongs to the 65/16
residual-development regime.

---

## 5. Documented 65/16 model progression

From the colleague's `docs/submission_log.md`:

| Candidate | Rel-L2 | TKE | MVPE | SPS | Local proxy |
|---|---:|---:|---:|---:|---:|
| E3 CNO only | 0.101558 | 0.593833 | 0.092624 | 41.364861 | 77.824316 |
| E3 CNO + residual h32/u1200 | 0.095492 | 0.523265 | 0.086761 | 43.464214 | 79.657262 |
| E3 CNO + residual h32/u1600 | 0.094830 | 0.516260 | 0.086174 | 43.725785 | 79.758424 |
| E3 CNO + residual h32/u2400 | 0.093833 | 0.507095 | 0.085441 | 44.122848 | 79.914030 |
| E3 CNO + residual h48/b2/u2400 | 0.092650 | 0.494225 | 0.083971 | 44.597158 | 80.079433 |
| **E3 CNO + residual h64/b2/u2400** | **0.091532** | **0.490739** | **0.082907** | **45.055720** | **80.203498** |

The progression is coherent:

```text
E3 CNO
  -> residual h32, longer training
  -> residual h48
  -> residual h64
```

This is characteristic of an offline development sequence, not a sequence of
blind full-data leaderboard submissions.

---

## 6. Online correspondence

The later colleague handoff records:

```text
9-1 champion
CNO E3 + residual h64/b2/u2400
online final = 78.077620
```

This configuration matches the strongest documented 65/16 offline candidate in
Section 5.

This establishes a strong configuration-level correspondence between the
offline R&D champion and the early online submission.

It does NOT by itself prove checkpoint identity.

---

## 7. Later transition to all81 full training

The colleague handoff later records the full-data path:

```text
official / existing CNO
  -> all81 CNO fine-tune, stride=1
  -> all81 residual corrector
  -> all81 uncertainty head
  -> 80.078849 online
```

The final Stage-1 CNO:

`cno_final_all81.pt`

SHA256:

`ff28aaf0114d57e320e5ea9cf9945d874a7482f22e3f0aeaed5ebbe286573f8a`

The final Stage-2 residual:

`residual_h96x8_all81_20260920/model_best.pth`

SHA256:

`909fdc7f8a6a42335e4ea4ce7471fc9a507135e9a29a4927bc7b7318be9c85b2`

This is a **full-data submission baseline**, not the clean offline R&D baseline.

It must not be used as the reference for claims about held-out generalization.

---

## 8. Important split-champion checkpoint clue

The handoff's frozen-cache code uses the default:

```text
/runs/residual_split_20260916/model_best.pth
```

and later all81 cache metadata records:

```json
{
  "champion": "/runs/residual_split_20260916/model_best.pth",
  "champion_used": "/runs/residual_h96x8_all81_20260920/model_best.pth"
}
```

This proves that a later **split champion checkpoint** existed on the internal
server and remained an explicit reference point even after the final all81 model
was adopted.

What GitHub does NOT currently establish:

- its SHA256;
- exact creation command;
- exact corrector width/depth;
- update count;
- exact E3/base checkpoint embedded in it;
- exact validation metrics;
- whether it is identical to the 2026-09-01 h64/u2400 champion or a later
  improved 65/16 checkpoint.

This checkpoint is the highest-priority object for the internal-GPU audit.

---

## 9. What is already safe to call the colleague's offline protocol

The following are directly supported:

```text
Usable real trajectories : 81
Excluded file            : 7575_0.h5
Split seed               : 41
Offline training         : 65 trajectories
Offline validation       : 16 trajectories
Train windows            : 2701 at stride20
Validation windows       : 640 at stride20
Past / Future            : 20 / 20
Spatial subsample        : 2
Residual architecture    : 3D convolution corrector family
Early champion config    : h64 / b2 / u2400 / max_delta0.04 / alpha1
Early champion metrics   : 0.091532 / 0.490739 / 0.082907
Early online counterpart : 78.077620
```

---

## 10. What is NOT yet safe to call proven

### A. E3 CNO training ancestry

GitHub records the E3 CNO's validation metrics and its role as the frozen base,
but the currently recovered Git material does not provide a complete training
manifest proving:

- which starting checkpoint E3 used;
- whether that starting checkpoint had already seen real Dev16 trajectories;
- exactly which real trajectories E3 itself was trained on;
- E3 checkpoint SHA256;
- E3 training step count / learning rate / complete loss weights.

This is the most important remaining provenance gap.

If E3 or one of its ancestors had already trained on Dev16 real PIV data, then
the residual stage is 65/16-disjoint but the overall pipeline is not
end-to-end clean.

### B. Identity of residual_split_20260916

The path is confirmed, but checkpoint identity/config/metrics are not.

### C. Exact checkpoint file for the 2026-09-01 h64/u2400 result

The package name and metrics are recorded, but Git ignores large checkpoint
files.  SHA and server path must be recovered from the internal machine.

### D. Whether there was a stronger 65/16 champion after 2026-09-01

The existence of `residual_split_20260916` strongly suggests the split line
continued.  GitHub handoff does not fully document that evolution.

---

## 11. Baseline hierarchy to use until GPU audit finishes

### Candidate offline R&D baseline

Do not yet mark FROZEN.

```text
Candidate:
E3 CNO + residual h64/b2/u2400
split: seed41 65 train / 16 validation
metrics: Rel 0.091532 / TKE 0.490739 / MVPE 0.082907
status: GITHUB-CONFIRMED CONFIG + METRICS
        CHECKPOINT/ANCESTRY NOT YET CONFIRMED
```

### Candidate later split champion

```text
/runs/residual_split_20260916/model_best.pth

status:
EXISTENCE CONFIRMED
CONFIG / SHA / METRICS / ANCESTRY UNKNOWN
```

This object may supersede the 2026-09-01 h64/u2400 candidate after the GPU
audit.

### Full-data submission baseline

```text
80.078849 all81 solution
status: FROZEN TECHNICAL SUBMISSION BASELINE
not valid as a held-out offline research baseline
```

---

## 12. Internal GPU audit targets

The next audit should search, without starting training, for:

1. `/runs/residual_split_20260916/model_best.pth`
2. adjacent:
   - `run_config.json`
   - `summary.json`
   - logs
   - launch shell/history
   - eval JSON
3. all checkpoints/packages matching:
   - `*cnoE3*`
   - `*E3*`
   - `*residual*h64*2400*`
   - `*residual_split*`
4. checkpoint-embedded metadata:
   - run_config
   - corrector_config
   - iteration / best_iteration
   - base-model state keys
5. SHA256 for every plausible baseline checkpoint.
6. shell history / nohup logs / archived command logs around:
   - 2026-09-01
   - 2026-09-16
7. E3 CNO ancestry:
   - starting checkpoint
   - training data membership
   - split seed
   - update count
   - LR
   - loss weights
   - checkpoint SHA
8. exact 65/16 filenames and a manifest hash reconstructed from the server data.
9. direct reevaluation, only if the checkpoint exists and evaluation is
   non-training:
   - 640 Dev windows
   - Rel-L2
   - TKE
   - MVPE
   - optional SPS/proxy for historical parity

The audit must not start any training, full refit, packaging, Codabench access,
or locked-final/private access.

---

## 13. Decision rule after GPU audit

A checkpoint can become the frozen offline baseline only when all of these are
known:

```text
checkpoint identity + SHA
base-model ancestry
65 training trajectory membership
16 validation trajectory membership
no real-data leakage into the validation trajectories for the claimed scope
model config
training budget
loss
offline metrics reproduced
```

If the strongest later `residual_split_20260916` satisfies these conditions,
use it.

If not, fall back to the fully reconstructable 2026-09-01 E3+h64/u2400
configuration, after its E3 ancestry is audited.

Until then, label the offline baseline:

`PROVISIONAL / PROVENANCE_INCOMPLETE`
