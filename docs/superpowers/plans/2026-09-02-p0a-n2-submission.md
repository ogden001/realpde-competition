# P0-A + N2 Track 1 Submission Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and offline-verify a manually uploadable Track 1 submission ZIP from the explicit 15,300-update P0-A + N2 checkpoint.

**Architecture:** A package builder materializes a minimal self-contained submission directory using a generated `submission.py`, selected model state and vendored CNO module. A separate verifier compares that package to the training implementation and executes isolated clean-room and official-kit checks, then writes a manifest from measured evidence.

**Tech Stack:** Python 3.10+, PyTorch 2.2.2 CUDA official-compatible container, NumPy, `zipfile`, `hashlib`, official Track 1 v9 starting kit.

**Spec:** `docs/superpowers/specs/2026-09-02-p0a-n2-submission-design.md`

## Global Constraints

- Only accept an explicit `--checkpoint`; reject a checkpoint whose saved iteration is not 15,300.
- Do not read locked-final, training data, HDF5, network resources, Codabench or metadata values.
- Submission inference accepts only `(N,20,32,64,3)` float arrays, ignores optional metadata, and returns finite float32 `[u,v,p]` outputs of the same shape.
- Use only input history for P0-A, `model.eval()` and `torch.inference_mode()`; pressure is zero.
- ZIP root must contain `submission.py`, the model state and necessary vendored CNO source only; total compressed size must be below 256 MiB.
- Artifacts must be written under an explicit `artifacts/` directory and never committed.
- Any validation failure produces `NOT_READY`; do not claim upload readiness without fresh complete evidence.

### Task 1: Inspect and encode the official submission contract

**Files:**
- Create: `tests/test_p0a_submission_package.py`
- Create: `tools/build_p0a_n2_submission.py`

**Interfaces:**
- Consumes: `kit_root/submission_template.py`, an explicit checkpoint path and output directory.
- Produces: `validate_checkpoint_for_submission(checkpoint: Path, required_iteration: int) -> dict` and `package_file_inventory(root: Path) -> list[dict]`.

- [ ] **Step 1: Read the remote official v9 template and smoke-test source without invoking Codabench**

Run inside the official-compatible container:

```bash
sed -n '1,260p' "$KIT_ROOT/submission_template.py"
sed -n '1,320p' "$KIT_ROOT/smoke_test_kit.py"
```

Record the precise callable/module convention in the builder's verification report; do not infer it from an older kit.

- [ ] **Step 2: Write failing checkpoint/inventory tests**

```python
def test_validate_checkpoint_rejects_non_15300_iteration(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"iteration": 1}, checkpoint)
    with pytest.raises(ValueError, match="15300"):
        validate_checkpoint_for_submission(checkpoint, required_iteration=15300)

def test_package_inventory_rejects_training_artifacts(tmp_path):
    (tmp_path / "optimizer.pth").write_bytes(b"x")
    with pytest.raises(ValueError, match="optimizer"):
        package_file_inventory(tmp_path)
```

- [ ] **Step 3: Run the tests and verify the intended RED failure**

Run:

```bash
pytest -q tests/test_p0a_submission_package.py -k 'checkpoint or inventory'
```

Expected: import failure because the builder module/functions do not exist.

- [ ] **Step 4: Implement explicit provenance and allowlisted package inventory**

Implement the two functions. Require `iteration == 15300`, P0-A feature set, expected N2 weights and a model state. Permit only `submission.py`, `model.pth`, vendored package `.py` files and package markers. Compute SHA256 and byte sizes with `Path` APIs.

- [ ] **Step 5: Run the focused tests and commit**

Run:

```bash
pytest -q tests/test_p0a_submission_package.py -k 'checkpoint or inventory'
```

Commit:

```bash
git add tests/test_p0a_submission_package.py tools/build_p0a_n2_submission.py
git commit -m "Add validated P0-A submission package builder"
```

### Task 2: Generate a self-contained submission implementation

**Files:**
- Modify: `tools/build_p0a_n2_submission.py`
- Modify: `tests/test_p0a_submission_package.py`

**Interfaces:**
- Consumes: explicit selected model state and vendored official `rpde_baselines/model/cno.py`.
- Produces: a staging directory whose root has `submission.py`, `model.pth` and `rpde_baselines/` source.

- [ ] **Step 1: Write failing submission-contract tests**

```python
def test_generated_submission_uses_file_relative_singleton_inference(tmp_path):
    generate_submission_module(tmp_path)
    source = (tmp_path / "submission.py").read_text()
    assert "Path(__file__).resolve().parent" in source
    assert "torch.inference_mode" in source
    assert "model.eval()" in source
    assert "metadata=None" in source
    assert "h5py" not in source
    assert "http" not in source

def test_submission_input_contract_rejects_bad_shape_and_preserves_pressure_zero(package_module):
    with pytest.raises(ValueError):
        package_module.predict(np.zeros((1, 20, 32, 64, 2), np.float32))
    result = package_module.predict(np.zeros((1, 20, 32, 64, 3), np.float32))
    assert result.dtype == np.float32
    assert np.all(result[..., 2] == 0)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
pytest -q tests/test_p0a_submission_package.py -k 'submission'
```

Expected: generated module API is absent or fails the static contract checks.

- [ ] **Step 3: Implement the generated module without changing P0-A computation**

Generate a `submission.py` that contains the exact pure-torch P0-A feature math
from `tools/realpde_p0a_n2_full.py`, imports only `numpy`, `torch`, `Path` and
the vendored CNO source, caches the loaded model globally, validates input,
uses `eval()` and inference mode, and zeroes pressure after the forward pass.

- [ ] **Step 4: Run module tests and commit**

Run:

```bash
pytest -q tests/test_p0a_submission_package.py -k 'submission'
```

Commit:

```bash
git add tests/test_p0a_submission_package.py tools/build_p0a_n2_submission.py
git commit -m "Generate self-contained P0-A submission module"
```

### Task 3: Implement independent consistency and clean-room verification

**Files:**
- Create: `tools/realpde_p0a_submission_smoke.py`
- Modify: `tests/test_p0a_submission_package.py`

**Interfaces:**
- Consumes: explicit checkpoint, staged ZIP, kit root and an explicit output directory.
- Produces: JSON verification report containing direct/package max absolute and relative errors, N=1/N>1 checks, deterministic replay, timing, GPU peak allocated memory and official-kit result.

- [ ] **Step 1: Write failing verifier-report tests**

```python
def test_error_summary_is_zero_for_equal_predictions():
    result = prediction_error_summary(np.ones((1, 2), np.float32), np.ones((1, 2), np.float32))
    assert result == {"max_abs_error": 0.0, "max_rel_error": 0.0}

def test_error_summary_uses_safe_relative_denominator():
    result = prediction_error_summary(np.array([0.0], np.float32), np.array([1.0], np.float32))
    assert result["max_rel_error"] > 0.0
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest -q tests/test_p0a_submission_package.py -k 'error_summary'
```

Expected: import failure because the smoke verifier does not exist.

- [ ] **Step 3: Implement isolated verifier**

Build deterministic synthetic N=1 and N=3 arrays; load the selected checkpoint
with the training implementation; extract the ZIP using `tempfile.TemporaryDirectory`;
clear module cache; import only extracted `submission.py`; run direct/package
comparison, contract checks, repeated-call check, timing and `torch.cuda`
peak-memory reset/measurement. Call the official kit smoke script only after
inspection confirms its local interface applies. Store command, return code,
stdout and stderr in the report.

- [ ] **Step 4: Run focused tests and commit**

Run:

```bash
pytest -q tests/test_p0a_submission_package.py -k 'error_summary'
```

Commit:

```bash
git add tests/test_p0a_submission_package.py tools/realpde_p0a_submission_smoke.py
git commit -m "Add clean-room P0-A submission verifier"
```

### Task 4: Build and verify the explicit 15,300 package

**Files:**
- Generate outside Git: `artifacts/p0a_n2_15300_submission_20260902/submission.zip`
- Generate outside Git: `artifacts/p0a_n2_15300_submission_20260902/submission.zip.sha256`
- Generate outside Git: `artifacts/p0a_n2_15300_submission_20260902/SUBMISSION_MANIFEST.md`
- Generate outside Git: `artifacts/p0a_n2_15300_submission_20260902/verification.json`

**Interfaces:**
- Consumes: the verified 15,300 checkpoint, official v9 kit and generated builder/verifier tools.
- Produces: one offline ZIP and auditable evidence without Codabench access.

- [ ] **Step 1: Transfer source only and validate provenance in the official-compatible container**

Use an explicit remote artifact output directory. Run the builder with the 15,300
checkpoint path, Git commit and official kit root. Require selected iteration 15,300
and record the checkpoint SHA256 before copying the model state into the staging
directory.

- [ ] **Step 2: Build ZIP and enforce size/inventory gates**

Run the builder. Require ZIP root `submission.py`, model state and vendored CNO
only; record uncompressed and compressed byte counts; fail at or above 256 MiB.

- [ ] **Step 3: Execute full clean-room and official-kit verification**

Run the smoke tool in the official-compatible container. Require zero direct/package
prediction difference, valid N=1/N=3 float32 finite outputs, zero pressure,
deterministic replay, recorded timing/GPU peak, successful re-extraction and a
successful applicable official-kit test.

- [ ] **Step 4: Render manifest from measured results and determine status**

Write the requested Markdown fields from the real JSON results. Set
`READY_TO_SUBMIT` only if every gate in this plan passes; otherwise write
`NOT_READY` with the failing command/output and do not describe the ZIP as ready.

- [ ] **Step 5: Final fresh verification and report**

Run:

```bash
python tools/realpde_p0a_submission_smoke.py --zip "$ZIP" --checkpoint "$CHECKPOINT" --kit-root "$KIT_ROOT" --out-dir "$ARTIFACT_DIR/recheck"
sha256sum "$ZIP"
unzip -l "$ZIP"
```

Compare the new report to the manifest before claiming status. Do not upload,
open Codabench, or update `docs/submission_log.md` because no official submission
has occurred.
