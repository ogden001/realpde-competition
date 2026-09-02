#!/usr/bin/env python3
"""Clean-room verifier for an offline P0-A Track 1 submission ZIP."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from build_p0a_n2_submission import REQUIRED_ITERATION, sha256, validate_checkpoint_for_submission


def prediction_error_summary(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    difference = np.abs(np.asarray(reference, dtype=np.float64) - np.asarray(candidate, dtype=np.float64))
    denominator = np.maximum(np.abs(np.asarray(reference, dtype=np.float64)), 1e-8)
    return {"max_abs_error": float(difference.max(initial=0.0)),
            "max_rel_error": float((difference / denominator).max(initial=0.0))}


def _load_module(path: Path):
    name = f"submission_cleanroom_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _direct_predict(checkpoint: Path, kit_root: Path, trainer: Path, inputs: np.ndarray) -> np.ndarray:
    sys.path.insert(0, str(trainer.parent))
    from realpde_p0a_n2_full import P0AConfig, _forward, _load_p0a_cno
    payload = validate_checkpoint_for_submission(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_p0a_cno(kit_root, checkpoint, device)
    model.eval()
    config = P0AConfig(**payload["feature_config"])
    with torch.inference_mode():
        output = _forward(model, config, torch.from_numpy(inputs).to(device))
        output[..., 2] = 0.0
    return output.cpu().numpy().astype(np.float32, copy=False)


def _run_package(zip_path: Path, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    with tempfile.TemporaryDirectory(prefix="p0a_cleanroom_") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(root)
        forbidden = [key for key in list(sys.modules) if key == "rpde_baselines" or key.startswith("rpde_baselines.")]
        for key in forbidden:
            sys.modules.pop(key, None)
        old_path = list(sys.path)
        try:
            sys.path[:] = [str(root)]
            module = _load_module(root / "submission.py")
            if torch.cuda.is_available():
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            first = module.predict(inputs, metadata=None)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            second = module.predict(inputs, metadata={})
            peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        finally:
            sys.path[:] = old_path
        checks = {"shape": list(first.shape), "dtype": str(first.dtype),
                  "finite": bool(np.isfinite(first).all()), "pressure_zero": bool(np.all(first[..., 2] == 0.0)),
                  "deterministic": bool(np.array_equal(first, second)), "elapsed_seconds": elapsed,
                  "peak_gpu_bytes": peak}
        return first, second, checks


def _official_kit_smoke(kit_root: Path, work_dir: Path) -> dict[str, object]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(kit_root)
    completed = subprocess.run([sys.executable, str(kit_root / "smoke_test_kit.py")], cwd=work_dir,
                               env=environment, text=True, capture_output=True, check=False)
    return {"command": [sys.executable, str(kit_root / "smoke_test_kit.py")], "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr}


def _manifest(report: dict[str, object], build: dict[str, object], status: str) -> str:
    inventory = "\n".join(f"- `{item['path']}` — {item['bytes']} bytes" for item in build["inventory"])
    lines = ["# Track 1 P0-A + N2 submission manifest", "", f"**Status:** `{status}`", "",
             "## Provenance", "", f"- Experiment ID: `{build['experiment_id']}`", f"- Git commit: `{build['git_commit']}`",
             f"- Checkpoint: `{build['checkpoint']}`", f"- Checkpoint SHA256: `{build['checkpoint_sha256']}`",
             f"- ZIP SHA256: `{build['zip_sha256']}`", f"- ZIP size: {build['zip_bytes']} bytes", "",
             "## ZIP inventory", "", inventory, "", "## Protocol", "",
             "`predict(input_array, metadata=None)` consumes and returns float32 `(N,20,32,64,3)` `[u,v,p]`; metadata is ignored, P0-A uses only input history, and pressure is zero.",
             "", "## Verification", "", "```json", json.dumps(report, indent=2, sort_keys=True), "```"]
    return "\n".join(lines) + "\n"


def verify(*, zip_path: Path, checkpoint: Path, kit_root: Path, trainer: Path, out_dir: Path) -> dict[str, object]:
    if out_dir.exists(): raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    payload = validate_checkpoint_for_submission(checkpoint, required_iteration=REQUIRED_ITERATION)
    if not zip_path.is_file(): raise FileNotFoundError(zip_path)
    rng = np.random.default_rng(20260902)
    reports: dict[str, object] = {"checkpoint_iteration": payload["iteration"], "zip_sha256": sha256(zip_path), "batches": {}}
    failures: list[str] = []
    for count in (1, 3):
        inputs = rng.standard_normal((count, 20, 32, 64, 3), dtype=np.float32)
        direct = _direct_predict(checkpoint, kit_root, trainer, inputs)
        packaged, _, checks = _run_package(zip_path, inputs)
        errors = prediction_error_summary(direct, packaged)
        reports["batches"][str(count)] = {"checks": checks, "direct_vs_package": errors}
        if (direct.shape != packaged.shape or packaged.dtype != np.float32 or not checks["finite"] or
                not checks["pressure_zero"] or not checks["deterministic"] or errors["max_abs_error"] != 0.0):
            failures.append(f"batch_{count}")
    reports["official_kit_smoke"] = _official_kit_smoke(kit_root, out_dir)
    if reports["official_kit_smoke"]["returncode"] != 0: failures.append("official_kit_smoke")
    reports["failures"] = failures
    reports["status"] = "READY_TO_SUBMIT" if not failures else "NOT_READY"
    (out_dir / "verification.json").write_text(json.dumps(reports, indent=2, sort_keys=True), encoding="utf-8")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = verify(zip_path=args.zip, checkpoint=args.checkpoint, kit_root=args.kit_root,
                    trainer=args.trainer, out_dir=args.out_dir)
    print(json.dumps(report, indent=2))
    if report["status"] != "READY_TO_SUBMIT": raise SystemExit(1)


if __name__ == "__main__": main()
