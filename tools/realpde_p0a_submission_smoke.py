#!/usr/bin/env python3
"""Minimal clean-room smoke for a packaged P0-A Track 1 submission."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from build_p0a_n2_submission import sha256


def check_prediction_bundle(
    bundle: object, *, bound_abs: float, bound_rel: float
) -> dict[str, object]:
    if not isinstance(bundle, dict) or not all(key in bundle for key in ("prediction", "lower", "upper")):
        raise ValueError("explicit-bounds submission must return prediction/lower/upper")
    prediction = np.asarray(bundle["prediction"])
    lower = np.asarray(bundle["lower"])
    upper = np.asarray(bundle["upper"])
    if prediction.shape != lower.shape or prediction.shape != upper.shape:
        raise ValueError("prediction/lower/upper shapes differ")
    if prediction.dtype != np.float32 or lower.dtype != np.float32 or upper.dtype != np.float32:
        raise ValueError("prediction/lower/upper must be float32")
    if not (np.isfinite(prediction).all() and np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise ValueError("prediction bundle contains non-finite values")
    expected_half = np.float32(bound_abs) + np.float32(bound_rel) * np.abs(prediction)
    if not (
        np.allclose(lower, prediction - expected_half, rtol=0.0, atol=1e-6)
        and np.allclose(upper, prediction + expected_half, rtol=0.0, atol=1e-6)
    ):
        raise ValueError("bounds do not match abs + rel*|prediction|")
    if np.any(lower > upper):
        raise ValueError("lower exceeds upper")
    return {
        "shape": list(prediction.shape),
        "dtype": str(prediction.dtype),
        "finite": True,
        "pressure_zero": bool(np.all(prediction[..., 2] == 0.0)),
        "bounds_match": True,
    }


def _load_module(path: Path):
    module_name = f"realpde_submission_smoke_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def verify_zip(
    *, zip_path: Path, required_iteration: int, bound_abs: float, bound_rel: float, out: Path
) -> dict[str, object]:
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="realpde_p0a_smoke_") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(root)
        if not (root / "submission.py").is_file() or not (root / "model.pth").is_file():
            raise ValueError("submission ZIP must contain root-level submission.py and model.pth")
        payload = torch.load(root / "model.pth", map_location="cpu", weights_only=False)
        if payload.get("iteration") != required_iteration:
            raise ValueError(
                f"packaged checkpoint iteration {payload.get('iteration')!r} != {required_iteration}"
            )
        if payload.get("feature_set") != "P0-A":
            raise ValueError("packaged checkpoint is not P0-A")

        old_path = list(sys.path)
        try:
            sys.path.insert(0, str(root))
            module = _load_module(root / "submission.py")
            rng = np.random.default_rng(20260904)
            inputs = rng.standard_normal((1, 20, 32, 64, 3), dtype=np.float32)
            first = module.predict(inputs, metadata=None)
            second = module.predict(inputs, metadata={})
        finally:
            sys.path[:] = old_path

        checks = check_prediction_bundle(first, bound_abs=bound_abs, bound_rel=bound_rel)
        second_checks = check_prediction_bundle(second, bound_abs=bound_abs, bound_rel=bound_rel)
        deterministic = all(
            np.array_equal(np.asarray(first[key]), np.asarray(second[key]))
            for key in ("prediction", "lower", "upper")
        )
        if not deterministic:
            raise ValueError("packaged inference is not deterministic")
        if not checks["pressure_zero"]:
            raise ValueError("prediction pressure channel is not zero")

        result = {
            "status": "READY_FOR_SUBMISSION_REVIEW",
            "zip": str(zip_path.resolve()),
            "zip_sha256": sha256(zip_path),
            "zip_bytes": zip_path.stat().st_size,
            "checkpoint_iteration": int(payload["iteration"]),
            "bounds": {"abs": float(bound_abs), "rel": float(bound_rel)},
            "checks": checks,
            "second_checks": second_checks,
            "deterministic": deterministic,
        }
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--required-iteration", type=int, required=True)
    parser.add_argument("--bound-abs", type=float, required=True)
    parser.add_argument("--bound-rel", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_zip(
                zip_path=args.zip,
                required_iteration=args.required_iteration,
                bound_abs=args.bound_abs,
                bound_rel=args.bound_rel,
                out=args.out,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
