#!/usr/bin/env python3
"""Clean-room parity/safety smoke for a SOTA-V2 adaptive submission ZIP."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np
import torch

from build_sota_v2_adaptive_package import (
    EXPECTED_FULL_CHECKPOINT_SHA,
    sha256,
    strip_mf_prefix,
    validate_full_checkpoint_metadata,
)
from sota_v2_adaptive_runtime import mf_forward


def validate_package_outputs(reference_prediction: np.ndarray, result: dict[str, np.ndarray], *, tolerance: float) -> dict[str, float]:
    for name in ("prediction", "lower", "upper"):
        if name not in result:
            raise ValueError(f"package output missing {name}")
        if result[name].shape != reference_prediction.shape or result[name].dtype != np.float32 or not np.isfinite(result[name]).all():
            raise ValueError(f"invalid package {name}")
    pred, lower, upper = result["prediction"], result["lower"], result["upper"]
    max_diff = float(np.max(np.abs(pred - reference_prediction)))
    if max_diff > tolerance:
        raise ValueError(f"prediction parity failed: {max_diff} > {tolerance}")
    if float(np.max(np.abs(pred[..., 2]))) != 0.0:
        raise ValueError("prediction pressure must be zero")
    if float(np.max(np.abs(upper[..., 2] - lower[..., 2]))) != 0.0:
        raise ValueError("pressure interval width must be zero")
    if np.any(lower > pred) or np.any(pred > upper):
        raise ValueError("prediction must lie inside interval")
    return {
        "max_abs_prediction_diff": max_diff,
        "mean_width_uv": float(np.mean(upper[..., :2] - lower[..., :2])),
    }


def _load_input(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        u = np.asarray(handle["u"][:20], dtype=np.float32)
        v = np.asarray(handle["v"][:20], dtype=np.float32)
    if u.shape != (20, 32, 64) or v.shape != u.shape:
        raise ValueError("fixture must already be 20+ frames at 32x64")
    p = np.zeros_like(u)
    return np.stack([u, v, p], axis=-1)[None]


def _reference_prediction(full_checkpoint: Path, kit_root: Path, x: np.ndarray) -> np.ndarray:
    from realpde_p0_features import P0FeatureBuilder, P0FeatureConfig
    sys.path.insert(0, str(kit_root.resolve()))
    from rpde_baselines.model.cno import CNO3d

    payload = torch.load(full_checkpoint, map_location="cpu", weights_only=False)
    actual_sha = sha256(full_checkpoint)
    validate_full_checkpoint_metadata(payload, actual_sha, EXPECTED_FULL_CHECKPOINT_SHA)
    raw = payload["feature_config"]
    config = P0FeatureConfig(
        include_p0_a=bool(raw.get("include_p0_a", True)),
        include_p0_b=bool(raw.get("include_p0_b", False)),
        dx=float(raw["dx"]), dy=float(raw["dy"]),
        dt=None if raw.get("dt") is None else float(raw["dt"]),
        re_center=float(raw.get("re_center", 0.0)), re_scale=float(raw.get("re_scale", 1.0)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    builder = P0FeatureBuilder(config).to(device)
    cno = CNO3d(in_dim=len(builder.feature_names), out_dim=5, out_dim_mult=1, in_size=64, N_layers=3).to(device)
    cno.load_state_dict(strip_mf_prefix(payload["model_state_dict"]), strict=True)
    cno.eval()
    with torch.inference_mode():
        pred = mf_forward(cno, builder, torch.from_numpy(x).to(device)).cpu().numpy().astype(np.float32)
    return pred


def run(args: argparse.Namespace) -> dict[str, object]:
    x = _load_input(args.fixture)
    reference = _reference_prediction(args.full_checkpoint, args.kit_root, x)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="sota_v2_adaptive_smoke_") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(args.zip, "r") as archive:
            archive.extractall(root)
        spec = importlib.util.spec_from_file_location("sota_v2_submission_smoke", root / "submission.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot import submission.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        t0 = time.perf_counter(); first = module.predict(x); first_s = time.perf_counter() - t0
        t1 = time.perf_counter(); second = module.predict(x); second_s = time.perf_counter() - t1
    parity = validate_package_outputs(reference, first, tolerance=args.tolerance)
    deterministic = max(float(np.max(np.abs(first[k] - second[k]))) for k in ("prediction", "lower", "upper"))
    if deterministic != 0.0:
        raise ValueError(f"package is not deterministic: {deterministic}")
    report = {
        "status": "PASS",
        **parity,
        "deterministic_max_abs_diff": deterministic,
        "first_call_seconds": first_s,
        "second_call_seconds": second_s,
        "zip": str(args.zip),
        "zip_bytes": args.zip.stat().st_size,
        "zip_sha256": sha256(args.zip),
        "fixture": str(args.fixture),
        "full_checkpoint_sha256": sha256(args.full_checkpoint),
        "total_seconds": time.perf_counter() - started,
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=Path, required=True)
    p.add_argument("--full-checkpoint", type=Path, required=True)
    p.add_argument("--kit-root", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tolerance", type=float, default=1e-6)
    run(p.parse_args())


if __name__ == "__main__":
    main()
