#!/usr/bin/env python3
"""Clean-room parity/safety smoke for a SOTA-V3 submission ZIP."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np
import torch

import realpde_residual_corrector_projection as projection
import realpde_sota_v2_integrated as base
import realpde_sota_v3_residual as residual
import realpde_teammate_residual_transfer as transfer
from build_sota_v3_package import sha256


def _load_input(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        u = np.asarray(handle["u"][:20], dtype=np.float32)
        v = np.asarray(handle["v"][:20], dtype=np.float32)
    if u.shape == (20, 64, 128):
        u, v = u[:, ::2, ::2], v[:, ::2, ::2]
    if u.shape != (20, 32, 64) or v.shape != u.shape:
        raise ValueError(f"fixture must resolve to (20,32,64), got {u.shape}")
    p = np.zeros_like(u)
    return np.stack([u, v, p], axis=-1)[None]


@torch.no_grad()
def _reference(backbone_checkpoint: Path, corrector_checkpoint: Path,
               kit_root: Path, x: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, builder, backbone = residual._load_backbone(backbone_checkpoint, kit_root, device)
    corrector, _ = residual._load_corrector(
        corrector_checkpoint, device, base.sha256(backbone_checkpoint)
    )
    tensor = torch.from_numpy(np.ascontiguousarray(x)).to(device)
    base_prediction = base.forward_mf(backbone, builder, tensor)
    delta = transfer._delta_from_corrector(corrector, tensor, base_prediction)
    corrected = transfer.apply_scaled_correction(base_prediction, delta, 1.0)
    final = projection.spatial_tke_map_projection(base_prediction, corrected)
    return final.cpu().numpy().astype(np.float32)


def _run_clean_room(root: Path, x: np.ndarray):
    np.save(root / "smoke_input.npy", x)
    script = r'''
import json, time, numpy as np, torch
import submission
x = np.load("smoke_input.npy")
if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
t0=time.perf_counter(); a=submission.predict(x); first=time.perf_counter()-t0
t1=time.perf_counter(); b=submission.predict(x); second=time.perf_counter()-t1
for key in ("prediction","lower","upper"):
    if not np.array_equal(a[key], b[key]): raise RuntimeError("nondeterministic:"+key)
np.savez("smoke_output.npz", **a)
print(json.dumps({"first_call_seconds":first,"second_call_seconds":second,
"peak_cuda_memory_allocated":int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0}))
'''
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd=root, text=True,
        capture_output=True, check=True
    )
    timing = json.loads(proc.stdout.strip().splitlines()[-1])
    packed = np.load(root / "smoke_output.npz")
    result = {
        k: packed[k].astype(np.float32, copy=False)
        for k in ("prediction", "lower", "upper")
    }
    return result, timing


def validate(reference: np.ndarray, result: dict[str, np.ndarray],
             tolerance: float) -> dict[str, float]:
    for key in ("prediction", "lower", "upper"):
        value = result[key]
        if value.shape != reference.shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"invalid package output: {key}")
    pred, lower, upper = result["prediction"], result["lower"], result["upper"]
    diff = float(np.max(np.abs(pred - reference)))
    if diff > tolerance:
        raise ValueError(f"V3 point parity failed: {diff} > {tolerance}")
    if np.any(lower > pred) or np.any(pred > upper):
        raise ValueError("prediction not inside interval")
    if float(np.max(np.abs(pred[..., 2]))) != 0.0:
        raise ValueError("pressure prediction must be zero")
    if float(np.max(np.abs(upper[..., 2] - lower[..., 2]))) != 0.0:
        raise ValueError("pressure interval width must be zero")
    return {
        "max_abs_prediction_diff": diff,
        "mean_width_uv": float(np.mean(upper[..., :2] - lower[..., :2])),
    }


def run(args: argparse.Namespace) -> dict:
    x = _load_input(args.fixture)
    reference = _reference(
        args.backbone_checkpoint, args.corrector_checkpoint, args.kit_root, x
    )
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="sota_v3_smoke_") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(args.zip, "r") as archive:
            archive.extractall(root)
        result, timing = _run_clean_room(root, x)
    checks = validate(reference, result, args.tolerance)
    report = {
        "status": "PASS_REVIEW_REQUIRED",
        **checks,
        **timing,
        "zip": str(args.zip),
        "zip_bytes": args.zip.stat().st_size,
        "zip_sha256": sha256(args.zip),
        "fixture": str(args.fixture),
        "backbone_sha256": base.sha256(args.backbone_checkpoint),
        "corrector_sha256": base.sha256(args.corrector_checkpoint),
        "total_seconds": time.perf_counter() - started,
        "locked_final_accessed": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=Path, required=True)
    p.add_argument("--backbone-checkpoint", type=Path, required=True)
    p.add_argument("--corrector-checkpoint", type=Path, required=True)
    p.add_argument("--kit-root", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tolerance", type=float, default=1e-6)
    run(p.parse_args())


if __name__ == "__main__":
    main()
