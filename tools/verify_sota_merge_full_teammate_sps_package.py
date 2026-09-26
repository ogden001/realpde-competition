#!/usr/bin/env python3
"""Clean-room parity verifier for Full Joint@6500 + exact teammate SPS package."""
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

from build_sota_merge_full_teammate_sps_package import (
    EXPECTED_BACKBONE_SHA256,
    EXPECTED_CORRECTOR_SHA256,
    sha256,
    validate_assets,
)
from realpde_adaptive_probe import ResidualCorrector3D, adaptive_features
from realpde_p0_features import P0FeatureBuilder, P0FeatureConfig
from sota_v2_adaptive_runtime import mf_forward
from sps_teammate_uncertainty_runtime import (
    TeammateUncertaintyHead,
    sigma_from_log_std,
)

STATUS = "REVIEW_REQUIRED"
_EPS = 1e-12


def _spatial_tke_map_projection(
    base: torch.Tensor,
    corrected: torch.Tensor,
) -> torch.Tensor:
    b_uv, c_uv = base[..., :2], corrected[..., :2]
    b_fluct = b_uv - b_uv.mean(dim=1, keepdim=True)
    c_mean = c_uv.mean(dim=1, keepdim=True)
    c_fluct = c_uv - c_mean
    numerator = b_fluct.double().square().mean(dim=1).sum(dim=-1)
    denominator = c_fluct.double().square().mean(dim=1).sum(dim=-1)
    scale = torch.where(
        denominator > _EPS,
        torch.sqrt(
            numerator.clamp_min(0.0) / denominator.clamp_min(_EPS)
        ),
        torch.ones_like(denominator),
    ).to(dtype=corrected.dtype)[:, None, :, :, None]
    out = corrected.clone()
    out[..., :2] = c_mean + scale * c_fluct
    out[..., 2] = 0.0
    return out


def _load_input(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        u = np.asarray(handle["u"][:20], dtype=np.float32)
        v = np.asarray(handle["v"][:20], dtype=np.float32)
    if u.shape == (20, 64, 128):
        u = u[:, ::2, ::2]
        v = v[:, ::2, ::2]
    if u.shape != (20, 32, 64) or v.shape != u.shape:
        raise ValueError(
            f"fixture must resolve to (20,32,64), got u={u.shape}, v={v.shape}"
        )
    p = np.zeros_like(u)
    return np.stack([u, v, p], axis=-1)[None]


@torch.no_grad()
def _reference(
    *,
    backbone_checkpoint: Path,
    corrector_checkpoint: Path,
    head_checkpoint: Path,
    kit_root: Path,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assets = validate_assets(
        backbone_checkpoint,
        corrector_checkpoint,
        head_checkpoint,
    )

    sys.path.insert(0, str(kit_root.resolve()))
    from rpde_baselines.model.cno import CNO3d

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = P0FeatureConfig(**assets["feature_config"])
    builder = P0FeatureBuilder(cfg).to(device)

    cno = CNO3d(
        in_dim=len(builder.feature_names),
        out_dim=5,
        out_dim_mult=1,
        in_size=64,
        N_layers=3,
    ).to(device)
    cno.load_state_dict(assets["cno_state_dict"], strict=True)

    corrector = ResidualCorrector3D(
        in_channels=42,
        hidden=64,
        blocks=2,
        max_delta=0.04,
    ).to(device)
    corrector.load_state_dict(assets["corrector_state_dict"], strict=True)

    head = TeammateUncertaintyHead(
        hidden=32,
        blocks=2,
        dropout=0.0,
        include_pressure=True,
    ).to(device)
    head.load_state_dict(assets["head_state_dict"], strict=True)

    cno.eval()
    corrector.eval()
    head.eval()

    tensor = torch.from_numpy(np.ascontiguousarray(x)).to(device)
    base = mf_forward(cno, builder, tensor)

    features = adaptive_features(
        tensor,
        base,
    ).permute(0, 4, 1, 2, 3).contiguous()
    delta = corrector(features).permute(0, 2, 3, 4, 1).contiguous()
    corrected = base + delta
    corrected[..., 2] = 0.0
    prediction = _spatial_tke_map_projection(base, corrected)

    sigma = sigma_from_log_std(head(tensor, base))
    calibration = assets["calibration"]
    half_uv = (
        float(calibration["floor"])
        + float(calibration["mult"]) * sigma
    )
    half = torch.cat(
        [half_uv, torch.zeros_like(half_uv[..., :1])],
        dim=-1,
    )
    lower = prediction - half
    upper = prediction + half

    return tuple(
        value.cpu().numpy().astype(np.float32)
        for value in (prediction, lower, upper)
    )


def _run_clean_room(
    root: Path,
    x: np.ndarray,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, object],
]:
    np.save(root / "smoke_input.npy", x)
    script = r"""
import json
import time
import numpy as np
import torch
import submission

x = np.load("smoke_input.npy")
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

t0 = time.perf_counter()
a = submission.predict(x)
first = time.perf_counter() - t0

t1 = time.perf_counter()
b = submission.predict(x)
second = time.perf_counter() - t1

peak = (
    int(torch.cuda.max_memory_allocated())
    if torch.cuda.is_available()
    else 0
)
np.savez(
    "smoke_output.npz",
    prediction=a["prediction"],
    lower=a["lower"],
    upper=a["upper"],
    prediction2=b["prediction"],
    lower2=b["lower"],
    upper2=b["upper"],
)
print(json.dumps({
    "first_call_seconds": first,
    "second_call_seconds": second,
    "peak_cuda_memory_allocated": peak,
}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    timing = json.loads(proc.stdout.strip().splitlines()[-1])
    packed = np.load(root / "smoke_output.npz")
    first = {
        name: packed[name].astype(np.float32, copy=False)
        for name in ("prediction", "lower", "upper")
    }
    second = {
        name: packed[f"{name}2"].astype(np.float32, copy=False)
        for name in ("prediction", "lower", "upper")
    }
    return first, second, timing


def _validate_outputs(
    reference: tuple[np.ndarray, np.ndarray, np.ndarray],
    first: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
    *,
    tolerance: float,
) -> dict[str, float]:
    ref = dict(zip(("prediction", "lower", "upper"), reference))
    diffs: dict[str, float] = {}
    deterministic: dict[str, float] = {}

    for name in ("prediction", "lower", "upper"):
        value = first.get(name)
        if value is None:
            raise ValueError(f"package output missing {name}")
        if value.shape != ref[name].shape:
            raise ValueError(
                f"{name} shape mismatch: {value.shape} != {ref[name].shape}"
            )
        if value.dtype != np.float32:
            raise ValueError(f"{name} dtype must be float32")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")

        diff = float(np.max(np.abs(value - ref[name])))
        diffs[name] = diff
        if diff > tolerance:
            raise ValueError(
                f"{name} parity failed: {diff} > {tolerance}"
            )

        det = float(np.max(np.abs(value - second[name])))
        deterministic[name] = det
        if det != 0.0:
            raise ValueError(
                f"{name} is not deterministic: max abs diff {det}"
            )

    prediction = first["prediction"]
    lower = first["lower"]
    upper = first["upper"]
    if np.any(lower > prediction) or np.any(prediction > upper):
        raise ValueError("prediction must satisfy lower <= prediction <= upper")
    if float(np.max(np.abs(prediction[..., 2]))) != 0.0:
        raise ValueError("prediction pressure must be exactly zero")
    if float(np.max(np.abs(upper[..., 2] - lower[..., 2]))) != 0.0:
        raise ValueError("pressure interval width must be exactly zero")

    return {
        "parity_prediction_max_abs": diffs["prediction"],
        "parity_lower_max_abs": diffs["lower"],
        "parity_upper_max_abs": diffs["upper"],
        "deterministic_prediction_max_abs": deterministic["prediction"],
        "deterministic_lower_max_abs": deterministic["lower"],
        "deterministic_upper_max_abs": deterministic["upper"],
        "mean_width_uv": float(
            np.mean(upper[..., :2] - lower[..., :2])
        ),
        "coverage_of_prediction": float(
            np.mean(
                (lower[..., :2] <= prediction[..., :2])
                & (prediction[..., :2] <= upper[..., :2])
            )
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if sha256(args.backbone_checkpoint) != EXPECTED_BACKBONE_SHA256:
        raise ValueError("unexpected Full Joint backbone SHA")
    if sha256(args.corrector_checkpoint) != EXPECTED_CORRECTOR_SHA256:
        raise ValueError("unexpected Full Joint corrector SHA")

    x = _load_input(args.fixture)
    reference = _reference(
        backbone_checkpoint=args.backbone_checkpoint,
        corrector_checkpoint=args.corrector_checkpoint,
        head_checkpoint=args.head_checkpoint,
        kit_root=args.kit_root,
        x=x,
    )

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix="full_joint_teammate_sps_smoke_"
    ) as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(args.zip, "r") as archive:
            archive.testzip()
            archive.extractall(root)
        first, second, timing = _run_clean_room(root, x)

    checks = _validate_outputs(
        reference,
        first,
        second,
        tolerance=args.tolerance,
    )
    report = {
        "status": STATUS,
        "verification_status": "PASS",
        "ready_for_manual_submission": True,
        **checks,
        "first_call_seconds": float(timing["first_call_seconds"]),
        "second_call_seconds": float(timing["second_call_seconds"]),
        "peak_cuda_memory_allocated": int(
            timing["peak_cuda_memory_allocated"]
        ),
        "zip": str(args.zip),
        "zip_bytes": int(args.zip.stat().st_size),
        "zip_sha256": sha256(args.zip),
        "fixture": str(args.fixture),
        "backbone_sha256": sha256(args.backbone_checkpoint),
        "corrector_sha256": sha256(args.corrector_checkpoint),
        "head_sha256": sha256(args.head_checkpoint),
        "total_seconds": time.perf_counter() - started,
        "locked_final_accessed": False,
        "codabench_accessed": False,
        "auto_submit": False,
    }
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--corrector-checkpoint", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
