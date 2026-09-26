#!/usr/bin/env python3
"""Clean-room verifier for Full Joint@6500 + SPS@2000 submission ZIP.

The verifier compares the unpacked submission against the original frozen
backbone/corrector/SPS assets on the same safe fixture.  It does not train,
recalibrate, access locked-final data, or submit anywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

import h5py
import numpy as np
import torch

import realpde_sota_merge_joint_runtime as joint
import realpde_sota_v2_integrated as strong
from build_sota_merge_full_package import (
    EXPECTED_BACKBONE_SHA256,
    EXPECTED_CORRECTOR_SHA256,
    EXPECTED_HEAD_SHA256,
    EXPECTED_JOINT_PROTOCOL,
    EXPECTED_JOINT_UPDATE,
    EXPECTED_SPS_STEP,
    FROZEN_CALIBRATION,
    sha256,
)
from realpde_adaptive_probe import (
    ResidualCorrector3D,
    feature_config_from_checkpoint,
)
from realpde_p0_features import P0FeatureBuilder
from sota_merge_full_sps_runtime import (
    Head3D,
    HeadConfig,
    adaptive_half_width,
)

STATUS = "REVIEW_REQUIRED"


def _require_sha(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: {actual} != {expected}")


def _load_input(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        u = np.asarray(handle["u"][:20], dtype=np.float32)
        v = np.asarray(handle["v"][:20], dtype=np.float32)
    if u.shape == (20, 64, 128):
        u, v = u[:, ::2, ::2], v[:, ::2, ::2]
    if u.shape != (20, 32, 64) or v.shape != u.shape:
        raise ValueError(
            f"fixture must resolve to (20,32,64), got u={u.shape}, v={v.shape}"
        )
    p = np.zeros_like(u)
    x = np.stack([u, v, p], axis=-1)[None]
    if not np.isfinite(x).all():
        raise ValueError("fixture contains non-finite values")
    return np.ascontiguousarray(x, dtype=np.float32)


@torch.no_grad()
def _reference_outputs(
    *,
    backbone_checkpoint: Path,
    corrector_checkpoint: Path,
    head_checkpoint: Path,
    kit_root: Path,
    x: np.ndarray,
) -> dict[str, np.ndarray]:
    _require_sha(backbone_checkpoint, EXPECTED_BACKBONE_SHA256, "backbone")
    _require_sha(corrector_checkpoint, EXPECTED_CORRECTOR_SHA256, "corrector")
    _require_sha(head_checkpoint, EXPECTED_HEAD_SHA256, "SPS head")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bp = torch.load(backbone_checkpoint, map_location="cpu", weights_only=False)
    if bp.get("protocol") != EXPECTED_JOINT_PROTOCOL:
        raise ValueError("reference backbone protocol mismatch")
    if int(bp.get("joint_update", -1)) != EXPECTED_JOINT_UPDATE:
        raise ValueError("reference backbone joint_update mismatch")
    cfg = feature_config_from_checkpoint(bp)
    builder = P0FeatureBuilder(cfg).to(device)
    backbone = strong.MF01CNO(kit_root, len(builder.feature_names), device)
    backbone.load_state_dict(bp["model_state_dict"], strict=True)
    backbone.eval()

    cp = torch.load(corrector_checkpoint, map_location="cpu", weights_only=False)
    if cp.get("backbone_sha256") != EXPECTED_BACKBONE_SHA256:
        raise ValueError("reference corrector/backbone provenance mismatch")
    corrector = ResidualCorrector3D(
        in_channels=42, hidden=64, blocks=2, max_delta=0.04
    ).to(device)
    corrector.load_state_dict(cp["corrector_state_dict"], strict=True)
    corrector.eval()

    hp = torch.load(head_checkpoint, map_location="cpu", weights_only=False)
    selected_step = int(hp.get("selected_step", hp.get("step", -1)))
    if selected_step != EXPECTED_SPS_STEP:
        raise ValueError("reference SPS selected_step mismatch")
    if hp.get("backbone_sha256") != EXPECTED_BACKBONE_SHA256:
        raise ValueError("reference SPS backbone provenance mismatch")
    if hp.get("corrector_sha256") != EXPECTED_CORRECTOR_SHA256:
        raise ValueError("reference SPS corrector provenance mismatch")
    raw_calibration = hp.get("calibration")
    if raw_calibration is None:
        calibration = dict(FROZEN_CALIBRATION)
    else:
        if not isinstance(raw_calibration, dict):
            raise ValueError("reference SPS calibration metadata has unexpected type")
        calibration = {
            key: float(raw_calibration[key]) for key in FROZEN_CALIBRATION
        }
        if calibration != FROZEN_CALIBRATION:
            raise ValueError("reference SPS frozen calibration mismatch")
    hcfg = HeadConfig(**hp["head_config"])
    hcfg.validate()
    head = Head3D(hcfg).to(device)
    head.load_state_dict(hp["head_state_dict"], strict=True)
    head.eval()

    tensor = torch.from_numpy(x).to(device)
    base, prediction = joint.forward_final(
        backbone, builder, corrector, tensor
    )
    sigma = torch.exp(head(tensor, base))
    half = adaptive_half_width(
        prediction,
        sigma,
        floor=calibration["floor"],
        mult_u=calibration["mult_u"],
        mult_v=calibration["mult_v"],
        rel=calibration["rel"],
    )
    lower = prediction - half
    upper = prediction + half
    return {
        "prediction": prediction.cpu().numpy().astype(np.float32),
        "lower": lower.cpu().numpy().astype(np.float32),
        "upper": upper.cpu().numpy().astype(np.float32),
    }


def _safe_zip_infos(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    infos = archive.infolist()
    names: set[str] = set()
    for info in infos:
        name = info.filename
        if name in names:
            raise ValueError(f"duplicate ZIP member: {name}")
        names.add(name)
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"unsafe ZIP path: {name}")
        mode = (info.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            raise ValueError(f"symlink not allowed in ZIP: {name}")
    return infos


def _verify_manifest(root: Path, zip_names: set[str]) -> dict[str, object]:
    manifest_path = root / "MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError("package lacks MANIFEST.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("joint_protocol") != EXPECTED_JOINT_PROTOCOL:
        raise ValueError("manifest joint protocol mismatch")
    if int(manifest.get("joint_update", -1)) != EXPECTED_JOINT_UPDATE:
        raise ValueError("manifest joint_update mismatch")
    if int(manifest.get("sps_selected_step", -1)) != EXPECTED_SPS_STEP:
        raise ValueError("manifest SPS step mismatch")
    if manifest.get("source_assets") != {
        "backbone_sha256": EXPECTED_BACKBONE_SHA256,
        "corrector_sha256": EXPECTED_CORRECTOR_SHA256,
        "head_sha256": EXPECTED_HEAD_SHA256,
    }:
        raise ValueError("manifest frozen source asset SHA mismatch")
    calibration = {
        key: float(manifest["calibration"][key])
        for key in FROZEN_CALIBRATION
    }
    if calibration != FROZEN_CALIBRATION:
        raise ValueError("manifest frozen calibration mismatch")

    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest has no file inventory")
    expected_names = {"MANIFEST.json"}
    for row in rows:
        rel = str(row["path"])
        expected_names.add(rel)
        path = root / rel
        if not path.is_file():
            raise ValueError(f"manifest file missing after extraction: {rel}")
        size = int(path.stat().st_size)
        if size != int(row["bytes"]):
            raise ValueError(f"manifest size mismatch: {rel}")
        digest = sha256(path)
        if digest != str(row["sha256"]):
            raise ValueError(f"manifest SHA256 mismatch: {rel}")

    if zip_names != expected_names:
        missing = sorted(expected_names - zip_names)
        extra = sorted(zip_names - expected_names)
        raise ValueError(
            f"ZIP/manifest inventory mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    return manifest


def _run_clean_room(
    root: Path,
    x: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    np.save(root / "smoke_input.npy", x)
    script = r'''
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

for key in ("prediction", "lower", "upper"):
    if not np.array_equal(a[key], b[key]):
        raise RuntimeError("nondeterministic:" + key)

np.savez(
    "smoke_output.npz",
    prediction=a["prediction"],
    lower=a["lower"],
    upper=a["upper"],
)
print(json.dumps({
    "first_call_seconds": first,
    "second_call_seconds": second,
    "peak_cuda_memory_allocated": (
        int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available() else 0
    ),
}))
'''
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    timing = json.loads(proc.stdout.strip().splitlines()[-1])
    packed = np.load(root / "smoke_output.npz")
    result = {
        key: packed[key].astype(np.float32, copy=False)
        for key in ("prediction", "lower", "upper")
    }
    return result, {
        "first_call_seconds": float(timing["first_call_seconds"]),
        "second_call_seconds": float(timing["second_call_seconds"]),
        "peak_cuda_memory_allocated": int(
            timing["peak_cuda_memory_allocated"]
        ),
    }


def _validate_outputs(
    reference: dict[str, np.ndarray],
    result: dict[str, np.ndarray],
    *,
    tolerance: float,
) -> dict[str, float]:
    diffs: dict[str, float] = {}
    for key in ("prediction", "lower", "upper"):
        ref = reference[key]
        value = result[key]
        if value.shape != ref.shape:
            raise ValueError(f"{key} shape mismatch: {value.shape} != {ref.shape}")
        if value.dtype != np.float32:
            raise ValueError(f"{key} dtype must be float32")
        if not np.isfinite(value).all():
            raise ValueError(f"{key} contains non-finite values")
        diff = float(np.max(np.abs(value - ref)))
        diffs[f"max_abs_{key}_diff"] = diff
        if diff > tolerance:
            raise ValueError(f"{key} parity failed: {diff} > {tolerance}")

    pred = result["prediction"]
    lower = result["lower"]
    upper = result["upper"]
    if np.any(lower > pred) or np.any(pred > upper):
        raise ValueError("prediction must lie inside [lower, upper]")
    if float(np.max(np.abs(pred[..., 2]))) != 0.0:
        raise ValueError("pressure prediction must be exactly zero")
    if float(np.max(np.abs(upper[..., 2] - lower[..., 2]))) != 0.0:
        raise ValueError("pressure interval width must be exactly zero")

    ref_width = reference["upper"] - reference["lower"]
    got_width = upper - lower
    width_diff = float(np.max(np.abs(got_width - ref_width)))
    if width_diff > tolerance:
        raise ValueError(f"SPS width parity failed: {width_diff} > {tolerance}")

    return {
        **diffs,
        "max_abs_sps_width_diff": width_diff,
        "mean_width_uv": float(np.mean(got_width[..., :2])),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    x = _load_input(args.fixture)
    reference = _reference_outputs(
        backbone_checkpoint=args.backbone_checkpoint,
        corrector_checkpoint=args.corrector_checkpoint,
        head_checkpoint=args.head_checkpoint,
        kit_root=args.kit_root,
        x=x,
    )

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix="sota_merge_full_clean_room_"
    ) as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(args.zip, "r") as archive:
            infos = _safe_zip_infos(archive)
            archive.extractall(root)
        zip_names = {info.filename for info in infos if not info.is_dir()}
        manifest = _verify_manifest(root, zip_names)
        result, timing = _run_clean_room(root, x)

    checks = _validate_outputs(
        reference, result, tolerance=args.tolerance
    )
    report = {
        "status": "PASS_REVIEW_REQUIRED",
        "ready_for_manual_submission": True,
        **checks,
        **timing,
        "zip": str(args.zip),
        "zip_bytes": int(args.zip.stat().st_size),
        "zip_sha256": sha256(args.zip),
        "fixture": str(args.fixture),
        "backbone_sha256": sha256(args.backbone_checkpoint),
        "corrector_sha256": sha256(args.corrector_checkpoint),
        "head_sha256": sha256(args.head_checkpoint),
        "joint_protocol": EXPECTED_JOINT_PROTOCOL,
        "joint_update": EXPECTED_JOINT_UPDATE,
        "sps_selected_step": EXPECTED_SPS_STEP,
        "calibration": FROZEN_CALIBRATION,
        "manifest_package_protocol": manifest.get("package_protocol"),
        "total_seconds": time.perf_counter() - started,
        "locked_final_accessed": False,
        "codabench_accessed": False,
        "auto_submit": False,
        "next_action": "REVIEW_REQUIRED: manual submission only after human review.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
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
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
