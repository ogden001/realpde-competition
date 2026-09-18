#!/usr/bin/env python3
"""Build the white-list SOTA-V3 Track-1 submission package."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

import torch

from realpde_adaptive_probe import ResidualCorrector3D
from sps_teammate_uncertainty_runtime import TeammateUncertaintyHead

MAX_ZIP_BYTES = 256 * 1024 * 1024
ALLOWED_BOUNDS = {
    (floor, mult)
    for floor in (0.0, 0.0025, 0.005, 0.0075)
    for mult in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strip_mf_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state or not all(name.startswith("cno.") for name in state):
        raise ValueError("MF state must contain only cno.* keys")
    return {name.removeprefix("cno."): value for name, value in state.items()}


def _feature_config(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict) or "dx" not in raw or "dy" not in raw:
        raise ValueError("backbone checkpoint lacks feature_config")
    return {
        "include_p0_a": bool(raw.get("include_p0_a", True)),
        "include_p0_b": bool(raw.get("include_p0_b", False)),
        "dx": float(raw["dx"]),
        "dy": float(raw["dy"]),
        "dt": None if raw.get("dt") is None else float(raw["dt"]),
        "re_center": float(raw.get("re_center", 0.0)),
        "re_scale": float(raw.get("re_scale", 1.0)),
    }


def submission_source() -> str:
    return r'''from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "_vendor"))
from rpde_baselines.model.cno import CNO3d
from realpde_p0_features import P0FeatureBuilder, P0FeatureConfig
from adaptive_runtime import mf_forward
from residual_runtime import ResidualCorrector3D, adaptive_features
from teammate_runtime import TeammateUncertaintyHead, sigma_from_log_std

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODELS = None
_EPS = 1e-12

def _spatial_tke_map_projection(base, corrected):
    b_uv, c_uv = base[..., :2], corrected[..., :2]
    b_fluct = b_uv - b_uv.mean(dim=1, keepdim=True)
    c_mean = c_uv.mean(dim=1, keepdim=True)
    c_fluct = c_uv - c_mean
    numerator = b_fluct.double().square().mean(dim=1).sum(dim=-1)
    denominator = c_fluct.double().square().mean(dim=1).sum(dim=-1)
    scale = torch.where(
        denominator > _EPS,
        torch.sqrt(numerator.clamp_min(0.0) / denominator.clamp_min(_EPS)),
        torch.ones_like(denominator),
    ).to(dtype=corrected.dtype)[:, None, :, :, None]
    out = corrected.clone()
    out[..., :2] = c_mean + scale * c_fluct
    out[..., 2] = 0.0
    return out

def _load_models():
    global _MODELS
    if _MODELS is None:
        payload = torch.load(_ROOT / "model.pth", map_location="cpu", weights_only=False)
        config = P0FeatureConfig(**payload["feature_config"])
        builder = P0FeatureBuilder(config).to(_DEVICE)
        cno = CNO3d(in_dim=len(builder.feature_names), out_dim=5, out_dim_mult=1, in_size=64, N_layers=3).to(_DEVICE)
        cno.load_state_dict(payload["cno_state_dict"], strict=True)
        corrector = ResidualCorrector3D(in_channels=42, hidden=64, blocks=2, max_delta=0.04).to(_DEVICE)
        corrector.load_state_dict(payload["corrector_state_dict"], strict=True)
        head = TeammateUncertaintyHead(hidden=32, blocks=2, dropout=0.0, include_pressure=True).to(_DEVICE)
        head.load_state_dict(payload["head_state_dict"], strict=True)
        cno.eval(); corrector.eval(); head.eval()
        _MODELS = payload, builder, cno, corrector, head
    return _MODELS

def predict(input_array, metadata=None):
    x = np.asarray(input_array, dtype=np.float32)
    if x.ndim != 5 or x.shape[1:] != (20, 32, 64, 3):
        raise ValueError("input_array must have shape (N,20,32,64,3)")
    if x.shape[0] < 1 or not np.isfinite(x).all():
        raise ValueError("input_array must contain finite samples")
    payload, builder, cno, corrector, head = _load_models()
    with torch.inference_mode():
        tensor = torch.from_numpy(np.ascontiguousarray(x)).to(_DEVICE)
        base = mf_forward(cno, builder, tensor)
        features = adaptive_features(tensor, base).permute(0, 4, 1, 2, 3).contiguous()
        delta = corrector(features).permute(0, 2, 3, 4, 1).contiguous()
        corrected = base + delta
        corrected[..., 2] = 0.0
        prediction = _spatial_tke_map_projection(base, corrected)
        sigma = sigma_from_log_std(head(tensor, base))
        half_uv = float(payload["bound_floor"]) + float(payload["bound_mult"]) * sigma
        half = torch.cat([half_uv, torch.zeros_like(half_uv[..., :1])], dim=-1)
        lower, upper = prediction - half, prediction + half
    return {
        "prediction": prediction.cpu().numpy().astype(np.float32),
        "lower": lower.cpu().numpy().astype(np.float32),
        "upper": upper.cpu().numpy().astype(np.float32),
    }
'''


def build(*, backbone_checkpoint: Path, corrector_checkpoint: Path,
          head_checkpoint: Path, kit_root: Path, out_root: Path,
          execution_commit: str) -> dict[str, object]:
    if out_root.exists():
        raise FileExistsError(out_root)
    backbone = torch.load(backbone_checkpoint, map_location="cpu", weights_only=False)
    if backbone.get("feature_set") != "P0-A":
        raise ValueError("backbone must be P0-A")
    cno_state = strip_mf_prefix(backbone.get("model_state_dict", {}))
    config = _feature_config(backbone.get("feature_config"))
    backbone_sha = sha256(backbone_checkpoint)

    corr = torch.load(corrector_checkpoint, map_location="cpu", weights_only=False)
    if corr.get("backbone_sha256") != backbone_sha:
        raise ValueError("corrector/backbone SHA mismatch")
    corr_state = corr.get("corrector_state_dict")
    checker = ResidualCorrector3D(in_channels=42, hidden=64, blocks=2, max_delta=0.04)
    checker.load_state_dict(corr_state, strict=True)

    head_payload = torch.load(head_checkpoint, map_location="cpu", weights_only=False)
    head_state = head_payload.get("head_state_dict")
    meta = head_payload.get("metadata", {})
    head_checker = TeammateUncertaintyHead(hidden=32, blocks=2, dropout=0.0, include_pressure=True)
    head_checker.load_state_dict(head_state, strict=True)
    if meta.get("backbone_sha256") != backbone_sha:
        raise ValueError("head/backbone SHA mismatch")
    if meta.get("corrector_sha256") != sha256(corrector_checkpoint):
        raise ValueError("head/corrector SHA mismatch")
    if _feature_config(meta.get("feature_config")) != config:
        raise ValueError("head/backbone feature_config mismatch")
    floor, mult = float(meta["bound_floor"]), float(meta["bound_mult"])
    if (floor, mult) not in ALLOWED_BOUNDS:
        raise ValueError("head bounds outside frozen calibration grid")

    staging = out_root / "staging"
    staging.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    whitelist = {
        "sota_v2_adaptive_runtime.py": "adaptive_runtime.py",
        "realpde_adaptive_probe.py": "residual_runtime.py",
        "sps_teammate_uncertainty_runtime.py": "teammate_runtime.py",
        "realpde_p0_features.py": "realpde_p0_features.py",
    }
    for src, dst in whitelist.items():
        shutil.copy2(here / src, staging / dst)
    shutil.copytree(kit_root / "rpde_baselines", staging / "rpde_baselines")
    shutil.copytree(kit_root / "_vendor", staging / "_vendor")
    torch.save({
        "cno_state_dict": {k: v.detach().cpu() for k, v in cno_state.items()},
        "corrector_state_dict": {k: v.detach().cpu() for k, v in corr_state.items()},
        "head_state_dict": {k: v.detach().cpu() for k, v in head_state.items()},
        "feature_config": config,
        "backbone_iteration": int(backbone.get("iteration", -1)),
        "bound_floor": floor,
        "bound_mult": mult,
        "backbone_checkpoint_sha256": backbone_sha,
        "corrector_checkpoint_sha256": sha256(corrector_checkpoint),
        "head_checkpoint_sha256": sha256(head_checkpoint),
        "recipe": "SOTA-V3_AoA+ResidualSpatialTKE+Teammate35",
    }, staging / "model.pth")
    (staging / "submission.py").write_text(submission_source(), encoding="utf-8")

    inventory = [
        {"path": str(p.relative_to(staging)), "bytes": p.stat().st_size}
        for p in sorted(staging.rglob("*")) if p.is_file()
    ]
    zip_path = out_root / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in inventory:
            archive.write(staging / item["path"], item["path"])
    if zip_path.stat().st_size >= MAX_ZIP_BYTES:
        raise ValueError("submission ZIP exceeds 256 MiB")
    report = {
        "status": "PACKAGED_REVIEW_REQUIRED",
        "execution_commit": execution_commit,
        "backbone_sha256": backbone_sha,
        "corrector_sha256": sha256(corrector_checkpoint),
        "head_sha256": sha256(head_checkpoint),
        "bounds": {"floor": floor, "mult": mult},
        "zip": str(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": sha256(zip_path),
        "inventory": inventory,
        "created_at": time.time(),
        "codabench_submitted": False,
    }
    (out_root / "package_build.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone-checkpoint", type=Path, required=True)
    p.add_argument("--corrector-checkpoint", type=Path, required=True)
    p.add_argument("--head-checkpoint", type=Path, required=True)
    p.add_argument("--kit-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--execution-commit", required=True)
    a = p.parse_args()
    build(
        backbone_checkpoint=a.backbone_checkpoint,
        corrector_checkpoint=a.corrector_checkpoint,
        head_checkpoint=a.head_checkpoint,
        kit_root=a.kit_root,
        out_root=a.out_root,
        execution_commit=a.execution_commit,
    )


if __name__ == "__main__":
    main()
