#!/usr/bin/env python3
"""Build Full Joint@6500 + exact teammate-style SPS submission package.

This builder preserves the reviewed Full Joint point-prediction path exactly and
changes only the interval head/path:
- teammate exact 35-channel uncertainty features;
- h32/b2/drop0 head;
- symmetric per-element bounds = floor + mult * sigma;
- same multiplier for u/v;
- no rel term and no channel-specific multiplier.

It validates point-model provenance and the exact teammate SPS training metadata.
It never trains, recalibrates, touches locked-final/private data, or submits.
"""
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

STATUS = "REVIEW_REQUIRED"
MAX_ZIP_BYTES = 256 * 1024 * 1024
PACKAGE_PROTOCOL = "REALPDE_FULL_JOINT_6500_TEAMMATE_SPS_EXACT_V1"

EXPECTED_JOINT_PROTOCOL = "REALPDE_SOTA_MERGE_FULL_JOINT_V1"
EXPECTED_SPS_PROTOCOL = "REALPDE_FULL_JOINT_TEAMMATE_SPS_EXACT_V1"
EXPECTED_RECIPE = "teammate35_exact_full_joint"
EXPECTED_JOINT_UPDATE = 6_500
EXPECTED_BACKBONE_SHA256 = "e965958385004c0fa17e34d4b6cb0aea518cfa0b6181c7e79d3ec82eb931b305"
EXPECTED_CORRECTOR_SHA256 = "0abd4029b2e55225127b18e10fbcd6d47e8ce40fbeac8607de3d28c2a36908d9"
ALLOWED_BOUNDS = {
    (float(floor), float(mult))
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
    if not isinstance(state, dict) or not state:
        raise ValueError("backbone checkpoint lacks model_state_dict")
    if not all(name.startswith("cno.") for name in state):
        raise ValueError("Full Joint backbone state must contain only cno.* keys")
    return {name.removeprefix("cno."): value for name, value in state.items()}


def feature_config(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict) or "dx" not in raw or "dy" not in raw:
        raise ValueError("backbone checkpoint lacks frozen P0-A feature_config")
    return {
        "include_p0_a": bool(raw.get("include_p0_a", True)),
        "include_p0_b": bool(raw.get("include_p0_b", False)),
        "dx": float(raw["dx"]),
        "dy": float(raw["dy"]),
        "dt": None if raw.get("dt") is None else float(raw["dt"]),
        "re_center": float(raw.get("re_center", 0.0)),
        "re_scale": float(raw.get("re_scale", 1.0)),
    }


def require_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: {actual} != {expected}")
    return actual


def validate_assets(
    backbone_checkpoint: Path,
    corrector_checkpoint: Path,
    head_checkpoint: Path,
) -> dict[str, object]:
    backbone_sha = require_sha(
        backbone_checkpoint, EXPECTED_BACKBONE_SHA256, "backbone"
    )
    corrector_sha = require_sha(
        corrector_checkpoint, EXPECTED_CORRECTOR_SHA256, "corrector"
    )

    backbone = torch.load(
        backbone_checkpoint, map_location="cpu", weights_only=False
    )
    if backbone.get("protocol") != EXPECTED_JOINT_PROTOCOL:
        raise ValueError("backbone Full Joint protocol mismatch")
    if int(backbone.get("joint_update", -1)) != EXPECTED_JOINT_UPDATE:
        raise ValueError("backbone joint_update mismatch")
    if backbone.get("scope") != "full" or backbone.get("stage") != "JOINT":
        raise ValueError("backbone must be Full scope JOINT checkpoint")
    if backbone.get("feature_set") != "P0-A":
        raise ValueError("backbone must use P0-A")
    cfg = feature_config(backbone.get("feature_config"))
    cno_state = strip_mf_prefix(backbone.get("model_state_dict"))

    corrector = torch.load(
        corrector_checkpoint, map_location="cpu", weights_only=False
    )
    if corrector.get("protocol") != EXPECTED_JOINT_PROTOCOL:
        raise ValueError("corrector Full Joint protocol mismatch")
    if int(corrector.get("joint_update", -1)) != EXPECTED_JOINT_UPDATE:
        raise ValueError("corrector joint_update mismatch")
    if corrector.get("scope") != "full":
        raise ValueError("corrector must be Full scope")
    if corrector.get("backbone_sha256") != backbone_sha:
        raise ValueError("corrector is not bound to the exact frozen backbone")
    expected_arch = {
        "in_channels": 42,
        "hidden": 64,
        "blocks": 2,
        "max_delta": 0.04,
    }
    arch = corrector.get("architecture", {})
    for key, value in expected_arch.items():
        if arch.get(key, value) != value:
            raise ValueError(f"corrector architecture mismatch: {key}")
    corrector_state = corrector.get("corrector_state_dict")
    corrector_checker = ResidualCorrector3D(**expected_arch)
    corrector_checker.load_state_dict(corrector_state, strict=True)

    head_sha = sha256(head_checkpoint)
    head_payload = torch.load(head_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(head_payload, dict):
        raise ValueError("teammate SPS head checkpoint must be a mapping")
    head_state = head_payload.get("head_state_dict")
    meta = head_payload.get("metadata")
    if not isinstance(head_state, dict) or not head_state:
        raise ValueError("teammate SPS head lacks head_state_dict")
    if not isinstance(meta, dict):
        raise ValueError("teammate SPS head lacks metadata")
    if meta.get("protocol") != EXPECTED_SPS_PROTOCOL:
        raise ValueError("teammate SPS protocol mismatch")
    if meta.get("recipe") != EXPECTED_RECIPE:
        raise ValueError("teammate SPS recipe mismatch")

    point_meta = meta.get("point_model", {})
    if point_meta.get("protocol") != EXPECTED_JOINT_PROTOCOL:
        raise ValueError("teammate SPS point protocol mismatch")
    if int(point_meta.get("joint_update", -1)) != EXPECTED_JOINT_UPDATE:
        raise ValueError("teammate SPS point joint_update mismatch")
    if point_meta.get("backbone_sha256") != backbone_sha:
        raise ValueError("teammate SPS backbone provenance mismatch")
    if point_meta.get("corrector_sha256") != corrector_sha:
        raise ValueError("teammate SPS corrector provenance mismatch")
    if point_meta.get("frozen") is not True:
        raise ValueError("teammate SPS metadata must mark point model frozen")

    architecture = meta.get("architecture", {})
    expected_head_arch = {
        "in_channels": 35,
        "hidden": 32,
        "blocks": 2,
        "dropout": 0.0,
        "include_pressure": True,
    }
    if architecture != expected_head_arch:
        raise ValueError(
            f"teammate SPS architecture mismatch: {architecture} != "
            f"{expected_head_arch}"
        )
    if meta.get("loss") != "masked_gaussian_nll_nonzero_uv":
        raise ValueError("teammate SPS loss mismatch")
    if int(meta.get("max_updates", -1)) != 2000:
        raise ValueError("teammate SPS max_updates mismatch")
    if int(meta.get("eval_interval", -1)) != 200:
        raise ValueError("teammate SPS eval interval mismatch")
    selected_updates = int(meta.get("selected_updates", -1))
    if selected_updates not in range(200, 2001, 200):
        raise ValueError("selected_updates is not a teammate 200-step milestone")
    if int(meta.get("seed", -1)) != 41:
        raise ValueError("teammate SPS seed mismatch")
    if float(meta.get("lr", float("nan"))) != 1e-3:
        raise ValueError("teammate SPS lr mismatch")
    if float(meta.get("weight_decay", float("nan"))) != 1e-5:
        raise ValueError("teammate SPS weight_decay mismatch")
    if float(meta.get("sigma0", float("nan"))) != 0.02:
        raise ValueError("teammate SPS sigma0 mismatch")

    split = meta.get("split", {})
    expected_split = {
        "usable_trajectories": 81,
        "seed": 41,
        "train_trajectories": 65,
        "val_trajectories": 16,
        "train_windows": 2701,
        "val_windows": 640,
        "stride": 20,
        "sub_sample": 2,
    }
    for key, value in expected_split.items():
        if split.get(key) != value:
            raise ValueError(f"teammate SPS split mismatch: {key}")

    if feature_config(meta.get("feature_config")) != cfg:
        raise ValueError("teammate SPS/full feature_config mismatch")

    calibration = meta.get("calibration", {})
    floor = float(calibration.get("floor"))
    mult = float(calibration.get("mult"))
    if (floor, mult) not in ALLOWED_BOUNDS:
        raise ValueError("teammate SPS calibration outside exact 28-row grid")
    if calibration.get("same_multiplier_u_v") is not True:
        raise ValueError("teammate SPS must use one multiplier for u/v")
    if float(calibration.get("rel_term", float("nan"))) != 0.0:
        raise ValueError("teammate SPS exact package must have rel_term=0")

    head_checker = TeammateUncertaintyHead(
        hidden=32,
        blocks=2,
        dropout=0.0,
        include_pressure=True,
    )
    head_checker.load_state_dict(head_state, strict=True)

    return {
        "backbone_sha256": backbone_sha,
        "corrector_sha256": corrector_sha,
        "head_sha256": head_sha,
        "feature_config": cfg,
        "cno_state_dict": cno_state,
        "corrector_state_dict": corrector_state,
        "head_state_dict": head_state,
        "head_metadata": meta,
        "selected_updates": selected_updates,
        "calibration": {"floor": floor, "mult": mult},
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

        cno = CNO3d(
            in_dim=len(builder.feature_names),
            out_dim=5,
            out_dim_mult=1,
            in_size=64,
            N_layers=3,
        ).to(_DEVICE)
        cno.load_state_dict(payload["cno_state_dict"], strict=True)

        corrector = ResidualCorrector3D(
            in_channels=42, hidden=64, blocks=2, max_delta=0.04
        ).to(_DEVICE)
        corrector.load_state_dict(payload["corrector_state_dict"], strict=True)

        head = TeammateUncertaintyHead(
            hidden=32,
            blocks=2,
            dropout=0.0,
            include_pressure=True,
        ).to(_DEVICE)
        head.load_state_dict(payload["head_state_dict"], strict=True)

        cno.eval()
        corrector.eval()
        head.eval()
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

        # Exact teammate semantics: uncertainty observes pre-corrector base,
        # but bounds are centered on the final post-corrector prediction.
        sigma = sigma_from_log_std(head(tensor, base))
        half_uv = (
            float(payload["calibration"]["floor"])
            + float(payload["calibration"]["mult"]) * sigma
        )
        half = torch.cat(
            [half_uv, torch.zeros_like(half_uv[..., :1])],
            dim=-1,
        )
        lower = prediction - half
        upper = prediction + half

    return {
        "prediction": prediction.cpu().numpy().astype(np.float32),
        "lower": lower.cpu().numpy().astype(np.float32),
        "upper": upper.cpu().numpy().astype(np.float32),
    }
'''


def copy_runtime(staging: Path, kit_root: Path) -> None:
    here = Path(__file__).resolve().parent
    whitelist = {
        "sota_v2_adaptive_runtime.py": "adaptive_runtime.py",
        "realpde_adaptive_probe.py": "residual_runtime.py",
        "sps_teammate_uncertainty_runtime.py": "teammate_runtime.py",
        "realpde_p0_features.py": "realpde_p0_features.py",
    }
    for source_name, target_name in whitelist.items():
        source = here / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, staging / target_name)

    for dirname in ("rpde_baselines", "_vendor"):
        source = kit_root / dirname
        if not source.is_dir():
            raise FileNotFoundError(source)
        shutil.copytree(source, staging / dirname)


def inventory(root: Path, *, exclude: set[str] | None = None) -> list[dict[str, object]]:
    excluded = exclude or set()
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel in excluded:
            continue
        rows.append(
            {
                "path": rel,
                "bytes": int(path.stat().st_size),
                "sha256": sha256(path),
            }
        )
    return rows


def build(
    *,
    backbone_checkpoint: Path,
    corrector_checkpoint: Path,
    head_checkpoint: Path,
    kit_root: Path,
    out_root: Path,
    execution_commit: str,
) -> dict[str, object]:
    if out_root.exists():
        raise FileExistsError(f"refusing to overwrite OUT_ROOT: {out_root}")

    assets = validate_assets(
        backbone_checkpoint,
        corrector_checkpoint,
        head_checkpoint,
    )

    staging = out_root / "staging"
    staging.mkdir(parents=True)
    copy_runtime(staging, kit_root)

    torch.save(
        {
            "cno_state_dict": {
                key: value.detach().cpu()
                for key, value in assets["cno_state_dict"].items()
            },
            "corrector_state_dict": {
                key: value.detach().cpu()
                for key, value in assets["corrector_state_dict"].items()
            },
            "head_state_dict": {
                key: value.detach().cpu()
                for key, value in assets["head_state_dict"].items()
            },
            "feature_config": assets["feature_config"],
            "calibration": assets["calibration"],
            "joint_protocol": EXPECTED_JOINT_PROTOCOL,
            "joint_update": EXPECTED_JOINT_UPDATE,
            "sps_protocol": EXPECTED_SPS_PROTOCOL,
            "sps_recipe": EXPECTED_RECIPE,
            "sps_selected_updates": assets["selected_updates"],
            "source_backbone_sha256": assets["backbone_sha256"],
            "source_corrector_sha256": assets["corrector_sha256"],
            "source_head_sha256": assets["head_sha256"],
        },
        staging / "model.pth",
    )
    (staging / "submission.py").write_text(
        submission_source(),
        encoding="utf-8",
    )

    payload_inventory = inventory(staging, exclude={"MANIFEST.json"})
    manifest = {
        "status": STATUS,
        "package_protocol": PACKAGE_PROTOCOL,
        "joint_protocol": EXPECTED_JOINT_PROTOCOL,
        "joint_update": EXPECTED_JOINT_UPDATE,
        "sps_protocol": EXPECTED_SPS_PROTOCOL,
        "sps_recipe": EXPECTED_RECIPE,
        "sps_selected_updates": assets["selected_updates"],
        "source_assets": {
            "backbone_sha256": assets["backbone_sha256"],
            "corrector_sha256": assets["corrector_sha256"],
            "head_sha256": assets["head_sha256"],
        },
        "calibration": assets["calibration"],
        "files": payload_inventory,
        "locked_final_accessed": False,
        "codabench_accessed": False,
        "auto_submit": False,
    }
    (staging / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    zip_path = out_root / "submission.zip"
    full_inventory = inventory(staging)
    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for item in full_inventory:
            archive.write(
                staging / str(item["path"]),
                str(item["path"]),
            )

    if zip_path.stat().st_size >= MAX_ZIP_BYTES:
        raise ValueError(
            f"submission ZIP exceeds 256 MiB: {zip_path.stat().st_size}"
        )

    report = {
        "status": STATUS,
        "ready_for_manual_submission": True,
        "execution_commit": execution_commit,
        "package_protocol": PACKAGE_PROTOCOL,
        "backbone_checkpoint": str(backbone_checkpoint),
        "backbone_sha256": assets["backbone_sha256"],
        "corrector_checkpoint": str(corrector_checkpoint),
        "corrector_sha256": assets["corrector_sha256"],
        "head_checkpoint": str(head_checkpoint),
        "head_sha256": assets["head_sha256"],
        "selected_updates": assets["selected_updates"],
        "calibration": assets["calibration"],
        "zip": str(zip_path),
        "zip_bytes": int(zip_path.stat().st_size),
        "zip_sha256": sha256(zip_path),
        "created_at": time.time(),
        "locked_final_accessed": False,
        "codabench_accessed": False,
        "auto_submit": False,
    }
    (out_root / "package_build.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--corrector-checkpoint", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    args = parser.parse_args()
    build(
        backbone_checkpoint=args.backbone_checkpoint,
        corrector_checkpoint=args.corrector_checkpoint,
        head_checkpoint=args.head_checkpoint,
        kit_root=args.kit_root,
        out_root=args.out_root,
        execution_commit=args.execution_commit,
    )


if __name__ == "__main__":
    main()
