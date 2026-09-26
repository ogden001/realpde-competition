#!/usr/bin/env python3
"""Build the frozen Full Joint@6500 + SPS@2000 Track-1 submission package.

This is an inference-only packaging adapter.  It does not train, recalibrate,
select checkpoints, access locked-final data, or submit anywhere.
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
from sota_merge_full_sps_runtime import Head3D, HeadConfig

STATUS = "REVIEW_REQUIRED"
MAX_ZIP_BYTES = 256 * 1024 * 1024

EXPECTED_JOINT_PROTOCOL = "REALPDE_SOTA_MERGE_FULL_JOINT_V1"
EXPECTED_JOINT_UPDATE = 6_500
EXPECTED_SPS_STEP = 2_000

EXPECTED_BACKBONE_SHA256 = "e965958385004c0fa17e34d4b6cb0aea518cfa0b6181c7e79d3ec82eb931b305"
EXPECTED_CORRECTOR_SHA256 = "0abd4029b2e55225127b18e10fbcd6d47e8ce40fbeac8607de3d28c2a36908d9"
EXPECTED_HEAD_SHA256 = "6796e6ebff8c623596312e462e93323fd9a69da118c63b73b8166279d6ba8728"

FROZEN_CALIBRATION = {
    "floor": 0.0025,
    "mult_u": 1.25,
    "mult_v": 1.50,
    "rel": 0.0025,
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


def _feature_config(raw: object) -> dict[str, object]:
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


def _require_exact_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: {actual} != {expected}")
    return actual


def _validate_calibration(raw: object) -> dict[str, float]:
    """Return the reviewed frozen calibration.

    The selected SPS @2000 artifact is a milestone snapshot created by
    save_head_snapshot(), which intentionally stores model/provenance metadata
    but no calibration. Calibration was selected and frozen separately during
    the reviewed SPS audit. If a checkpoint does carry calibration, require it
    to match exactly; otherwise use the package-level frozen constants.
    """
    if raw is None:
        return dict(FROZEN_CALIBRATION)
    if not isinstance(raw, dict):
        raise ValueError("SPS calibration metadata has unexpected type")
    got = {key: float(raw[key]) for key in FROZEN_CALIBRATION}
    if got != FROZEN_CALIBRATION:
        raise ValueError(
            f"frozen SPS calibration mismatch: {got} != {FROZEN_CALIBRATION}"
        )
    return got


def _validate_assets(
    backbone_checkpoint: Path,
    corrector_checkpoint: Path,
    head_checkpoint: Path,
) -> dict[str, object]:
    backbone_sha = _require_exact_sha(
        backbone_checkpoint, EXPECTED_BACKBONE_SHA256, "backbone"
    )
    corrector_sha = _require_exact_sha(
        corrector_checkpoint, EXPECTED_CORRECTOR_SHA256, "corrector"
    )
    head_sha = _require_exact_sha(head_checkpoint, EXPECTED_HEAD_SHA256, "SPS head")

    backbone = torch.load(backbone_checkpoint, map_location="cpu", weights_only=False)
    if backbone.get("protocol") != EXPECTED_JOINT_PROTOCOL:
        raise ValueError("backbone Full Joint protocol mismatch")
    if int(backbone.get("joint_update", -1)) != EXPECTED_JOINT_UPDATE:
        raise ValueError("backbone joint_update mismatch")
    if backbone.get("scope") != "full" or backbone.get("stage") != "JOINT":
        raise ValueError("backbone must be Full scope JOINT checkpoint")
    if backbone.get("feature_set") != "P0-A":
        raise ValueError("backbone must use P0-A")
    feature_config = _feature_config(backbone.get("feature_config"))
    cno_state = strip_mf_prefix(backbone.get("model_state_dict"))

    corrector = torch.load(corrector_checkpoint, map_location="cpu", weights_only=False)
    if corrector.get("protocol") != EXPECTED_JOINT_PROTOCOL:
        raise ValueError("corrector Full Joint protocol mismatch")
    if int(corrector.get("joint_update", -1)) != EXPECTED_JOINT_UPDATE:
        raise ValueError("corrector joint_update mismatch")
    if corrector.get("scope") != "full":
        raise ValueError("corrector must be Full scope")
    if corrector.get("backbone_sha256") != backbone_sha:
        raise ValueError("corrector is not bound to the exact frozen backbone")
    arch = corrector.get("architecture", {})
    expected_arch = {
        "in_channels": 42,
        "hidden": 64,
        "blocks": 2,
        "max_delta": 0.04,
    }
    for key, value in expected_arch.items():
        if arch.get(key, value) != value:
            raise ValueError(f"corrector architecture mismatch: {key}")
    corrector_state = corrector.get("corrector_state_dict")
    checker = ResidualCorrector3D(**expected_arch)
    checker.load_state_dict(corrector_state, strict=True)

    head_payload = torch.load(head_checkpoint, map_location="cpu", weights_only=False)
    head_step = int(
        head_payload.get("selected_step", head_payload.get("step", -1))
    )
    if head_step != EXPECTED_SPS_STEP:
        raise ValueError(f"SPS selected step mismatch: {head_step}")
    if int(head_payload.get("joint_update", -1)) != EXPECTED_JOINT_UPDATE:
        raise ValueError("SPS head joint_update mismatch")
    if head_payload.get("backbone_sha256") != backbone_sha:
        raise ValueError("SPS head backbone provenance mismatch")
    if head_payload.get("corrector_sha256") != corrector_sha:
        raise ValueError("SPS head corrector provenance mismatch")
    head_config_raw = head_payload.get("head_config")
    if not isinstance(head_config_raw, dict):
        raise ValueError("SPS head lacks head_config")
    head_config = HeadConfig(**head_config_raw)
    head_config.validate()
    head_state = head_payload.get("head_state_dict")
    head_checker = Head3D(head_config)
    head_checker.load_state_dict(head_state, strict=True)
    calibration = _validate_calibration(head_payload.get("calibration"))

    return {
        "backbone_sha256": backbone_sha,
        "corrector_sha256": corrector_sha,
        "head_sha256": head_sha,
        "feature_config": feature_config,
        "cno_state_dict": cno_state,
        "corrector_state_dict": corrector_state,
        "head_state_dict": head_state,
        "head_config": head_config_raw,
        "head_protocol": head_payload.get("protocol"),
        "calibration": calibration,
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
from sps_runtime import Head3D, HeadConfig, adaptive_half_width

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

        head_config = HeadConfig(**payload["head_config"])
        head = Head3D(head_config).to(_DEVICE)
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

        log_std = head(tensor, base)
        sigma = torch.exp(log_std)
        half = adaptive_half_width(
            prediction,
            sigma,
            floor=float(payload["calibration"]["floor"]),
            mult_u=float(payload["calibration"]["mult_u"]),
            mult_v=float(payload["calibration"]["mult_v"]),
            rel=float(payload["calibration"]["rel"]),
        )
        lower = prediction - half
        upper = prediction + half

    return {
        "prediction": prediction.cpu().numpy().astype(np.float32),
        "lower": lower.cpu().numpy().astype(np.float32),
        "upper": upper.cpu().numpy().astype(np.float32),
    }
'''


def _copy_runtime(staging: Path, kit_root: Path) -> None:
    here = Path(__file__).resolve().parent
    whitelist = {
        "sota_v2_adaptive_runtime.py": "adaptive_runtime.py",
        "realpde_adaptive_probe.py": "residual_runtime.py",
        "sota_merge_full_sps_runtime.py": "sps_runtime.py",
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


def _inventory(root: Path, *, exclude: set[str] | None = None) -> list[dict[str, object]]:
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

    assets = _validate_assets(
        backbone_checkpoint, corrector_checkpoint, head_checkpoint
    )

    staging = out_root / "staging"
    staging.mkdir(parents=True)
    _copy_runtime(staging, kit_root)

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
            "head_config": assets["head_config"],
            "calibration": assets["calibration"],
            "joint_protocol": EXPECTED_JOINT_PROTOCOL,
            "joint_update": EXPECTED_JOINT_UPDATE,
            "sps_selected_step": EXPECTED_SPS_STEP,
            "source_backbone_sha256": assets["backbone_sha256"],
            "source_corrector_sha256": assets["corrector_sha256"],
            "source_head_sha256": assets["head_sha256"],
            "source_head_protocol": assets["head_protocol"],
        },
        staging / "model.pth",
    )
    (staging / "submission.py").write_text(
        submission_source(), encoding="utf-8"
    )

    payload_inventory = _inventory(staging, exclude={"MANIFEST.json"})
    manifest = {
        "status": STATUS,
        "package_protocol": "REALPDE_FULL_JOINT_6500_SPS_2000_PACKAGE_V1",
        "joint_protocol": EXPECTED_JOINT_PROTOCOL,
        "joint_update": EXPECTED_JOINT_UPDATE,
        "sps_selected_step": EXPECTED_SPS_STEP,
        "source_assets": {
            "backbone_sha256": assets["backbone_sha256"],
            "corrector_sha256": assets["corrector_sha256"],
            "head_sha256": assets["head_sha256"],
        },
        "calibration": assets["calibration"],
        "calibration_source": "review_frozen_package_constants",
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
    full_inventory = _inventory(staging)
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for item in full_inventory:
            archive.write(staging / str(item["path"]), str(item["path"]))

    if zip_path.stat().st_size >= MAX_ZIP_BYTES:
        raise ValueError(
            f"submission ZIP exceeds 256 MiB: {zip_path.stat().st_size}"
        )

    report = {
        "status": "PACKAGED_REVIEW_REQUIRED",
        "ready_for_manual_submission": False,
        "execution_commit": execution_commit,
        "backbone_checkpoint": str(backbone_checkpoint),
        "backbone_sha256": assets["backbone_sha256"],
        "corrector_checkpoint": str(corrector_checkpoint),
        "corrector_sha256": assets["corrector_sha256"],
        "head_checkpoint": str(head_checkpoint),
        "head_sha256": assets["head_sha256"],
        "joint_protocol": EXPECTED_JOINT_PROTOCOL,
        "joint_update": EXPECTED_JOINT_UPDATE,
        "sps_selected_step": EXPECTED_SPS_STEP,
        "calibration": assets["calibration"],
        "calibration_source": "review_frozen_package_constants",
        "zip": str(zip_path),
        "zip_bytes": int(zip_path.stat().st_size),
        "zip_sha256": sha256(zip_path),
        "manifest_sha256": sha256(staging / "MANIFEST.json"),
        "inventory_count": len(full_inventory),
        "created_at": time.time(),
        "locked_final_accessed": False,
        "codabench_accessed": False,
        "auto_submit": False,
        "next_action": "REVIEW_REQUIRED: run verify_sota_merge_full_package.py in clean-room mode.",
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
