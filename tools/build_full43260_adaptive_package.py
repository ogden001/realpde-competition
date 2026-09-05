#!/usr/bin/env python3
"""Build exactly one inference-only full@43260 adaptive package."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

import torch

from full43260_base_adaptive_runtime import AdaptiveUncertaintyHead


EXPECTED_CHECKPOINT_SHA = "50b692e236d5df9285a5cee976a51e3457a7eeed0f87d55b6568745077645d71"
EXPECTED_PROBE_SHA = "be027b9bfb6431522e0a51586102e7d98a34acc6aefa7412e2aa52a294821db0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable_config(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict) or "dx" not in raw or "dy" not in raw:
        raise ValueError("full checkpoint lacks frozen P0-A feature_config")
    return {
        "include_p0_a": bool(raw.get("include_p0_a", True)),
        "include_p0_b": bool(raw.get("include_p0_b", False)),
        "dx": float(raw["dx"]), "dy": float(raw["dy"]),
        "dt": None if raw.get("dt") is None else float(raw["dt"]),
        "re_center": float(raw.get("re_center", 0.0)), "re_scale": float(raw.get("re_scale", 1.0)),
    }


def _submission_source() -> str:
    return '''from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "_vendor"))
from rpde_baselines.model.cno import CNO3d
from realpde_p0_features import P0FeatureBuilder, P0FeatureConfig
from adaptive_runtime import AdaptiveUncertaintyHead, adaptive_bounds, flow_features

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODELS = None

def _load_models():
    global _MODELS
    if _MODELS is None:
        payload = torch.load(_ROOT / "model.pth", map_location="cpu", weights_only=False)
        config = P0FeatureConfig(**payload["feature_config"])
        builder = P0FeatureBuilder(config).to(_DEVICE)
        backbone = CNO3d(in_dim=len(builder.feature_names), out_dim=3, out_dim_mult=1, in_size=64, N_layers=3).to(_DEVICE)
        backbone.load_state_dict(payload["backbone_state_dict"], strict=True)
        head = AdaptiveUncertaintyHead(hidden=32, blocks=2).to(_DEVICE)
        head.load_state_dict(payload["base_head_state_dict"], strict=True)
        backbone.eval(); head.eval()
        _MODELS = payload, builder, backbone, head
    return _MODELS

def predict(input_array, metadata=None):
    x = np.asarray(input_array, dtype=np.float32)
    if x.ndim != 5 or x.shape[1:] != (20, 32, 64, 3):
        raise ValueError("input_array must have shape (N,20,32,64,3)")
    if x.shape[0] < 1 or not np.isfinite(x).all():
        raise ValueError("input_array must contain at least one finite sample")
    payload, builder, backbone, head = _load_models()
    with torch.inference_mode():
        tensor = torch.from_numpy(np.ascontiguousarray(x)).to(_DEVICE)
        prediction = backbone(builder(tensor).permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        prediction[..., 2] = 0.0
        uncertainty_input = torch.cat([tensor, flow_features(prediction[..., :2])], dim=-1)
        sigma_uv = head(uncertainty_input.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        lower, upper = adaptive_bounds(prediction, sigma_uv, floor=float(payload["bound_floor"]), mult=float(payload["bound_mult"]))
    return {"prediction": prediction.cpu().numpy().astype(np.float32),
            "lower": lower.cpu().numpy().astype(np.float32),
            "upper": upper.cpu().numpy().astype(np.float32)}
'''


def build(*, checkpoint: Path, probe: Path, kit_root: Path, out_root: Path, execution_commit: str) -> dict[str, object]:
    if out_root.exists():
        raise FileExistsError(f"refusing to overwrite OUT_ROOT: {out_root}")
    checkpoint_sha, probe_sha = sha256(checkpoint), sha256(probe)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA:
        raise ValueError(f"checkpoint SHA mismatch: {checkpoint_sha}")
    if probe_sha != EXPECTED_PROBE_SHA:
        raise ValueError(f"probe SHA mismatch: {probe_sha}")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if checkpoint_payload.get("iteration") != 43260 or checkpoint_payload.get("feature_set") != "P0-A":
        raise ValueError("checkpoint is not the required full@43260 P0-A artifact")
    config = _jsonable_config(checkpoint_payload.get("feature_config"))
    probe_payload = torch.load(probe, map_location="cpu", weights_only=False)
    head_state = probe_payload.get("base_head_state_dict")
    if not isinstance(head_state, dict) or not head_state:
        raise ValueError("probe lacks base_head_state_dict")
    head = AdaptiveUncertaintyHead(hidden=32, blocks=2)
    head.load_state_dict(head_state, strict=True)
    staging = out_root / "staging"
    staging.mkdir(parents=True)
    shutil.copy2(Path(__file__).with_name("full43260_base_adaptive_runtime.py"), staging / "adaptive_runtime.py")
    shutil.copy2(Path(__file__).with_name("realpde_p0_features.py"), staging / "realpde_p0_features.py")
    shutil.copytree(kit_root / "rpde_baselines", staging / "rpde_baselines")
    shutil.copytree(kit_root / "_vendor", staging / "_vendor")
    payload = {
        "backbone_state_dict": {key: value.detach().cpu() for key, value in checkpoint_payload["model_state_dict"].items()},
        "base_head_state_dict": {key: value.detach().cpu() for key, value in head_state.items()},
        "iteration": 43260, "feature_set": "P0-A", "feature_config": config,
        "bound_floor": 0.0025, "bound_mult": 1.0,
        "checkpoint_sha256": checkpoint_sha, "probe_sha256": probe_sha,
    }
    torch.save(payload, staging / "model.pth")
    (staging / "submission.py").write_text(_submission_source(), encoding="utf-8")
    inventory = [{"path": str(path.relative_to(staging)), "bytes": path.stat().st_size}
                 for path in sorted(staging.rglob("*")) if path.is_file()]
    zip_path = out_root / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in inventory:
            archive.write(staging / item["path"], item["path"])
    report = {
        "status": "PACKAGED", "execution_commit": execution_commit,
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "probe": str(probe), "probe_sha256": probe_sha,
        "base_head": "base_head_state_dict@1400", "feature_config": config,
        "bounds": {"floor": 0.0025, "mult": 1.0},
        "inventory": inventory, "zip": str(zip_path),
        "zip_bytes": zip_path.stat().st_size, "zip_sha256": sha256(zip_path),
        "created_at": time.time(),
    }
    (out_root / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    args = parser.parse_args()
    build(checkpoint=args.checkpoint, probe=args.probe, kit_root=args.kit_root,
          out_root=args.out_root, execution_commit=args.execution_commit)


if __name__ == "__main__":
    main()
