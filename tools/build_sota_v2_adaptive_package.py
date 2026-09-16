#!/usr/bin/env python3
"""Build the frozen SOTA-V2 MF-CNO + adaptive uncertainty submission package."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

import torch

from sota_v2_adaptive_runtime import AdaptiveUncertaintyHead

EXPECTED_FULL_CHECKPOINT_SHA = "f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce"
EXPECTED_FULL_ITERATION = 53_582
EXPECTED_FULL_DENSE_WINDOWS = 66_755
EXPECTED_FULL_TRAJECTORIES = 82
MAX_ZIP_BYTES = 256 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strip_mf_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state or not all(name.startswith("cno.") for name in state):
        raise ValueError("MF checkpoint state must contain only cno.* keys")
    return {name.removeprefix("cno."): value for name, value in state.items()}


def validate_full_checkpoint_metadata(payload: dict, actual_sha: str, expected_sha: str = EXPECTED_FULL_CHECKPOINT_SHA) -> None:
    if actual_sha != expected_sha:
        raise ValueError(f"full checkpoint SHA mismatch: {actual_sha}")
    if payload.get("iteration") != EXPECTED_FULL_ITERATION:
        raise ValueError(f"full checkpoint iteration must be {EXPECTED_FULL_ITERATION}")
    if payload.get("feature_set") != "P0-A":
        raise ValueError("full checkpoint feature_set must be P0-A")
    strip_mf_prefix(payload.get("model_state_dict", {}))


def validate_head_provenance(meta: dict, *, full_sha: str) -> str:
    """Validate either the historical validation head or a matched full head.

    Legacy packages used a head trained on the 50-train validation backbone and
    then moved that head onto the full backbone.  The stride=1 SPS replica adds
    a safer mode where the head is trained directly against residuals from the
    frozen full backbone.  Both modes remain supported so historical rebuilds
    stay reproducible.
    """
    if not isinstance(meta, dict):
        raise ValueError("head checkpoint lacks metadata")
    if meta.get("updates") != 1400:
        raise ValueError("head metadata updates must equal 1400")

    if meta.get("head_scope") == "full_specific":
        if meta.get("backbone_checkpoint_iteration") != EXPECTED_FULL_ITERATION:
            raise ValueError("full-specific head backbone iteration mismatch")
        if meta.get("backbone_checkpoint_sha256") != full_sha:
            raise ValueError("full-specific head backbone SHA mismatch")
        if meta.get("train_windows") != EXPECTED_FULL_DENSE_WINDOWS:
            raise ValueError("full-specific head dense window count mismatch")
        if meta.get("train_trajectories") != EXPECTED_FULL_TRAJECTORIES:
            raise ValueError("full-specific head trajectory count mismatch")
        if meta.get("window_mode") != "dense_all":
            raise ValueError("full-specific head must use dense_all windows")
        return "full_specific"

    if meta.get("validation_checkpoint_iteration") != 32500 or meta.get("train_windows") != 2052:
        raise ValueError("head metadata does not match frozen legacy validation protocol")
    return "legacy_validation"


def validate_calibration_gate(calibration: dict) -> str:
    gate = calibration.get("gate") if isinstance(calibration, dict) else None
    if gate not in {"ADAPTIVE_GO", "SPS_REPLICA_GO"}:
        raise ValueError("calibration gate is not GO")
    return str(gate)


def _jsonable_config(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict) or "dx" not in raw or "dy" not in raw:
        raise ValueError("checkpoint lacks P0-A feature_config")
    return {
        "include_p0_a": bool(raw.get("include_p0_a", True)),
        "include_p0_b": bool(raw.get("include_p0_b", False)),
        "dx": float(raw["dx"]), "dy": float(raw["dy"]),
        "dt": None if raw.get("dt") is None else float(raw["dt"]),
        "re_center": float(raw.get("re_center", 0.0)),
        "re_scale": float(raw.get("re_scale", 1.0)),
    }


def submission_source() -> str:
    return '''from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "_vendor"))
from rpde_baselines.model.cno import CNO3d
from realpde_p0_features import P0FeatureBuilder, P0FeatureConfig
from adaptive_runtime import AdaptiveUncertaintyHead, adaptive_bounds, flow_features, mf_forward

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODELS = None

def _load_models():
    global _MODELS
    if _MODELS is None:
        payload = torch.load(_ROOT / "model.pth", map_location="cpu", weights_only=False)
        config = P0FeatureConfig(**payload["feature_config"])
        builder = P0FeatureBuilder(config).to(_DEVICE)
        cno = CNO3d(in_dim=len(builder.feature_names), out_dim=5, out_dim_mult=1, in_size=64, N_layers=3).to(_DEVICE)
        cno.load_state_dict(payload["cno_state_dict"], strict=True)
        head = AdaptiveUncertaintyHead(in_channels=15, hidden=32, blocks=2).to(_DEVICE)
        head.load_state_dict(payload["head_state_dict"], strict=True)
        cno.eval(); head.eval()
        _MODELS = payload, builder, cno, head
    return _MODELS

def predict(input_array, metadata=None):
    x = np.asarray(input_array, dtype=np.float32)
    if x.ndim != 5 or x.shape[1:] != (20, 32, 64, 3):
        raise ValueError("input_array must have shape (N,20,32,64,3)")
    if x.shape[0] < 1 or not np.isfinite(x).all():
        raise ValueError("input_array must contain at least one finite sample")
    payload, builder, cno, head = _load_models()
    with torch.inference_mode():
        tensor = torch.from_numpy(np.ascontiguousarray(x)).to(_DEVICE)
        prediction = mf_forward(cno, builder, tensor)
        features = torch.cat([tensor, flow_features(prediction[..., :2])], dim=-1).permute(0,4,1,2,3)
        sigma = head(features).permute(0,2,3,4,1)
        lower, upper = adaptive_bounds(prediction, sigma, floor=float(payload["bound_floor"]), mult=float(payload["bound_mult"]))
    return {"prediction": prediction.cpu().numpy().astype(np.float32),
            "lower": lower.cpu().numpy().astype(np.float32),
            "upper": upper.cpu().numpy().astype(np.float32)}
'''


def build(*, full_checkpoint: Path, head_checkpoint: Path, calibration_summary: Path,
          kit_root: Path, out_root: Path, execution_commit: str) -> dict[str, object]:
    if out_root.exists():
        raise FileExistsError(out_root)
    full_sha = sha256(full_checkpoint)
    full_payload = torch.load(full_checkpoint, map_location="cpu", weights_only=False)
    validate_full_checkpoint_metadata(full_payload, full_sha)
    full_config = _jsonable_config(full_payload.get("feature_config"))

    head_sha = sha256(head_checkpoint)
    head_payload = torch.load(head_checkpoint, map_location="cpu", weights_only=False)
    head_state = head_payload.get("head_state_dict")
    meta = head_payload.get("metadata", {})
    if not isinstance(head_state, dict) or not head_state:
        raise ValueError("head checkpoint lacks head_state_dict")
    head = AdaptiveUncertaintyHead(); head.load_state_dict(head_state, strict=True)
    head_scope = validate_head_provenance(meta, full_sha=full_sha)
    head_config = _jsonable_config(meta.get("feature_config"))
    if head_config != full_config:
        raise ValueError("head/full feature_config mismatch")

    calibration = json.loads(calibration_summary.read_text(encoding="utf-8"))
    calibration_gate = validate_calibration_gate(calibration)
    best = calibration.get("best", {})
    floor, mult = float(best["floor"]), float(best["mult"])
    allowed = {(f, m) for f in (0.0, 0.0025, 0.005, 0.0075) for m in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)}
    if (floor, mult) not in allowed:
        raise ValueError("calibration bounds are outside frozen 28-row grid")

    staging = out_root / "staging"; staging.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    shutil.copy2(here / "sota_v2_adaptive_runtime.py", staging / "adaptive_runtime.py")
    shutil.copy2(here / "realpde_p0_features.py", staging / "realpde_p0_features.py")
    shutil.copytree(kit_root / "rpde_baselines", staging / "rpde_baselines")
    shutil.copytree(kit_root / "_vendor", staging / "_vendor")
    torch.save({
        "cno_state_dict": {k: v.detach().cpu() for k, v in strip_mf_prefix(full_payload["model_state_dict"]).items()},
        "head_state_dict": {k: v.detach().cpu() for k, v in head_state.items()},
        "feature_config": full_config,
        "iteration": EXPECTED_FULL_ITERATION,
        "bound_floor": floor,
        "bound_mult": mult,
        "full_checkpoint_sha256": full_sha,
        "head_checkpoint_sha256": head_sha,
        "head_scope": head_scope,
        "head_backbone_checkpoint_sha256": meta.get("backbone_checkpoint_sha256"),
        "validation_checkpoint_sha256": meta.get("validation_checkpoint_sha256"),
        "calibration_gate": calibration_gate,
    }, staging / "model.pth")
    (staging / "submission.py").write_text(submission_source(), encoding="utf-8")
    inventory = [{"path": str(path.relative_to(staging)), "bytes": path.stat().st_size}
                 for path in sorted(staging.rglob("*")) if path.is_file()]
    zip_path = out_root / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in inventory:
            archive.write(staging / item["path"], item["path"])
    if zip_path.stat().st_size >= MAX_ZIP_BYTES:
        raise ValueError("submission ZIP exceeds 256 MiB")
    report = {
        "status": "PACKAGED",
        "execution_commit": execution_commit,
        "full_checkpoint": str(full_checkpoint),
        "full_checkpoint_sha256": full_sha,
        "head_checkpoint": str(head_checkpoint),
        "head_checkpoint_sha256": head_sha,
        "head_scope": head_scope,
        "calibration_summary": str(calibration_summary),
        "calibration_gate": calibration_gate,
        "bounds": {"floor": floor, "mult": mult},
        "zip": str(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": sha256(zip_path),
        "inventory": inventory,
        "created_at": time.time(),
    }
    (out_root / "package_build.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--full-checkpoint", type=Path, required=True)
    p.add_argument("--head-checkpoint", type=Path, required=True)
    p.add_argument("--calibration-summary", type=Path, required=True)
    p.add_argument("--kit-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--execution-commit", required=True)
    a = p.parse_args()
    build(full_checkpoint=a.full_checkpoint, head_checkpoint=a.head_checkpoint,
          calibration_summary=a.calibration_summary, kit_root=a.kit_root,
          out_root=a.out_root, execution_commit=a.execution_commit)


if __name__ == "__main__":
    main()
