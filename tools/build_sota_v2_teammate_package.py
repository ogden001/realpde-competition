#!/usr/bin/env python3
"""Build SOTA-V2 with the teammate-style 35-channel SPS uncertainty head."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

import torch

from sps_teammate_uncertainty_runtime import TeammateUncertaintyHead

EXPECTED_FULL_CHECKPOINT_SHA = "f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce"
EXPECTED_FULL_ITERATION = 53_582
EXPECTED_FULL_CANONICAL_WINDOWS = 3_383
EXPECTED_FULL_TRAJECTORIES = 82
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
        raise ValueError("MF checkpoint state must contain only cno.* keys")
    return {name.removeprefix("cno."): value for name, value in state.items()}


def validate_full_checkpoint_metadata(
    payload: dict,
    actual_sha: str,
    expected_sha: str = EXPECTED_FULL_CHECKPOINT_SHA,
) -> None:
    if actual_sha != expected_sha:
        raise ValueError(f"full checkpoint SHA mismatch: {actual_sha}")
    if payload.get("iteration") != EXPECTED_FULL_ITERATION:
        raise ValueError(f"full checkpoint iteration must be {EXPECTED_FULL_ITERATION}")
    if payload.get("feature_set") != "P0-A":
        raise ValueError("full checkpoint feature_set must be P0-A")
    strip_mf_prefix(payload.get("model_state_dict", {}))


def validate_teammate_head_provenance(meta: dict, *, full_sha: str) -> str:
    if not isinstance(meta, dict):
        raise ValueError("head checkpoint lacks metadata")
    scope = meta.get("head_scope")
    common = {
        "backbone_checkpoint_iteration": EXPECTED_FULL_ITERATION,
        "backbone_checkpoint_sha256": full_sha,
        "window_mode": "fixed",
    }
    if scope == "full_specific_teammate35":
        expected = {
            **common,
            "train_windows": EXPECTED_FULL_CANONICAL_WINDOWS,
            "train_trajectories": EXPECTED_FULL_TRAJECTORIES,
            "recipe": meta.get("recipe", "teammate35"),
        }
    elif scope == "full53582_train50_teammate35_exact":
        expected = {
            **common,
            "train_windows": 2052,
            "train_trajectories": 50,
            "dev_windows": 659,
            "dev_trajectories": 16,
            "recipe": "teammate35_exact",
            "loss": "masked_gaussian_nll_nonzero_uv",
            "seed": 41,
            "max_updates": 2000,
            "eval_interval": 200,
        }
    else:
        raise ValueError("wrong teammate head scope")

    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"teammate head {key} mismatch")
    updates = meta.get("selected_updates")
    if not isinstance(updates, int) or not 200 <= updates <= 2000 or updates % 200:
        raise ValueError("selected_updates must be a 200-step checkpoint in [200,2000]")
    return str(scope)


def validate_teammate_calibration(calibration: dict) -> tuple[float, float, str]:
    if not isinstance(calibration, dict):
        raise ValueError("teammate calibration must be a dict")
    gate = calibration.get("gate")
    if gate not in {"SPS_TEAMMATE_GO", "SPS_TEAMMATE_NO_GO", "SPS_TEAMMATE_EXACT_READY"}:
        raise ValueError("teammate calibration gate is invalid")
    best = calibration.get("best", {})
    floor, mult = float(best["floor"]), float(best["mult"])
    if (floor, mult) not in ALLOWED_BOUNDS:
        raise ValueError("calibration bounds are outside frozen 28-row grid")
    return floor, mult, str(gate)


def _jsonable_config(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict) or "dx" not in raw or "dy" not in raw:
        raise ValueError("checkpoint lacks P0-A feature_config")
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
    return '''from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "_vendor"))
from rpde_baselines.model.cno import CNO3d
from realpde_p0_features import P0FeatureBuilder, P0FeatureConfig
from adaptive_runtime import mf_forward
from teammate_runtime import TeammateUncertaintyHead, sigma_from_log_std

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
        head = TeammateUncertaintyHead(hidden=32, blocks=2, dropout=0.0, include_pressure=True).to(_DEVICE)
        head.load_state_dict(payload["head_state_dict"], strict=True)
        cno.eval(); head.eval()
        _MODELS = payload, builder, cno, head
    return _MODELS

def predict(input_array, metadata=None):
    x = np.asarray(input_array, dtype=np.float32)
    if x.ndim != 5 or x.shape[1:] != (20, 32, 64, 3):
        raise ValueError("input_array must have shape (N,20,32,64,3)")
    if x.shape[0] < 1 or not np.isfinite(x).all():
        raise ValueError("input_array must contain finite samples")
    payload, builder, cno, head = _load_models()
    with torch.inference_mode():
        tensor = torch.from_numpy(np.ascontiguousarray(x)).to(_DEVICE)
        prediction = mf_forward(cno, builder, tensor)
        sigma = sigma_from_log_std(head(tensor, prediction))
        half_uv = float(payload["bound_floor"]) + float(payload["bound_mult"]) * sigma
        half = torch.cat([half_uv, torch.zeros_like(half_uv[..., :1])], dim=-1)
        lower, upper = prediction - half, prediction + half
    return {"prediction": prediction.cpu().numpy().astype(np.float32),
            "lower": lower.cpu().numpy().astype(np.float32),
            "upper": upper.cpu().numpy().astype(np.float32)}
'''


def build(
    *,
    full_checkpoint: Path,
    head_checkpoint: Path,
    calibration_summary: Path,
    kit_root: Path,
    out_root: Path,
    execution_commit: str,
) -> dict[str, object]:
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
    head = TeammateUncertaintyHead()
    head.load_state_dict(head_state, strict=True)
    head_scope = validate_teammate_head_provenance(meta, full_sha=full_sha)
    if _jsonable_config(meta.get("feature_config")) != full_config:
        raise ValueError("head/full feature_config mismatch")

    calibration = json.loads(calibration_summary.read_text(encoding="utf-8"))
    floor, mult, calibration_gate = validate_teammate_calibration(calibration)
    submission_recommended = calibration_gate in {"SPS_TEAMMATE_GO", "SPS_TEAMMATE_EXACT_READY"}

    staging = out_root / "staging"
    staging.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    shutil.copy2(here / "sota_v2_adaptive_runtime.py", staging / "adaptive_runtime.py")
    shutil.copy2(here / "sps_teammate_uncertainty_runtime.py", staging / "teammate_runtime.py")
    shutil.copy2(here / "realpde_p0_features.py", staging / "realpde_p0_features.py")
    shutil.copytree(kit_root / "rpde_baselines", staging / "rpde_baselines")
    shutil.copytree(kit_root / "_vendor", staging / "_vendor")
    torch.save(
        {
            "cno_state_dict": {
                key: value.detach().cpu()
                for key, value in strip_mf_prefix(full_payload["model_state_dict"]).items()
            },
            "head_state_dict": {key: value.detach().cpu() for key, value in head_state.items()},
            "feature_config": full_config,
            "iteration": EXPECTED_FULL_ITERATION,
            "bound_floor": floor,
            "bound_mult": mult,
            "full_checkpoint_sha256": full_sha,
            "head_checkpoint_sha256": head_sha,
            "head_scope": head_scope,
            "head_backbone_checkpoint_sha256": meta.get("backbone_checkpoint_sha256"),
            "calibration_gate": calibration_gate,
            "submission_recommended": submission_recommended,
            "recipe": "teammate35",
        },
        staging / "model.pth",
    )
    (staging / "submission.py").write_text(submission_source(), encoding="utf-8")
    inventory = [
        {"path": str(path.relative_to(staging)), "bytes": path.stat().st_size}
        for path in sorted(staging.rglob("*"))
        if path.is_file()
    ]
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
        "calibration_gate": calibration_gate,
        "submission_recommended": submission_recommended,
        "bounds": {"floor": floor, "mult": mult},
        "zip": str(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": sha256(zip_path),
        "inventory": inventory,
        "created_at": time.time(),
    }
    (out_root / "package_build.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    args = parser.parse_args()
    build(
        full_checkpoint=args.full_checkpoint,
        head_checkpoint=args.head_checkpoint,
        calibration_summary=args.calibration_summary,
        kit_root=args.kit_root,
        out_root=args.out_root,
        execution_commit=args.execution_commit,
    )


if __name__ == "__main__":
    main()
