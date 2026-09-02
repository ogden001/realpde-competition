#!/usr/bin/env python3
"""Build a self-contained offline Track 1 P0-A + N2 submission package."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import textwrap
import zipfile
from pathlib import Path

import torch


REQUIRED_ITERATION = 15_300
N2_WEIGHTS = {"mse": 1.0, "tke": 0.05, "rel": 0.027514, "mvpe": 0.009757}
MAX_ZIP_BYTES = 256 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint_for_submission(checkpoint: Path, *, required_iteration: int = REQUIRED_ITERATION) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("iteration") != required_iteration:
        raise ValueError(f"checkpoint iteration must be {required_iteration}, got {payload.get('iteration')!r}")
    if payload.get("feature_set") != "P0-A":
        raise ValueError("checkpoint is not marked P0-A")
    if payload.get("loss_weights") != N2_WEIGHTS:
        raise ValueError("checkpoint does not carry the frozen N2 weights")
    if not isinstance(payload.get("model_state_dict"), dict):
        raise ValueError("checkpoint lacks model_state_dict")
    config = payload.get("feature_config")
    if not isinstance(config, dict) or not all(key in config for key in ("dx", "dy")):
        raise ValueError("checkpoint lacks P0-A spacing")
    return payload


def package_file_inventory(root: Path) -> list[dict[str, int | str]]:
    allowed_exact = {"submission.py", "model.pth"}
    entries: list[dict[str, int | str]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        allowed_source = ((relative.startswith("rpde_baselines/") or relative.startswith("_vendor/einops/"))
                          and path.suffix == ".py")
        if relative not in allowed_exact and not allowed_source:
            raise ValueError(f"package inventory forbids {relative}")
        if "optimizer" in relative.lower() or "log" in relative.lower() or path.suffix in {".h5", ".jsonl"}:
            raise ValueError(f"package inventory forbids training artifact {relative}")
        entries.append({"path": relative, "bytes": path.stat().st_size})
    required = allowed_exact - {entry["path"] for entry in entries}
    if required:
        raise ValueError(f"package inventory missing {sorted(required)}")
    return entries


def submission_source() -> str:
    """Return the package-local inference module; P0-A math matches the trainer."""
    return textwrap.dedent('''\
        from __future__ import annotations

        from pathlib import Path
        import sys
        import numpy as np
        import torch

        _ROOT = Path(__file__).resolve().parent
        _MODEL = None
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _derivative(value, dimension, spacing):
            result = torch.empty_like(value)
            first = [slice(None)] * value.ndim; second = [slice(None)] * value.ndim
            first[dimension], second[dimension] = 0, 1
            result[tuple(first)] = (value[tuple(second)] - value[tuple(first)]) / spacing
            last = [slice(None)] * value.ndim; before = [slice(None)] * value.ndim
            last[dimension], before[dimension] = -1, -2
            result[tuple(last)] = (value[tuple(last)] - value[tuple(before)]) / spacing
            middle = [slice(None)] * value.ndim; right = [slice(None)] * value.ndim; left = [slice(None)] * value.ndim
            middle[dimension], right[dimension], left[dimension] = slice(1, -1), slice(2, None), slice(None, -2)
            result[tuple(middle)] = (value[tuple(right)] - value[tuple(left)]) / (2.0 * spacing)
            return result

        def _p0a(input_window, dx, dy):
            raw = input_window[..., :3]
            u, v = raw[..., 0], raw[..., 1]
            du_dx = _derivative(u, -1, dx); du_dy = _derivative(u, -2, dy)
            dv_dx = _derivative(v, -1, dx); dv_dy = _derivative(v, -2, dy)
            vorticity = dv_dx - du_dy
            strain = torch.sqrt(du_dx.square() + dv_dy.square() + 0.5 * (du_dy + dv_dx).square())
            delta_u, delta_v = torch.zeros_like(u), torch.zeros_like(v)
            delta_u[:, 1:] = u[:, 1:] - u[:, :-1]; delta_v[:, 1:] = v[:, 1:] - v[:, :-1]
            u_mean, v_mean = u.mean(dim=1, keepdim=True), v.mean(dim=1, keepdim=True)
            u_std = torch.sqrt(u.var(dim=1, keepdim=True, unbiased=False)); v_std = torch.sqrt(v.var(dim=1, keepdim=True, unbiased=False))
            history_tke = 0.5 * (u_std.square() + v_std.square())
            broadcast = lambda value: value.expand_as(u)
            extras = (torch.sqrt(u.square() + v.square()), du_dx, du_dy, dv_dx, dv_dy,
                      vorticity, vorticity.abs(), strain, delta_u, delta_v,
                      broadcast(u_mean), broadcast(v_mean), broadcast(u_std), broadcast(v_std),
                      u - broadcast(u_mean), v - broadcast(v_mean), broadcast(history_tke))
            return torch.cat([raw, *(item.unsqueeze(-1) for item in extras)], dim=-1)

        def _load_model():
            global _MODEL
            if _MODEL is None:
                sys.path.insert(0, str(_ROOT))
                sys.path.insert(0, str(_ROOT / "_vendor"))
                from rpde_baselines.model.cno import CNO3d
                payload = torch.load(_ROOT / "model.pth", map_location="cpu", weights_only=False)
                model = CNO3d(in_dim=20, out_dim=3, out_dim_mult=1, in_size=64, N_layers=3).to(_DEVICE)
                model.load_state_dict(payload["model_state_dict"], strict=True)
                model.eval()
                _MODEL = (model, float(payload["feature_config"]["dx"]), float(payload["feature_config"]["dy"]))
            return _MODEL

        def predict(input_array, metadata=None):
            x = np.asarray(input_array, dtype=np.float32)
            if x.ndim != 5 or x.shape[1:] != (20, 32, 64, 3):
                raise ValueError("input_array must have shape (N,20,32,64,3)")
            if x.shape[0] < 1 or not np.isfinite(x).all():
                raise ValueError("input_array must contain at least one finite sample")
            model, dx, dy = _load_model()
            with torch.inference_mode():
                tensor = torch.from_numpy(np.ascontiguousarray(x)).to(_DEVICE)
                prediction = model(_p0a(tensor, dx, dy).permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
                prediction[..., 2] = 0.0
            return prediction.cpu().numpy().astype(np.float32, copy=False)
    ''')


def generate_submission_module(root: Path) -> None:
    (root / "submission.py").write_text(submission_source(), encoding="utf-8")


def build_submission(*, checkpoint: Path, kit_root: Path, out_dir: Path, experiment_id: str, git_commit: str) -> dict:
    if out_dir.exists():
        raise FileExistsError(out_dir)
    payload = validate_checkpoint_for_submission(checkpoint)
    source_cno = kit_root / "rpde_baselines"
    source_einops = kit_root / "_vendor" / "einops"
    if not source_cno.is_dir() or not source_einops.is_dir():
        raise FileNotFoundError("official kit lacks vendored CNO dependencies")
    out_dir.mkdir(parents=True)
    staging = out_dir / "staging"
    staging.mkdir()
    generate_submission_module(staging)
    minimal = {"model_state_dict": payload["model_state_dict"], "iteration": payload["iteration"],
               "feature_set": payload["feature_set"], "feature_config": payload["feature_config"],
               "loss_weights": payload["loss_weights"]}
    torch.save(minimal, staging / "model.pth")
    shutil.copytree(source_cno, staging / "rpde_baselines")
    shutil.copytree(source_einops, staging / "_vendor" / "einops")
    inventory = package_file_inventory(staging)
    zip_path = out_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for entry in inventory:
            archive.write(staging / str(entry["path"]), arcname=str(entry["path"]))
    if zip_path.stat().st_size >= MAX_ZIP_BYTES:
        raise ValueError(f"ZIP is {zip_path.stat().st_size} bytes, must be below {MAX_ZIP_BYTES}")
    report = {"experiment_id": experiment_id, "git_commit": git_commit, "checkpoint": str(checkpoint),
              "checkpoint_sha256": sha256(checkpoint), "checkpoint_iteration": payload["iteration"],
              "zip": str(zip_path), "zip_sha256": sha256(zip_path), "zip_bytes": zip_path.stat().st_size,
              "inventory": inventory, "official_template": str(kit_root / "submission_template.py")}
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "submission.zip.sha256").write_text(f"{report['zip_sha256']}  submission.zip\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(build_submission(checkpoint=args.checkpoint, kit_root=args.kit_root, out_dir=args.out_dir,
                                       experiment_id=args.experiment_id, git_commit=args.git_commit), indent=2))


if __name__ == "__main__":
    main()
