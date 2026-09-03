#!/usr/bin/env python3
"""Lightweight runtime and artifact context for RealPDE remote execution.

The tool does not make research decisions. It inventories machine facts,
released data, scorer identity, and checkpoint resume semantics so Sol can
design tasks against the actual GPU environment instead of guessed assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import time
from pathlib import Path
from typing import Iterable

import h5py
import torch


SCHEMA_VERSION = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def inventory_data(data_root: Path) -> dict:
    paths = sorted(data_root.glob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"no .h5 trajectories under {data_root}")
    windows = 0
    shapes: set[tuple[int, int]] = set()
    for path in paths:
        with h5py.File(path, "r") as handle:
            if "u" not in handle or "v" not in handle:
                raise ValueError(f"{path} lacks top-level u/v")
            frames, height, width = handle["u"].shape
            if handle["v"].shape != (frames, height, width):
                raise ValueError(f"{path} u/v shapes differ")
            shapes.add((height, width))
            if frames >= 40:
                windows += len(range(0, frames - 40 + 1, 20))
    return {
        "root": str(data_root.resolve()),
        "trajectories": len(paths),
        "windows": windows,
        "spatial_shapes": [list(shape) for shape in sorted(shapes)],
    }


def _has_optimizer_state(payload: dict) -> bool:
    optimizer = payload.get("optimizer_state_dict")
    return isinstance(optimizer, dict) and bool(optimizer.get("param_groups"))


def inspect_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        payload = {}
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return {
        "name": path.name,
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "iteration": payload.get("iteration"),
        "feature_set": payload.get("feature_set"),
        "feature_config": payload.get("feature_config"),
        "loss_weights": payload.get("loss_weights"),
        "optimizer_state": _has_optimizer_state(payload),
        "run": metadata.get("run"),
    }


def scan_checkpoints(paths: Iterable[Path]) -> list[dict]:
    unique: dict[str, Path] = {}
    for path in paths:
        if path.is_file() and path.suffix == ".pth":
            unique[str(path.resolve())] = path
        elif path.is_dir():
            for checkpoint in path.rglob("*.pth"):
                unique[str(checkpoint.resolve())] = checkpoint
    return [inspect_checkpoint(unique[key]) for key in sorted(unique)]


def runtime_host() -> dict:
    result = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        result.update(
            {
                "cuda_runtime": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            }
        )
    return result


def build_snapshot(
    *,
    data_root: Path | None,
    kit_root: Path | None,
    checkpoint_paths: Iterable[Path],
    artifact_roots: Iterable[Path],
) -> dict:
    scan_targets = [*checkpoint_paths, *artifact_roots]
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.time(),
        "host": runtime_host(),
        "data": None if data_root is None else inventory_data(data_root),
        "kit": None,
        "artifacts": scan_checkpoints(scan_targets),
    }
    if kit_root is not None:
        scorer = kit_root / "scoring.py"
        result["kit"] = {
            "root": str(kit_root.resolve()),
            "scorer_sha256": sha256(scorer) if scorer.is_file() else None,
        }
    return result


def generate_artifact_manifest(run_dir: Path) -> dict:
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    checkpoints = scan_checkpoints([run_dir])
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.time(),
        "run_dir": str(run_dir.resolve()),
        "metadata": _read_json(run_dir / "run_metadata.json"),
        "status": _read_json(run_dir / "status.json"),
        "checkpoints": checkpoints,
        "resumable_checkpoints": [row for row in checkpoints if row["optimizer_state"]],
    }


def resolve_checkpoint(
    snapshot: dict,
    *,
    iteration: int | None = None,
    feature_set: str | None = None,
    require_optimizer_state: bool = False,
    run: str | None = None,
) -> dict:
    candidates = []
    for row in snapshot.get("artifacts", []):
        if iteration is not None and row.get("iteration") != iteration:
            continue
        if feature_set is not None and row.get("feature_set") != feature_set:
            continue
        if require_optimizer_state and not row.get("optimizer_state"):
            continue
        if run is not None and row.get("run") != run:
            continue
        candidates.append(row)
    if not candidates:
        available = [
            {
                "iteration": row.get("iteration"),
                "feature_set": row.get("feature_set"),
                "optimizer_state": row.get("optimizer_state"),
                "run": row.get("run"),
                "path": row.get("path"),
            }
            for row in snapshot.get("artifacts", [])
        ]
        raise ValueError(f"no checkpoint matches requested semantics; available={available}")
    if len(candidates) != 1:
        raise ValueError(f"checkpoint semantics are ambiguous; matches={[row.get('path') for row in candidates]}")
    return candidates[0]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--data-root", type=Path)
    snapshot.add_argument("--kit-root", type=Path)
    snapshot.add_argument("--checkpoint", type=Path, action="append", default=[])
    snapshot.add_argument("--artifact-root", type=Path, action="append", default=[])
    snapshot.add_argument("--output", type=Path, required=True)

    manifest = sub.add_parser("manifest")
    manifest.add_argument("--run-dir", type=Path, required=True)
    manifest.add_argument("--output", type=Path)

    resolve = sub.add_parser("resolve")
    resolve.add_argument("--snapshot", type=Path, required=True)
    resolve.add_argument("--iteration", type=int)
    resolve.add_argument("--feature-set")
    resolve.add_argument("--run")
    resolve.add_argument("--require-optimizer-state", action="store_true")

    args = parser.parse_args()
    if args.command == "snapshot":
        result = build_snapshot(
            data_root=args.data_root,
            kit_root=args.kit_root,
            checkpoint_paths=args.checkpoint,
            artifact_roots=args.artifact_root,
        )
        write_json(args.output, result)
    elif args.command == "manifest":
        result = generate_artifact_manifest(args.run_dir)
        output = args.output or (args.run_dir / "artifact_manifest.json")
        write_json(output, result)
    else:
        snapshot_value = json.loads(args.snapshot.read_text(encoding="utf-8"))
        result = resolve_checkpoint(
            snapshot_value,
            iteration=args.iteration,
            feature_set=args.feature_set,
            require_optimizer_state=args.require_optimizer_state,
            run=args.run,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
