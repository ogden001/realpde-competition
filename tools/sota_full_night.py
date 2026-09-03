#!/usr/bin/env python3
"""Preflight and summarize the 2026-09-04 overnight full-data SOTA continuation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import torch


EXPECTED_UPDATE = 15300
EXPECTED_CHECKPOINT_SHA256 = "3ea4e4def03ae2f1d970975e4217358e1d762b88a69bdddfdf844d551baaa3e4"
EXPECTED_TRAJECTORIES = 82
EXPECTED_WINDOWS = 3383
EXPECTED_N2 = {"mse": 1.0, "tke": 0.05, "rel": 0.027514, "mvpe": 0.009757}
MILESTONES = (31100, 36500, 37850, 40560, 43260)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_resume_payload(payload: dict) -> None:
    if payload.get("iteration") != EXPECTED_UPDATE:
        raise ValueError(f"resume checkpoint must be update {EXPECTED_UPDATE}")
    if payload.get("feature_set") != "P0-A":
        raise ValueError("resume checkpoint must be P0-A")
    if payload.get("loss_weights") != EXPECTED_N2:
        raise ValueError("resume checkpoint N2 schema differs from the frozen four-key N2")
    optimizer = payload.get("optimizer_state_dict")
    if not isinstance(optimizer, dict) or not optimizer.get("param_groups"):
        raise ValueError("resume checkpoint lacks optimizer state")


def count_windows(path: Path) -> int:
    with h5py.File(path, "r") as handle:
        if "u" not in handle or "v" not in handle:
            raise ValueError(f"{path} lacks top-level u/v")
        frames, height, width = handle["u"].shape
        if handle["v"].shape != (frames, height, width):
            raise ValueError(f"{path} u/v shapes differ")
        if (height, width) != (64, 128):
            raise ValueError(f"{path} unexpected spatial shape {(height, width)}")
        if frames < 40:
            return 0
        return len(range(0, frames - 40 + 1, 20))


def release_inventory(data_root: Path) -> dict:
    paths = sorted(data_root.glob("*.h5"))
    windows = sum(count_windows(path) for path in paths)
    result = {"trajectories": len(paths), "windows": windows}
    if result["trajectories"] != EXPECTED_TRAJECTORIES:
        raise ValueError(
            f"released-data trajectory count {result['trajectories']} != expected {EXPECTED_TRAJECTORIES}"
        )
    if result["windows"] != EXPECTED_WINDOWS:
        raise ValueError(f"released-data window count {result['windows']} != expected {EXPECTED_WINDOWS}")
    return result


def preflight(*, data_root: Path, kit_root: Path, resume_checkpoint: Path) -> dict:
    if not resume_checkpoint.is_file():
        raise FileNotFoundError(resume_checkpoint)
    if not (kit_root / "scoring.py").is_file():
        raise FileNotFoundError(kit_root / "scoring.py")
    checkpoint_sha = sha256(resume_checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"resume checkpoint SHA256 {checkpoint_sha} != expected {EXPECTED_CHECKPOINT_SHA256}"
        )
    payload = torch.load(resume_checkpoint, map_location="cpu")
    validate_resume_payload(payload)
    inventory = release_inventory(data_root)
    return {
        "state": "PREFLIGHT_OK",
        "resume_update": EXPECTED_UPDATE,
        "resume_checkpoint_sha256": checkpoint_sha,
        "inventory": inventory,
        "target_update": 43260,
        "milestones": list(MILESTONES),
        "lr": 1e-5,
        "hard_train_seconds": 21300,
    }


def summarize_run(run_dir: Path, *, milestones: tuple[int, ...] = MILESTONES) -> dict:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if status.get("state") not in {"DONE", "TIME_CAPPED"}:
        raise ValueError(f"run is not in an accepted terminal state: {status.get('state')}")
    if metadata.get("initial_update") != EXPECTED_UPDATE:
        raise ValueError("run did not start from update 15300")
    if metadata.get("train_trajectories") != EXPECTED_TRAJECTORIES or metadata.get("train_windows") != EXPECTED_WINDOWS:
        raise ValueError("run was not the frozen 82-trajectory / 3383-window full-data continuation")
    reached = [update for update in milestones if (run_dir / f"model_update_{update:05d}.pth").is_file()]
    missing = [update for update in milestones if update not in reached]
    if not (run_dir / "model_last.pth").is_file():
        raise FileNotFoundError(run_dir / "model_last.pth")
    return {
        "decision": "REVIEW_REQUIRED",
        "state": status["state"],
        "stop_reason": status.get("stop_reason"),
        "completed_update": int(status["update"]),
        "reached_milestones": reached,
        "missing_milestones": missing,
        "model_last": str(run_dir / "model_last.pth"),
        "target_update": 43260,
        "lr": metadata.get("lr"),
        "train_trajectories": metadata.get("train_trajectories"),
        "train_windows": metadata.get("train_windows"),
        "elapsed_seconds": status.get("elapsed_seconds"),
        "peak_gpu_bytes": status.get("peak_gpu_bytes"),
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("preflight")
    pre.add_argument("--data-root", type=Path, required=True)
    pre.add_argument("--kit-root", type=Path, required=True)
    pre.add_argument("--resume-checkpoint", type=Path, required=True)
    pre.add_argument("--output", type=Path, required=True)

    summary = sub.add_parser("summarize")
    summary.add_argument("--run-dir", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(
            data_root=args.data_root,
            kit_root=args.kit_root,
            resume_checkpoint=args.resume_checkpoint,
        )
    else:
        result = summarize_run(args.run_dir)
    write_json(args.output, result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
