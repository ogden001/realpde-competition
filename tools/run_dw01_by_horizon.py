#!/usr/bin/env python3
"""Replay registered DW-01/RW-00 checkpoints for Future20 diagnostics only."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import realpde_loss_official_v9 as core
from dw01_by_horizon import aggregate_by_horizon, compute_window_horizon_metrics, write_csv, write_json
from realpde_b1_p0a_n2 import forward, load_model
from realpde_p0_data import H5WindowDataset, read_grid
from realpde_p0_features import P0FeatureBuilder, P0FeatureConfig


FRAME_FIELDS = ["experiment", "horizon", "frame_rel_l2", "frame_rmse", "tke_contrib_rel_l2", "tke_contrib_ratio", "mvpe_probe_rel_l2", "windows", "trajectories"]
WINDOW_FIELDS = ["experiment", "trajectory", "window_start", "horizon", "frame_rel_l2", "frame_rmse", "tke_contrib_rel_l2", "tke_contrib_ratio", "mvpe_probe_rel_l2"]
AGG_FIELDS = ["experiment", "update", "rel_l2", "tke", "mvpe", "windows", "trajectories", "parity_status"]


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_dev_paths(manifest: Path, data_root: Path) -> list[Path]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    paths = [data_root / row["file"] for row in payload["dev"]]
    if len(paths) != 16 or not all(path.is_file() for path in paths):
        raise RuntimeError("expected 16 readable frozen dev trajectories")
    return paths


def replay_checkpoint(checkpoint: Path, kit_root: Path, dataset: H5WindowDataset, paths: list[Path], batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    x_grid, y_grid = read_grid(paths[0], sub_sample=2)
    config = P0FeatureConfig(include_p0_a=True, include_p0_b=False,
                             dx=float(x_grid[0, 1] - x_grid[0, 0]), dy=float(y_grid[1, 0] - y_grid[0, 0]))
    builder = P0FeatureBuilder(config).to(device)
    model = load_model(kit_root, checkpoint, builder, device).eval()
    predictions, targets = [], []
    with torch.no_grad():
        for x, y, _, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            predictions.append(forward(model, builder, x.to(device)).cpu().numpy().astype(np.float32))
            targets.append(y.numpy().astype(np.float32))
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    if prediction.shape != (659, 20, 32, 64, 3):
        raise RuntimeError(f"unexpected replay shape {prediction.shape}")
    return prediction, target


def official_aggregate(kit_root: Path, prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    sys.path.insert(0, str(kit_root))
    import scoring
    channels = scoring.measured_channels(target)
    return {"rel_l2": float(np.mean(scoring.rel_l2_per_sample(prediction, target, channels))),
            "tke": float(np.mean(scoring.tke_rel_l2_per_sample(prediction, target, channels))),
            "mvpe": float(scoring.mvpe_rel_l2(prediction, target))}


def parity(aggregate: dict[str, float], summary: Path, update: int, tolerance: float) -> str:
    history = json.loads(summary.read_text(encoding="utf-8"))["history"]
    expected = next(row for row in history if int(row["iteration"]) == update)
    deltas = {key: abs(float(aggregate[key]) - float(expected[key])) for key in ("rel_l2", "tke", "mvpe")}
    return "PASS" if max(deltas.values()) <= tolerance else "INVALID_REPLAY"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--inventory-json", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--parity-tolerance", type=float, default=1e-5)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--parity-summary-root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    inventory = json.loads(args.inventory_json.read_text(encoding="utf-8"))
    paths = read_dev_paths(args.manifest, args.data_root)
    dataset = H5WindowDataset(paths, in_steps=20, out_steps=20, stride=20, sub_sample=2, window_mode="fixed")
    if len(dataset) != 659:
        raise RuntimeError(f"expected 659 dev windows, got {len(dataset)}")
    names = [ref.path.name for ref in dataset.refs]
    starts = [ref.start for ref in dataset.refs]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_frame, all_window, aggregate_rows, replay_records = [], [], [], []
    target_reference = None
    for item in inventory:
        checkpoint = args.checkpoint_root / item["checkpoint_relative"]
        record = {key: item.get(key) for key in ("experiment", "update", "checkpoint_relative", "checkpoint_sha256")}
        record["checkpoint_exists"] = checkpoint.is_file()
        if not checkpoint.is_file():
            record["replay_status"] = "MISSING_CHECKPOINT"
            replay_records.append(record)
            continue
        actual_sha = sha256(checkpoint)
        if actual_sha != item["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint SHA mismatch for {item['experiment']}: {actual_sha}")
        prediction, target = replay_checkpoint(checkpoint, args.kit_root, dataset, paths, args.batch_size, device)
        if target_reference is None:
            target_reference = target
        elif not np.array_equal(target_reference, target):
            raise RuntimeError("target replay changed across checkpoints")
        rows = compute_window_horizon_metrics(prediction, target, names, starts)
        experiment = item["experiment"]
        all_frame.extend({"experiment": experiment, **row} for row in aggregate_by_horizon(rows, experiment=experiment, trajectories=16))
        all_window.extend({"experiment": experiment, **row} for row in rows)
        aggregate = official_aggregate(args.kit_root, prediction, target)
        status = "NOT_APPLICABLE"
        summary_ref = item.get("parity_summary") or item.get("parity_summary_relative")
        if summary_ref:
            summary_path = Path(summary_ref)
            if not summary_path.is_absolute():
                summary_path = args.parity_summary_root / summary_path
            status = parity(aggregate, summary_path, int(item["update"]), args.parity_tolerance)
        aggregate_rows.append({"experiment": experiment, "update": int(item["update"]), **aggregate,
                               "windows": 659, "trajectories": 16, "parity_status": status})
        record.update({"replay_status": "REPLAYED", "checkpoint_sha256_verified": actual_sha, "prediction_shape": list(prediction.shape)})
        replay_records.append(record)
    write_csv(args.out_dir / "by_horizon.csv", all_frame, FRAME_FIELDS)
    write_csv(args.out_dir / "by_trajectory_horizon.csv", all_window, WINDOW_FIELDS)
    write_csv(args.out_dir / "aggregate_metrics.csv", aggregate_rows, AGG_FIELDS)
    inventory_fields = sorted({key for row in replay_records for key in row})
    write_csv(args.out_dir / "experiment_inventory.csv", replay_records, inventory_fields)
    overall_status = "PASS" if replay_records and all(r.get("replay_status") == "REPLAYED" for r in replay_records) and all(r.get("parity_status", "PASS") in ("PASS", "NOT_APPLICABLE") for r in aggregate_rows) else "REVIEW_REQUIRED"
    write_json(args.out_dir / "summary.json", {
        "status": overall_status, "prediction_only": True,
        "experiments": [r for r in aggregate_rows],
        "provenance": {"execution_commit": args.execution_commit, "manifest_sha256": sha256(args.manifest), "kit_scorer_sha256": sha256(args.kit_root / "scoring.py")},
        "dev_windows": len(dataset), "dev_trajectories": len(paths), "horizons": 20,
    })
    write_json(args.out_dir / "replay_metadata.json", {
        "execution_commit": args.execution_commit, "manifest_sha256": sha256(args.manifest),
        "kit_scorer_sha256": sha256(args.kit_root / "scoring.py"), "dev_windows": len(dataset),
        "dev_trajectories": len(paths), "device": str(device), "prediction_only": True,
        "checkpoint_count": len(inventory), "replay_count": sum(r.get("replay_status") == "REPLAYED" for r in replay_records),
    })


if __name__ == "__main__":
    main()
