#!/usr/bin/env python3
"""Audit parity between canonical P0-A validation and adaptive Gate loading."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import realpde_loss_official_v9 as core
import realpde_b1_p0a_n2 as base_api
from realpde_p0_data import H5WindowDataset
from realpde_p0_features import P0FeatureBuilder, P0FeatureConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_metrics(prediction: np.ndarray, target: np.ndarray, scoring) -> dict[str, float]:
    channels = scoring.measured_channels(target)
    return {
        "rel_l2": float(np.mean(scoring.rel_l2_per_sample(prediction, target, channels))),
        "tke": float(np.mean(scoring.tke_rel_l2_per_sample(prediction, target, channels))),
        "mvpe": float(scoring.mvpe_rel_l2(prediction, target)),
    }


def trajectory_metrics(dataset, prediction: np.ndarray, target: np.ndarray, scoring, label: str) -> list[dict]:
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index, ref in enumerate(dataset.refs):
        groups[ref.path.name].append((ref.start, index))
    rows = []
    for name, refs in sorted(groups.items()):
        indices = [index for _, index in sorted(refs)]
        p = np.concatenate([prediction[index] for index in indices], axis=0)[None]
        y = np.concatenate([target[index] for index in indices], axis=0)[None]
        channels = scoring.measured_channels(y)
        rows.append({
            "trajectory_id": name,
            "path": str(dataset.paths[next(i for i, path in enumerate(dataset.paths) if path.name == name)]),
            "windows": len(indices),
            f"{label}_rel_l2": float(scoring.rel_l2_per_sample(p, y, channels)[0]),
            f"{label}_tke": float(scoring.tke_rel_l2_per_sample(p, y, channels)[0]),
            f"{label}_mvpe": float(scoring.mvpe_rel_l2(p, y)),
        })
    return rows


def collect(model, builder, dataset, *, include_pressure: bool, batch_size: int, workers: int, device):
    # Rebuild explicitly so the canonical path and Gate path differ only in
    # the loader's pressure-channel policy.
    ds = dataset if dataset.include_pressure == include_pressure else type(dataset)(
        dataset.paths, in_steps=dataset.in_steps, out_steps=dataset.out_steps,
        stride=dataset.refs[1].start - dataset.refs[0].start if len(dataset.refs) > 1 else 20,
        sub_sample=dataset.sub_sample, include_pressure=include_pressure,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                         num_workers=workers, pin_memory=True,
                                         persistent_workers=False)
    predictions, targets, inputs = [], [], []
    with torch.inference_mode():
        for x, y, _, _ in loader:
            inputs.append(x.numpy().astype(np.float32))
            prediction = base_api.forward(model, builder, x.to(device, non_blocking=True))
            predictions.append(prediction.cpu().numpy().astype(np.float32))
            targets.append(y.numpy().astype(np.float32))
    return ds, np.concatenate(inputs), np.concatenate(predictions), np.concatenate(targets)


def run(args: argparse.Namespace) -> dict:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    dev_paths = [args.data_root / row["file"] for row in manifest["dev"]]
    checkpoint_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    feature_config = checkpoint_payload.get("feature_config")
    config = P0FeatureConfig(
        include_p0_a=feature_config.get("include_p0_a", True),
        include_p0_b=feature_config.get("include_p0_b", False),
        dx=feature_config["dx"], dy=feature_config["dy"],
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    builder = P0FeatureBuilder(config).to(device)
    gate_config = config
    gate_builder = P0FeatureBuilder(gate_config).to(device)
    model = base_api.load_model(args.kit_root, args.checkpoint, builder, device).eval()
    standard_ds = H5WindowDataset(dev_paths, include_pressure=False)
    gate_ds, standard_x, standard_pred, standard_y = collect(
        model, builder, standard_ds, include_pressure=False, batch_size=args.batch_size,
        workers=args.workers, device=device)
    _, loader_gate_x, loader_gate_pred, gate_y = collect(
        model, builder, standard_ds, include_pressure=True, batch_size=args.batch_size,
        workers=args.workers, device=device)
    gate_model = base_api.load_model(args.kit_root, args.checkpoint, gate_builder, device).eval()
    _, gate_x, gate_pred, gate_y = collect(
        gate_model, gate_builder, standard_ds, include_pressure=True, batch_size=args.batch_size,
        workers=args.workers, device=device)
    sys.path.insert(0, str(args.kit_root.resolve()))
    import scoring
    if not np.array_equal(standard_y, gate_y):
        raise AssertionError("canonical and Gate targets are not identical")
    trajectory = trajectory_metrics(gate_ds, standard_pred, standard_y, scoring, "base")
    gate_rows = trajectory_metrics(gate_ds, gate_pred, gate_y, scoring, "gate")
    by_id = {row["trajectory_id"]: row for row in gate_rows}
    for row in trajectory:
        row.update({key: value for key, value in by_id[row["trajectory_id"]].items() if key.startswith("gate_")})
    pressure_files = []
    for path in dev_paths:
        with h5py.File(path, "r") as handle:
            pressure_files.append({"trajectory_id": path.name, "has_pressure": "p" in handle or "measured_data/p" in handle})
    result = {
        "status": "REVIEW_REQUIRED",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_iteration": checkpoint_payload.get("iteration"),
        "feature_set": checkpoint_payload.get("feature_set"),
        "feature_names": checkpoint_payload.get("feature_names"),
        "feature_config": feature_config,
        "standard_feature_config": vars(config),
        "current_gate_feature_config": vars(gate_config),
        "checkpoint_manifest_sha256": checkpoint_payload.get("manifest_sha256"),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "scorer_sha256": sha256(args.kit_root / "scoring.py"),
        "dev_trajectories": len(dev_paths), "dev_windows": len(standard_ds),
        "standard_metrics": raw_metrics(standard_pred, standard_y, scoring),
        "loader_only_gate_metrics": raw_metrics(loader_gate_pred, gate_y, scoring),
        "gate_metrics": raw_metrics(gate_pred, gate_y, scoring),
        "prediction_parity": {
            "max_abs_diff": float(np.max(np.abs(standard_pred - gate_pred))),
            "mean_abs_diff": float(np.mean(np.abs(standard_pred - gate_pred))),
            "input_max_abs_diff": float(np.max(np.abs(standard_x - gate_x))),
            "input_mean_abs_diff": float(np.mean(np.abs(standard_x - gate_x))),
            "loader_only_prediction_max_abs_diff": float(np.max(np.abs(standard_pred - loader_gate_pred))),
            "loader_only_prediction_mean_abs_diff": float(np.mean(np.abs(standard_pred - loader_gate_pred))),
            "target_max_abs_diff": float(np.max(np.abs(standard_y - gate_y))),
            "target_mean_abs_diff": float(np.mean(np.abs(standard_y - gate_y))),
        },
        "pressure_presence": pressure_files,
        "historical_readme_metrics": {"rel_l2": 0.11284460, "tke": 0.50010282, "mvpe": 0.08728255},
        "trajectory_metrics": trajectory,
        "note": "Read-only baseline parity audit; no training, refit, package, locked-final/private-test, or Codabench.",
    }
    (args.out_dir / "baseline_parity.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.out_dir / "trajectory_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(trajectory[0])
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(trajectory)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("data-root", "kit-root", "checkpoint", "manifest", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
