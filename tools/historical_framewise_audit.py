#!/usr/bin/env python3
"""Read-only Future20 audit of explicitly registered historical checkpoints."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from realpde_p0_data import H5WindowDataset
from realpde_p0_features import P0FeatureBuilder, P0FeatureConfig

INVENTORY_FIELDS = ["experiment", "architecture", "features", "loss", "initialization", "train_split", "update", "checkpoint", "checkpoint_sha", "provenance_status", "replay_status"]
FRAME_FIELDS = ["experiment", "horizon", "rel_l2", "mvpe", "velocity_squared_error", "pred_speed_mean", "pred_speed_std", "target_speed_mean", "target_speed_std"]
WINDOW_FIELDS = ["experiment", "trajectory", "window_start", "horizon", "rel_l2", "mvpe", "velocity_squared_error", "pred_speed_mean", "target_speed_mean"]
AGGREGATE_FIELDS = ["experiment", "rel_l2", "tke", "mvpe", "windows", "trajectories"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def validate_inventory_row(row: dict) -> None:
    missing = [field for field in INVENTORY_FIELDS if field not in row]
    if missing:
        raise ValueError("inventory row missing " + ", ".join(missing))


def _velocity(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if prediction.shape != target.shape or prediction.ndim != 5 or prediction.shape[-1] < 2:
        raise ValueError(f"expected matching [N,T,H,W,C>=2], got {prediction.shape}/{target.shape}")
    return prediction[..., :2], target[..., :2]


def _rel_l2(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm((pred - target).reshape(pred.shape[0], -1), axis=1)
    denominator = np.linalg.norm(target.reshape(target.shape[0], -1), axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def _speed(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x, axis=-1)


def frame_metrics(prediction: np.ndarray, target: np.ndarray) -> list[dict]:
    pred, truth = _velocity(prediction, target)
    result = []
    for h in range(pred.shape[1]):
        p, y = pred[:, h], truth[:, h]
        result.append({"horizon": h + 1, "rel_l2": float(_rel_l2(p, y).mean()),
                       "mvpe": float(_rel_l2(p.mean(axis=1), y.mean(axis=1)).mean()),
                       "velocity_squared_error": float(np.square(p - y).sum(axis=-1).mean()),
                       "pred_speed_mean": float(_speed(p).mean()), "pred_speed_std": float(_speed(p).std()),
                       "target_speed_mean": float(_speed(y).mean()), "target_speed_std": float(_speed(y).std())})
    return result


def per_window_frame_metrics(experiment: str, trajectories: list[str], starts: list[int], prediction: np.ndarray, target: np.ndarray) -> list[dict]:
    pred, truth = _velocity(prediction, target)
    if len(trajectories) != len(pred) or len(starts) != len(pred):
        raise ValueError("window metadata length does not match predictions")
    rows = []
    for i, (trajectory, start) in enumerate(zip(trajectories, starts)):
        for h in range(pred.shape[1]):
            p, y = pred[i:i + 1, h], truth[i:i + 1, h]
            rows.append({"experiment": experiment, "trajectory": trajectory, "window_start": int(start), "horizon": h + 1,
                         "rel_l2": float(_rel_l2(p, y)[0]), "mvpe": float(_rel_l2(p.mean(axis=1), y.mean(axis=1))[0]),
                         "velocity_squared_error": float(np.square(p - y).sum(axis=-1).mean()), "pred_speed_mean": float(_speed(p).mean()),
                         "target_speed_mean": float(_speed(y).mean())})
    return rows


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader(); writer.writerows(rows)


def read_paths(manifest: Path, data_root: Path) -> list[Path]:
    payload = json.loads(manifest.read_text())
    paths = [data_root / row["file"] for row in payload["dev"]]
    if len(paths) != 16 or not all(p.is_file() for p in paths):
        raise RuntimeError("frozen dev split is not 16 readable trajectories")
    return paths


def cno(kit_root: Path, in_dim: int, out_dim: int, device: torch.device):
    sys.path.insert(0, str(kit_root))
    from rpde_baselines.model.cno import CNO3d
    return CNO3d(in_dim=in_dim, out_dim=out_dim, out_dim_mult=1, in_size=64, N_layers=3).to(device)


def p0_builder(paths: list[Path], device: torch.device) -> P0FeatureBuilder:
    from realpde_p0_data import read_grid
    x, y = read_grid(paths[0], sub_sample=2)
    return P0FeatureBuilder(P0FeatureConfig(include_p0_a=True, include_p0_b=False, dx=float(x[0, 1] - x[0, 0]), dy=float(y[1, 0] - y[0, 0]))).to(device)


def build_model(kind: str, payload: dict, kit_root: Path, paths: list[Path], device: torch.device):
    state = payload.get("model_state_dict", payload)
    if kind == "plain_cno":
        model = cno(kit_root, 3, 3, device); model.load_state_dict(state, strict=True)
        return model.eval(), None, lambda model, x: model(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
    builder = p0_builder(paths, device)
    if kind == "p0a_cno":
        model = cno(kit_root, len(builder.feature_names), 3, device); model.load_state_dict(state, strict=True)
        return model.eval(), builder, lambda model, x: model(builder(x).permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
    if kind == "mf01":
        from realpde_mf01 import MF01CNO
        model = MF01CNO(kit_root, len(builder.feature_names), device); model.load_state_dict(state, strict=True)
        return model.eval(), builder, lambda model, x: model(builder(x))
    raise ValueError(f"unsupported loader kind {kind}")


@torch.no_grad()
def replay(item: dict, dataset: H5WindowDataset, paths: list[Path], kit_root: Path, device: torch.device, batch_size: int, checkpoint: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model, _, forward = build_model(item["loader"], payload, kit_root, paths, device)
    preds, targets = [], []
    for x, y, _, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        preds.append(forward(model, x.to(device)).cpu().numpy().astype(np.float32)); targets.append(y.numpy().astype(np.float32))
    return np.concatenate(preds), np.concatenate(targets)


def aggregate(kit_root: Path, prediction: np.ndarray, target: np.ndarray) -> dict:
    sys.path.insert(0, str(kit_root)); import scoring
    c = scoring.measured_channels(target)
    return {"rel_l2": float(np.mean(scoring.rel_l2_per_sample(prediction, target, c))), "tke": float(np.mean(scoring.tke_rel_l2_per_sample(prediction, target, c))), "mvpe": float(scoring.mvpe_rel_l2(prediction, target))}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--data-root", type=Path, required=True); p.add_argument("--kit-root", type=Path, required=True); p.add_argument("--inventory-json", type=Path, required=True); p.add_argument("--out-dir", type=Path, required=True); p.add_argument("--batch-size", type=int, default=8); p.add_argument("--checkpoint-path-map-from", default=""); p.add_argument("--checkpoint-path-map-to", default=""); args = p.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()): raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    items = json.loads(args.inventory_json.read_text())
    for item in items: validate_inventory_row(item)
    paths = read_paths(args.manifest, args.data_root); ds = H5WindowDataset(paths, in_steps=20, out_steps=20, stride=20, sub_sample=2, include_pressure=False)
    if len(ds) != 659: raise RuntimeError(f"expected 659 windows, got {len(ds)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); inventory, frames, windows, aggregates, failures = [], [], [], [], []
    names, starts = [r.path.name for r in ds.refs], [r.start for r in ds.refs]
    for source in items:
        row = {field: source[field] for field in INVENTORY_FIELDS}
        checkpoint = Path(row["checkpoint"])
        if args.checkpoint_path_map_from and str(checkpoint).startswith(args.checkpoint_path_map_from):
            checkpoint = Path(args.checkpoint_path_map_to + str(checkpoint)[len(args.checkpoint_path_map_from):])
        if row["provenance_status"] != "COMPLETE_PROVENANCE": inventory.append(row); continue
        try:
            if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
            actual = sha256(checkpoint)
            if row["checkpoint_sha"] not in ("", actual): raise RuntimeError("checkpoint SHA mismatch")
            row["checkpoint_sha"] = actual
            prediction, target = replay(source, ds, paths, args.kit_root, device, args.batch_size, checkpoint)
            if prediction.shape != (659, 20, 32, 64, 3): raise RuntimeError(f"unexpected prediction shape {prediction.shape}")
            row["replay_status"] = "REPLAYED"; inventory.append(row)
            frames.extend({"experiment": row["experiment"], **x} for x in frame_metrics(prediction, target))
            windows.extend(per_window_frame_metrics(row["experiment"], names, starts, prediction, target))
            aggregates.append({"experiment": row["experiment"], **aggregate(args.kit_root, prediction, target), "windows": 659, "trajectories": 16})
        except Exception as exc:
            row["replay_status"] = "REPLAY_FAILED: " + str(exc); inventory.append(row); failures.append({"experiment": row["experiment"], "reason": str(exc)})
            if device.type == "cuda": torch.cuda.empty_cache()
    write_csv(args.out_dir / "experiment_inventory.csv", INVENTORY_FIELDS, inventory); write_csv(args.out_dir / "frame_metrics.csv", FRAME_FIELDS, frames); write_csv(args.out_dir / "per_window_frame_metrics.csv", WINDOW_FIELDS, windows); write_csv(args.out_dir / "aggregate_metrics.csv", AGGREGATE_FIELDS, aggregates)
    (args.out_dir / "replay_failures.json").write_text(json.dumps(failures, indent=2) + "\n")


if __name__ == "__main__": main()
