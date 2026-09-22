#!/usr/bin/env python3
"""Replay residual checkpoints on one frozen dev protocol and write standard diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from post_train_diagnostics import write_post_train_diagnostics  # noqa: E402
from realpde_h5_feature_adapter_train import (  # noqa: E402
    H5WindowDataset,
    measured_channels,
    mvpe_rel_l2_per_sample,
    paths_from_split_manifest,
    rel_l2_per_sample,
    tke_rel_l2_per_sample,
)
from residual_multi import load_full_residual_model  # noqa: E402


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model must be LABEL=/absolute/path/to/model.pth")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label or any(ch in label for ch in "/\\"):
        raise argparse.ArgumentTypeError("model label must be a non-empty path-safe name")
    return label, Path(raw_path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def replay(model_path: Path, loader: DataLoader, model_root: Path, device: torch.device):
    model, metadata = load_full_residual_model(model_path, model_root, device)
    bases, preds, targets = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        base = model.base_predict(x)
        delta = model.predict_delta(x, base)
        pred = model.combine(base, delta, 1.0)
        bases.append(base.cpu().numpy().astype(np.float32))
        preds.append(pred.cpu().numpy().astype(np.float32))
        targets.append(y.numpy().astype(np.float32))
    return np.concatenate(bases), np.concatenate(preds), np.concatenate(targets), metadata


def official_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    channels = measured_channels(target)
    return {
        "rel_l2_raw": float(np.mean(rel_l2_per_sample(pred, target, channels))),
        "tke_raw": float(np.mean(tke_rel_l2_per_sample(pred, target, channels))),
        "mvpe_raw": float(np.mean(mvpe_rel_l2_per_sample(pred, target))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--realpdebench-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--model", action="append", type=parse_model, required=True,
                        help="repeatable LABEL=/path/model.pth")
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    labels = [label for label, _ in args.model]
    if len(labels) != len(set(labels)):
        raise ValueError("model labels must be unique")
    if args.baseline_label not in labels:
        raise ValueError("baseline label must appear in --model")
    if args.out_root.exists() and any(args.out_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.out_root}")
    args.out_root.mkdir(parents=True, exist_ok=True)

    _, val_paths = paths_from_split_manifest(
        args.real_root, args.split_manifest, allow_train_dev_overlap=True)
    dataset = H5WindowDataset(
        val_paths, in_steps=20, out_steps=20, stride=20, sub_sample=2,
        max_windows_per_trajectory=None, include_pressure=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=torch.cuda.is_available())
    names = [ref.path.name for ref in dataset.refs]
    starts = [ref.start for ref in dataset.refs]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    summary_rows: list[dict[str, object]] = []
    horizon_by_label: dict[str, list[dict[str, str]]] = {}
    for label, path in args.model:
        if not path.is_file():
            raise FileNotFoundError(path)
        base, pred, target, metadata = replay(path, loader, args.realpdebench_root, device)
        diag_dir = args.out_root / label
        diag = write_post_train_diagnostics(
            out_dir=diag_dir,
            experiment=label,
            prediction=pred,
            target=target,
            trajectories=names,
            starts=starts,
            base_prediction=base,
        )
        metrics = official_metrics(pred, target)
        residual = diag["residual_before_after"]
        mean_fluct = diag["mean_fluctuation"]
        summary_rows.append({
            "label": label,
            "checkpoint": str(path),
            "checkpoint_iteration": int(metadata.get("iteration", -1)),
            **metrics,
            "mean_field_rel_l2": float(mean_fluct["mean_field_rel_l2"]),
            "fluctuation_rel_l2": float(mean_fluct["fluctuation_rel_l2"]),
            "tke_energy_ratio": float(mean_fluct["tke_energy_ratio"]),
            "correction_help_fraction": float(residual["correction_help_fraction"]),
            "delta_rms": float(residual["delta_rms"]),
        })
        with (diag_dir / "by_horizon.csv").open(encoding="utf-8") as handle:
            horizon_by_label[label] = list(csv.DictReader(handle))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    baseline = next(row for row in summary_rows if row["label"] == args.baseline_label)
    for row in summary_rows:
        for key in ("rel_l2_raw", "tke_raw", "mvpe_raw", "mean_field_rel_l2", "fluctuation_rel_l2"):
            base_value = float(baseline[key])
            row[f"delta_{key}_pct_vs_baseline"] = (
                100.0 * (float(row[key]) - base_value) / max(abs(base_value), 1e-12))
    write_csv(args.out_root / "comparison_summary.csv", summary_rows)

    horizon_compare: list[dict[str, object]] = []
    baseline_h = {int(row["horizon"]): row for row in horizon_by_label[args.baseline_label]}
    for label in labels:
        for row in horizon_by_label[label]:
            h = int(row["horizon"])
            base_row = baseline_h[h]
            rel = float(row["frame_rel_l2"])
            base_rel = float(base_row["frame_rel_l2"])
            tke_contrib = float(row["tke_contrib_rel_l2"])
            base_tke_contrib = float(base_row["tke_contrib_rel_l2"])
            horizon_compare.append({
                "label": label,
                "horizon": h,
                "frame_rel_l2": rel,
                "delta_frame_rel_l2_pct_vs_baseline": (
                    100.0 * (rel - base_rel) / max(abs(base_rel), 1e-12)),
                "velocity_rmse": float(row["velocity_rmse"]),
                "tke_contrib_rel_l2": tke_contrib,
                "delta_tke_contrib_rel_l2_pct_vs_baseline": (
                    100.0 * (tke_contrib - base_tke_contrib) / max(abs(base_tke_contrib), 1e-12)),
                "tke_contrib_ratio": float(row["tke_contrib_ratio"]),
                "correction_help_fraction": float(row.get("correction_help_fraction", "nan")),
            })
    write_csv(args.out_root / "comparison_by_horizon.csv", horizon_compare)
    (args.out_root / "manifest.json").write_text(json.dumps({
        "baseline_label": args.baseline_label,
        "models": [{"label": label, "checkpoint": str(path)} for label, path in args.model],
        "dev_windows": len(dataset),
        "dev_trajectories": len(val_paths),
        "diagnostics_version": 1,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
