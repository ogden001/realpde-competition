#!/usr/bin/env python3
"""Train-selected, dev-frozen PERSIST blend for CNO Future20 tail horizons."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import realpde_loss_official_v9 as core
from horizon_cliff_audit import (
    forward_p0a,
    load_cno,
    p0a_config_from_checkpoint,
    paths_from_manifest,
    sha256,
    windowwise_horizon_rel_l2,
)
from realpde_p0_data import H5WindowDataset
from realpde_p0_features import P0FeatureBuilder
from realpde_tail_horizon import blend_tail


def prepare_out_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(path)
    path.mkdir(parents=True, exist_ok=True)


def horizon_rel_l2(prediction: np.ndarray, target: np.ndarray, horizon_index: int) -> float:
    pred = prediction[:, horizon_index, ..., :2].reshape(prediction.shape[0], -1)
    truth = target[:, horizon_index, ..., :2].reshape(target.shape[0], -1)
    denom = np.linalg.norm(truth, axis=1).clip(min=1e-12)
    return float(np.mean(np.linalg.norm(pred - truth, axis=1) / denom))


def per_window_rel_l2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    diff = prediction[..., :2] - target[..., :2]
    numerator = np.linalg.norm(diff.reshape(diff.shape[0], diff.shape[1], -1), axis=2)
    truth = target[..., :2]
    denominator = np.linalg.norm(truth.reshape(truth.shape[0], truth.shape[1], -1), axis=2)
    return numerator / np.maximum(denominator, 1e-12)


def tail_se_fraction(prediction: np.ndarray, target: np.ndarray) -> float:
    error2 = np.square(prediction[..., :2] - target[..., :2])
    by_horizon = error2.sum(axis=(0, 2, 3, 4))
    return float(by_horizon[-2:].sum() / max(float(by_horizon.sum()), 1e-12))


def choose_alpha(
    cno: np.ndarray,
    persist: np.ndarray,
    target: np.ndarray,
    *,
    horizon_index: int,
    grid: list[float],
) -> tuple[float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    for alpha in grid:
        candidate = alpha * cno[:, horizon_index] + (1.0 - alpha) * persist[:, horizon_index]
        value = horizon_rel_l2(
            np.expand_dims(candidate, 1),
            np.expand_dims(target[:, horizon_index], 1),
            0,
        )
        rows.append({"horizon": float(horizon_index + 1), "alpha": float(alpha), "rel_l2": value})
    best = min(rows, key=lambda row: (row["rel_l2"], -row["alpha"]))
    return float(best["alpha"]), rows


@torch.no_grad()
def infer_split(
    paths: list[Path],
    *,
    model: torch.nn.Module,
    builder: P0FeatureBuilder,
    batch_size: int,
    device: torch.device,
) -> tuple[H5WindowDataset, np.ndarray, np.ndarray, np.ndarray]:
    dataset = H5WindowDataset(
        paths,
        in_steps=20,
        out_steps=20,
        stride=20,
        sub_sample=2,
        include_pressure=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    cno_rows: list[np.ndarray] = []
    persist_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    model.eval()
    for past, target, _, _ in loader:
        x = past.to(device)
        prediction = forward_p0a(model, builder, x).cpu().numpy().astype(np.float32)
        cno_rows.append(prediction)
        persist_rows.append(np.repeat(past[:, -1:].numpy(), 20, axis=1).astype(np.float32))
        target_rows.append(target.numpy().astype(np.float32))
    return dataset, np.concatenate(cno_rows), np.concatenate(persist_rows), np.concatenate(target_rows)


def score_prediction(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    dataset: H5WindowDataset,
    kit_root: Path,
    out_dir: Path,
) -> tuple[dict, list[dict]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = core.score_bundle(kit_root, prediction, target, 0.0, out_dir)
    rows, anatomy = core.trajectory_rows(dataset, prediction, target, kit_root)
    with (out_dir / "trajectory_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scored["trajectory_anatomy"] = anatomy
    return scored, rows


def run(args: argparse.Namespace) -> None:
    prepare_out_dir(args.out_dir)
    train_paths = paths_from_manifest(args.manifest, args.data_root, "train")
    dev_paths = paths_from_manifest(args.manifest, args.data_root, "dev")
    payload = torch.load(args.p0a_checkpoint, map_location="cpu", weights_only=False)
    config = p0a_config_from_checkpoint(payload)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    builder = P0FeatureBuilder(config).to(device)
    model = load_cno(
        args.kit_root,
        in_dim=len(builder.feature_names),
        checkpoint=args.p0a_checkpoint,
        device=device,
    )

    train_ds, train_cno, train_persist, train_target = infer_split(
        train_paths, model=model, builder=builder, batch_size=args.batch_size, device=device
    )
    dev_ds, dev_cno, dev_persist, dev_target = infer_split(
        dev_paths, model=model, builder=builder, batch_size=args.batch_size, device=device
    )
    grid = [round(value, 10) for value in np.linspace(0.0, 1.0, args.alpha_steps + 1)]
    alpha19, rows19 = choose_alpha(train_cno, train_persist, train_target, horizon_index=18, grid=grid)
    alpha20, rows20 = choose_alpha(train_cno, train_persist, train_target, horizon_index=19, grid=grid)
    with (args.out_dir / "alpha_scan_train.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["horizon", "alpha", "rel_l2"])
        writer.writeheader()
        writer.writerows(rows19 + rows20)

    blend = blend_tail(
        torch.from_numpy(dev_cno),
        torch.from_numpy(dev_persist),
        alpha19=alpha19,
        alpha20=alpha20,
    ).numpy()

    baseline_score, baseline_rows = score_prediction(
        prediction=dev_cno,
        target=dev_target,
        dataset=dev_ds,
        kit_root=args.kit_root,
        out_dir=args.out_dir / "dev_p0a",
    )
    blend_score, blend_rows = score_prediction(
        prediction=blend,
        target=dev_target,
        dataset=dev_ds,
        kit_root=args.kit_root,
        out_dir=args.out_dir / "dev_blend",
    )
    persist_score, _ = score_prediction(
        prediction=dev_persist,
        target=dev_target,
        dataset=dev_ds,
        kit_root=args.kit_root,
        out_dir=args.out_dir / "dev_persist",
    )

    baseline_h = windowwise_horizon_rel_l2(dev_cno, dev_target)
    blend_h = windowwise_horizon_rel_l2(blend, dev_target)
    persist_h = windowwise_horizon_rel_l2(dev_persist, dev_target)
    with (args.out_dir / "blend_horizon_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["horizon", "p0a_rel_l2", "blend_rel_l2", "persist_rel_l2"],
        )
        writer.writeheader()
        for horizon in range(20):
            writer.writerow(
                {
                    "horizon": horizon + 1,
                    "p0a_rel_l2": baseline_h[horizon],
                    "blend_rel_l2": blend_h[horizon],
                    "persist_rel_l2": persist_h[horizon],
                }
            )

    p0a_window = per_window_rel_l2(dev_cno, dev_target)
    blend_window = per_window_rel_l2(blend, dev_target)
    persist_window = per_window_rel_l2(dev_persist, dev_target)
    with (args.out_dir / "blend_per_window_horizon_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["window_index", "horizon", "p0a_rel_l2", "blend_rel_l2", "persist_rel_l2"],
        )
        writer.writeheader()
        for window_index in range(len(dev_ds)):
            for horizon in range(20):
                writer.writerow(
                    {
                        "window_index": window_index,
                        "horizon": horizon + 1,
                        "p0a_rel_l2": float(p0a_window[window_index, horizon]),
                        "blend_rel_l2": float(blend_window[window_index, horizon]),
                        "persist_rel_l2": float(persist_window[window_index, horizon]),
                    }
                )

    trajectory_rows: list[dict] = []
    baseline_by_id = {row["trajectory_id"]: row for row in baseline_rows}
    blend_by_id = {row["trajectory_id"]: row for row in blend_rows}
    for trajectory_id in sorted(baseline_by_id):
        row = {"trajectory_id": trajectory_id}
        for metric in ("rel_l2", "tke", "mvpe"):
            row[f"p0a_{metric}"] = float(baseline_by_id[trajectory_id][metric])
            row[f"blend_{metric}"] = float(blend_by_id[trajectory_id][metric])
        trajectory_rows.append(row)
    with (args.out_dir / "blend_trajectory_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectory_rows[0]))
        writer.writeheader()
        writer.writerows(trajectory_rows)

    evidence = {
        "selected_alpha": {"t19": alpha19, "t20": alpha20},
        "alpha_convention": "1.0=P0-A, 0.0=PERSIST",
        "alpha_grid": grid,
        "train_windows": len(train_ds),
        "dev_windows": len(dev_ds),
        "train_trajectories": len(train_paths),
        "dev_trajectories": len(dev_paths),
        "manifest_sha256": sha256(args.manifest),
        "p0a_checkpoint_sha256": sha256(args.p0a_checkpoint),
        "device": str(device),
        "raw_errors": {
            "p0a": baseline_score["raw_errors"],
            "blend": blend_score["raw_errors"],
            "persist": persist_score["raw_errors"],
        },
        "tail_se_fraction": {
            "p0a": tail_se_fraction(dev_cno, dev_target),
            "blend": tail_se_fraction(blend, dev_target),
            "persist": tail_se_fraction(dev_persist, dev_target),
        },
        "t18_t19_t20_rel_l2": {
            "p0a": baseline_h[17:20],
            "blend": blend_h[17:20],
            "persist": persist_h[17:20],
        },
        "locked_final_accessed": False,
        "codabench": False,
    }
    (args.out_dir / "blend_metrics.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--p0a-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--alpha-steps", type=int, default=20, help="20 gives grid spacing 0.05")
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    args = parser.parse_args()
    if args.alpha_steps < 1:
        raise ValueError("alpha_steps must be positive")
    run(args)


if __name__ == "__main__":
    main()
