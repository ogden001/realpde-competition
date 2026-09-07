#!/usr/bin/env python3
"""Matched P0-A+N2 continuation with only the MSE horizon weighting changed."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import realpde_loss_official_v9 as core
from horizon_cliff_audit import windowwise_horizon_rel_l2
from realpde_p0_data import H5WindowDataset
from realpde_p0_features import P0FeatureBuilder, P0FeatureConfig
from realpde_p0a_n2_full import (
    N2_WEIGHTS,
    build_model,
    historical_p0a_spacing,
    load_resume_checkpoint,
    manifest_paths,
    save_checkpoint,
    sha256,
)
from realpde_b1_p0a_n2 import forward
from realpde_tail_horizon import weighted_mse


def parse_offsets(value: str, additional_updates: int) -> tuple[int, ...]:
    offsets = tuple(sorted(set(int(item.strip()) for item in value.split(",") if item.strip())))
    if not offsets or offsets[-1] > additional_updates or offsets[0] < 1:
        raise ValueError("eval offsets must lie within additional update budget")
    return offsets


def prepare_out_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(path)
    path.mkdir(parents=True, exist_ok=True)


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


@torch.no_grad()
def evaluate_snapshot(
    model: torch.nn.Module,
    builder: P0FeatureBuilder,
    dev_paths: list[Path],
    *,
    kit_root: Path,
    batch_size: int,
    workers: int,
    device: torch.device,
    out_dir: Path,
) -> dict:
    dataset = H5WindowDataset(
        dev_paths,
        in_steps=20,
        out_steps=20,
        stride=20,
        sub_sample=2,
        include_pressure=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    for past, target, _, _ in loader:
        prediction = forward(model, builder, past.to(device, non_blocking=True))
        predictions.append(prediction.cpu().numpy().astype(np.float32))
        targets.append(target.numpy().astype(np.float32))
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = core.score_bundle(kit_root, prediction, target, 0.0, out_dir)
    trajectory_rows, anatomy = core.trajectory_rows(dataset, prediction, target, kit_root)
    with (out_dir / "trajectory_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectory_rows[0]))
        writer.writeheader()
        writer.writerows(trajectory_rows)
    window_rel = per_window_rel_l2(prediction, target)
    with (out_dir / "per_window_horizon_rel_l2.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["window_index", "horizon", "rel_l2"])
        writer.writeheader()
        for window_index in range(len(dataset)):
            for horizon in range(20):
                writer.writerow(
                    {
                        "window_index": window_index,
                        "horizon": horizon + 1,
                        "rel_l2": float(window_rel[window_index, horizon]),
                    }
                )
    return {
        "raw_errors": scored["raw_errors"],
        "horizon_rel_l2": windowwise_horizon_rel_l2(prediction, target),
        "tail_se_fraction": tail_se_fraction(prediction, target),
        "windows": len(dataset),
        "trajectories": len(trajectory_rows),
        "trajectory_anatomy": anatomy,
    }


def run(args: argparse.Namespace) -> None:
    prepare_out_dir(args.out_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("tail-weighted continuation requires CUDA")
    if args.tail_factor <= 0:
        raise ValueError("tail_factor must be positive")
    if args.additional_updates < 1 or args.micro_batch < 1 or args.accumulate < 1:
        raise ValueError("update and batch settings must be positive")
    eval_offsets = parse_offsets(args.eval_offsets, args.additional_updates)

    train_paths = manifest_paths(args.manifest, args.data_root, "train")
    dev_paths = manifest_paths(args.manifest, args.data_root, "dev")
    dx, dy = historical_p0a_spacing(train_paths)
    config = P0FeatureConfig(include_p0_a=True, include_p0_b=False, dx=dx, dy=dy)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    builder = P0FeatureBuilder(config).to(device)
    model = build_model(args.kit_root, builder, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-12)
    initial_update = load_resume_checkpoint(
        args.resume_checkpoint,
        model=model,
        optimizer=optimizer,
        config=config,
        lr_override=None,
    )
    if args.expected_start_update is not None and initial_update != args.expected_start_update:
        raise ValueError(f"resume update {initial_update} != expected {args.expected_start_update}")
    restored_lrs = sorted({float(group["lr"]) for group in optimizer.param_groups})

    train_dataset = H5WindowDataset(
        train_paths,
        in_steps=20,
        out_steps=20,
        stride=20,
        sub_sample=2,
        max_windows_per_trajectory=args.max_windows_per_trajectory,
        include_pressure=False,
    )
    generator = torch.Generator().manual_seed(args.seed + initial_update)
    loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
        drop_last=True,
    )
    iterator = iter(loader)

    metadata = {
        "experiment": "TAIL_WEIGHTED_N2_CONTINUATION",
        "tail_factor": float(args.tail_factor),
        "tail_count": 2,
        "mse_horizon_weights_normalized_mean_one": True,
        "other_n2_terms": "full20 unchanged",
        "n2_weights": N2_WEIGHTS,
        "manifest_sha256": sha256(args.manifest),
        "resume_checkpoint_sha256": sha256(args.resume_checkpoint),
        "initial_update": initial_update,
        "additional_updates": args.additional_updates,
        "eval_offsets": list(eval_offsets),
        "seed": args.seed,
        "sampler_seed": args.seed + initial_update,
        "micro_batch": args.micro_batch,
        "accumulate": args.accumulate,
        "effective_batch": args.micro_batch * args.accumulate,
        "workers": args.workers,
        "optimizer_lrs_restored": restored_lrs,
        "train_windows": len(train_dataset),
        "max_windows_per_trajectory": args.max_windows_per_trajectory,
        "train_trajectories": len(train_paths),
        "dev_trajectories": len(dev_paths),
        "locked_final_accessed": False,
        "codabench": False,
    }
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    snapshots: list[dict] = []
    baseline = evaluate_snapshot(
        model,
        builder,
        dev_paths,
        kit_root=args.kit_root,
        batch_size=args.eval_batch_size,
        workers=0,
        device=device,
        out_dir=args.out_dir / "eval_offset_0000",
    )
    snapshots.append({"offset": 0, "absolute_update": initial_update, **baseline})

    train_rows: list[dict] = []
    started = time.monotonic()
    for offset in range(1, args.additional_updates + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        aggregate = {name: 0.0 for name in N2_WEIGHTS}
        weighted_mse_value = 0.0
        for _ in range(args.accumulate):
            try:
                inputs, targets, _, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                inputs, targets, _, _ = next(iterator)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            prediction = forward(model, builder, inputs)
            parts = core.loss_parts(prediction, targets)
            mse_tail = weighted_mse(
                prediction,
                targets,
                tail_factor=args.tail_factor,
                tail_count=2,
            )
            loss = (
                N2_WEIGHTS["mse"] * mse_tail
                + N2_WEIGHTS["tke"] * parts["tke"]
                + N2_WEIGHTS["rel"] * parts["rel"]
                + N2_WEIGHTS["mvpe"] * parts["mvpe"]
            )
            (loss / args.accumulate).backward()
            weighted_mse_value += float(mse_tail.detach().cpu()) / args.accumulate
            for name in N2_WEIGHTS:
                aggregate[name] += float(parts[name].detach().cpu()) / args.accumulate
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        train_rows.append(
            {
                "offset": offset,
                "absolute_update": initial_update + offset,
                "tail_factor": args.tail_factor,
                "weighted_mse": weighted_mse_value,
                "plain_mse": aggregate["mse"],
                "tke": aggregate["tke"],
                "rel": aggregate["rel"],
                "mvpe": aggregate["mvpe"],
                "grad_norm": grad_norm,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        if offset in eval_offsets:
            snapshot = evaluate_snapshot(
                model,
                builder,
                dev_paths,
                kit_root=args.kit_root,
                batch_size=args.eval_batch_size,
                workers=0,
                device=device,
                out_dir=args.out_dir / f"eval_offset_{offset:04d}",
            )
            snapshots.append({"offset": offset, "absolute_update": initial_update + offset, **snapshot})

    with (args.out_dir / "training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(train_rows[0]))
        writer.writeheader()
        writer.writerows(train_rows)

    with (args.out_dir / "horizon_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["offset", "absolute_update", "horizon", "rel_l2"],
        )
        writer.writeheader()
        for snapshot in snapshots:
            for horizon, value in enumerate(snapshot["horizon_rel_l2"], start=1):
                writer.writerow(
                    {
                        "offset": snapshot["offset"],
                        "absolute_update": snapshot["absolute_update"],
                        "horizon": horizon,
                        "rel_l2": value,
                    }
                )

    with (args.out_dir / "eval_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "offset",
                "absolute_update",
                "rel_l2",
                "tke",
                "mvpe",
                "tail_se_fraction",
                "t18",
                "t19",
                "t20",
            ],
        )
        writer.writeheader()
        for snapshot in snapshots:
            h = snapshot["horizon_rel_l2"]
            raw = snapshot["raw_errors"]
            writer.writerow(
                {
                    "offset": snapshot["offset"],
                    "absolute_update": snapshot["absolute_update"],
                    "rel_l2": raw["rel_l2"],
                    "tke": raw["tke"],
                    "mvpe": raw["mvpe"],
                    "tail_se_fraction": snapshot["tail_se_fraction"],
                    "t18": h[17],
                    "t19": h[18],
                    "t20": h[19],
                }
            )

    final_update = initial_update + args.additional_updates
    save_checkpoint(
        args.out_dir / "model_resume.pth",
        model=model,
        optimizer=optimizer,
        update=final_update,
        config=config,
        metadata=metadata | {"completed_update": final_update, "tail_factor": args.tail_factor},
    )
    final = snapshots[-1]
    summary = {
        "tail_factor": args.tail_factor,
        "initial_update": initial_update,
        "completed_update": final_update,
        "optimizer_lrs_restored": restored_lrs,
        "baseline": snapshots[0],
        "final": final,
        "model_resume_sha256": sha256(args.out_dir / "model_resume.pth"),
        "locked_final_accessed": False,
        "codabench": False,
    }
    (args.out_dir / "final_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tail-factor", type=float, required=True)
    parser.add_argument("--additional-updates", type=int, default=820)
    parser.add_argument("--eval-offsets", default="410,820")
    parser.add_argument("--expected-start-update", type=int, default=30900)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-windows-per-trajectory", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
