#!/usr/bin/env python3
"""Official-v9-scored, submission-compatible P0-A ablation.

This keeps the loss experiment's manifest, CNO implementation, loss variants,
and supplied v9 scorer.  It deliberately enables only P0-A: every added field
is causally derived from the input window.  Track 1 v9 scored calls pass an
empty metadata dictionary, so Re/AoA-dependent P0-B fields are excluded.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

import realpde_loss_official_v9 as core
from realpde_p0_data import read_grid
from realpde_p0_features import P0FeatureBuilder, P0FeatureConfig


def adapt_input_weight(model: torch.nn.Module, checkpoint: dict, in_channels: int) -> None:
    """Expand a three-channel checkpoint without changing its initial output."""
    state = checkpoint.get("model_state_dict", checkpoint)
    key = "lift.inter_CNOBlock.convolution.weight"
    if key not in state:
        raise KeyError(f"checkpoint lacks {key}")
    old = state[key]
    expanded = model.state_dict()[key].clone()
    if old.shape[1] != 3 or expanded.shape[1] != in_channels:
        raise ValueError(f"unexpected input weights: checkpoint={tuple(old.shape)}, model={tuple(expanded.shape)}")
    expanded.zero_()
    expanded[:, :3].copy_(old)
    state = dict(state)
    state[key] = expanded
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def load_p0a_cno(kit_root: Path, checkpoint: Path, builder: P0FeatureBuilder, device: torch.device) -> torch.nn.Module:
    import sys
    sys.path.insert(0, str(kit_root))
    from rpde_baselines.model.cno import CNO3d
    model = CNO3d(in_dim=len(builder.feature_names), out_dim=3, out_dim_mult=1, in_size=64, N_layers=3).to(device)
    adapt_input_weight(model, torch.load(checkpoint, map_location="cpu"), len(builder.feature_names))
    return model


def forward(model: torch.nn.Module, builder: P0FeatureBuilder, x: torch.Tensor) -> torch.Tensor:
    # Explicit channel-first layout avoids CNO's historical C<T heuristic.
    features = builder(x)
    return model(features.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)


@torch.no_grad()
def evaluate(model: torch.nn.Module, builder: P0FeatureBuilder, paths: list[Path], args: argparse.Namespace,
             device: torch.device, kit_root: Path, out: Path) -> dict:
    """Use the supplied scorer and time P0 construction plus model inference."""
    import numpy as np
    out.mkdir(parents=True, exist_ok=True)
    ds, data = core.loader(paths, args, shuffle=False)
    model.eval()
    predictions, targets, elapsed = [], [], 0.0
    for x, y, _, _ in data:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda": torch.cuda.synchronize()
        started = time.perf_counter()
        pred = forward(model, builder, x)
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        predictions.append(pred.cpu().numpy().astype(np.float32))
        targets.append(y.numpy().astype(np.float32))
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    result = core.score_bundle(kit_root, prediction, target, elapsed / len(ds), out)
    rows, anatomy = core.trajectory_rows(ds, prediction, target, kit_root)
    result["windows"] = len(ds)
    result["trajectories"] = len(rows)
    result["trajectory_anatomy"] = anatomy
    import csv
    with (out / "trajectory_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return result


def run(args: argparse.Namespace) -> None:
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if not (args.kit_root / "scoring.py").is_file():
        raise FileNotFoundError(args.kit_root / "scoring.py")
    core.set_seed(args.seed)
    args.out_dir.mkdir(parents=True)
    manifest, train_paths = core.read_manifest(args.manifest, "train")
    _, eval_paths = core.read_manifest(args.manifest, args.eval_split)
    del manifest
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_grid, y_grid = read_grid(train_paths[0], sub_sample=2)
    builder = P0FeatureBuilder(
        P0FeatureConfig(include_p0_a=True, include_p0_b=False,
                        dx=float(x_grid[0, 1] - x_grid[0, 0]),
                        dy=float(y_grid[1, 0] - y_grid[0, 0]))
    ).to(device)
    weights = core.VARIANTS[args.variant]
    metadata = {
        "feature_set": "P0-A only (input-derived; no metadata)",
        "feature_names": list(builder.feature_names),
        "metadata_policy": "v9 scored calls have empty metadata; Re/AoA P0-B excluded",
        "variant": args.variant, "weights": weights, "seed": args.seed,
        "updates": args.updates, "eval_interval": args.eval_interval,
        "batch_size": args.batch_size, "workers": args.workers,
        "manifest": str(args.manifest.resolve()), "manifest_sha256": core.sha256(args.manifest),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": core.sha256(args.checkpoint),
        "kit_root": str(args.kit_root.resolve()), "kit_scorer_sha256": core.sha256(args.kit_root / "scoring.py"),
        "time_scope": "P0-A construction plus CNO forward; excludes loading and data I/O",
    }
    core.json_dump(args.out_dir / "run_metadata.json", metadata)
    model = load_p0a_cno(args.kit_root, args.checkpoint, builder, device)
    baseline = evaluate(model, builder, eval_paths, args, device, args.kit_root, args.out_dir / "eval_baseline")
    history = [{"iteration": 0, **baseline["raw_errors"]}]
    best, best_key = None, float("inf")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_ds, train_loader = core.loader(train_paths, args, shuffle=True)
    iterator = iter(train_loader)
    for step in range(1, args.updates + 1):
        try:
            x, y, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y, _, _ = next(iterator)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        model.train()
        parts = core.loss_parts(forward(model, builder, x), y)
        loss = sum(weights[name] * value for name, value in parts.items())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_interval == 0 or step == args.updates:
            dev = evaluate(model, builder, eval_paths, args, device, args.kit_root, args.out_dir / f"eval_{step:04d}")
            raw = dev["raw_errors"]
            history.append({"iteration": step, **raw, "official_v9_subscores": dev["official_v9_subscores"]})
            # Same gate as the loss runner: optimize TKE only when accuracy/MVPE stay near baseline.
            key = raw["tke"] if raw["rel_l2"] <= baseline["raw_errors"]["rel_l2"] * 1.02 and raw["mvpe"] <= baseline["raw_errors"]["mvpe"] * 1.02 else raw["tke"] + 1e6
            if key < best_key:
                best_key, best = key, step
                torch.save({"model_state_dict": model.state_dict(), "iteration": step, "weights": weights,
                            "feature_names": builder.feature_names}, args.out_dir / "model_best.pth")
    if best is None:
        shutil.copy2(args.out_dir / "model_latest.pth", args.out_dir / "model_best.pth")
    core.json_dump(args.out_dir / "summary.json", {"metadata": metadata | {"train_windows": len(train_ds), "best_iteration": best},
                                                    "baseline": baseline, "history": history})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--real-root", type=Path, required=True, help="required for interface parity; manifest supplies paths")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--kit-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--variant", choices=tuple(core.VARIANTS), default="E1")
    p.add_argument("--eval-split", default="dev")
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--eval-interval", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-5)
    args = p.parse_args()
    if args.updates < 1 or args.eval_interval < 1:
        p.error("updates and eval-interval must be positive")
    run(args)


if __name__ == "__main__":
    main()
