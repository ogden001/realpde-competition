#!/usr/bin/env python3
"""First-round, trajectory-held-out CNO fine-tuning for P0-A/P0-B.

This script is intentionally explicit about every data and output path.  It
does not extract archives or write derived feature datasets: features are made
online for each batch by :mod:`realpde_p0_features`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from realpde_p0_data import H5WindowDataset, list_h5, read_grid, split_paths
from realpde_p0_features import P0FeatureBuilder, P0FeatureConfig


def rel_l2(pred: Tensor, target: Tensor) -> Tensor:
    return (pred.reshape(pred.shape[0], -1).sub(target.reshape(target.shape[0], -1)).norm(dim=1)
            / target.reshape(target.shape[0], -1).norm(dim=1).clamp_min(1e-8)).mean()


def tke_rel_l2(pred: Tensor, target: Tensor) -> Tensor:
    def tke(x: Tensor) -> Tensor:
        uv = x[..., :2]
        fluctuation = uv - uv.mean(dim=1, keepdim=True)
        return 0.5 * fluctuation.square().mean(dim=1).sum(dim=-1)
    p, y = tke(pred), tke(target)
    return (p.reshape(p.shape[0], -1).sub(y.reshape(y.shape[0], -1)).norm(dim=1)
            / y.reshape(y.shape[0], -1).norm(dim=1).clamp_min(1e-8)).mean()


def adapt_input_weight(model: nn.Module, checkpoint: dict, in_channels: int) -> None:
    """Load a 3-channel CNO checkpoint and zero-initialize added feature paths."""
    state = checkpoint["model_state_dict"]
    key = "lift.inter_CNOBlock.convolution.weight"
    if key not in state:
        raise ValueError(f"checkpoint has no {key}")
    old = state[key]
    if old.shape[1] > in_channels:
        raise ValueError(f"checkpoint has {old.shape[1]} inputs, requested {in_channels}")
    model_state = model.state_dict()
    expanded = model_state[key].clone()
    expanded.zero_()
    expanded[:, :old.shape[1]].copy_(old)
    state = dict(state)
    state[key] = expanded
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if k != key]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def cno_forward(model: nn.Module, features: Tensor) -> Tensor:
    """Call the historical CNO with an explicit channel-first layout.

    Its built-in layout heuristic assumes ``C < T``.  P0 has 25 channels and
    20 input frames, so the heuristic is no longer valid.
    """
    raw = model(features.permute(0, 4, 1, 2, 3))
    return raw.permute(0, 2, 3, 4, 1)


def evaluate(model: nn.Module, builder: P0FeatureBuilder | None, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval(); sums = {"loss": 0.0, "rel_l2": 0.0, "tke": 0.0, "n": 0.0, "seconds": 0.0}
    with torch.no_grad():
        for x, y, condition, _ in loader:
            x, y, condition = x.to(device), y.to(device), condition.to(device)
            features = builder(x, {"re": condition[:, 0], "aoa": condition[:, 1]}) if builder else x
            if device.type == "cuda": torch.cuda.synchronize()
            started = time.perf_counter(); pred = cno_forward(model, features)
            if device.type == "cuda": torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            point = torch.mean((pred[..., :2] - y[..., :2]).square())
            count = x.shape[0]
            sums["loss"] += float(point) * count; sums["rel_l2"] += float(rel_l2(pred[..., :2], y[..., :2])) * count
            sums["tke"] += float(tke_rel_l2(pred, y)) * count; sums["seconds"] += elapsed; sums["n"] += count
    n = max(sums.pop("n"), 1.0)
    return {k: v / n for k, v in sums.items()} | {"n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--realpdebench-root", type=Path, required=True, help="source checkout containing realpdebench/model/cno.py")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--feature-set", choices=("baseline", "p0ab"), required=True)
    ap.add_argument("--updates", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--val-fraction", type=float, default=.2)
    ap.add_argument("--max-windows-per-trajectory", type=int, default=None)
    args = ap.parse_args()
    if args.updates < 1 or args.batch_size < 1: raise ValueError("updates and batch-size must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, str(args.realpdebench_root))
    from realpdebench.model.cno import CNO3d

    paths = list_h5(args.real_root); train_paths, val_paths = split_paths(paths, args.val_fraction, args.seed)
    train_set = H5WindowDataset(train_paths, max_windows_per_trajectory=args.max_windows_per_trajectory)
    val_set = H5WindowDataset(val_paths, max_windows_per_trajectory=args.max_windows_per_trajectory)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    builder = None
    channels = 3
    if args.feature_set == "p0ab":
        x_grid, y_grid = read_grid(train_paths[0], sub_sample=2)
        re_values = []
        for path in train_paths:
            import h5py
            with h5py.File(path, "r") as f: re_values.append(float(f["re"][()]))
        re_center, re_scale = float(np.mean(re_values)), max(float(np.std(re_values)), 1.0)
        builder = P0FeatureBuilder(P0FeatureConfig(dx=float(x_grid[0, 1] - x_grid[0, 0]), dy=float(y_grid[1, 0] - y_grid[0, 0]),
                                                     re_center=re_center, re_scale=re_scale), x_grid, y_grid).to(device)
        channels = len(builder.feature_names)

    model = CNO3d(in_dim=channels, out_dim=3, out_dim_mult=1, in_size=64, N_layers=3).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    adapt_input_weight(model, checkpoint, channels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history: list[dict[str, float]] = []
    baseline = evaluate(model, builder, val_loader, device); baseline["iteration"] = 0; history.append(baseline)
    best = baseline; torch.save({"model_state_dict": model.state_dict(), "metrics": best, "feature_set": args.feature_set}, args.out_dir / "best.pth")
    iterator = iter(train_loader)
    for step in range(1, args.updates + 1):
        try: x, y, condition, _ = next(iterator)
        except StopIteration: iterator = iter(train_loader); x, y, condition, _ = next(iterator)
        x, y, condition = x.to(device), y.to(device), condition.to(device)
        features = builder(x, {"re": condition[:, 0], "aoa": condition[:, 1]}) if builder else x
        pred = cno_forward(model, features)
        loss = torch.mean((pred[..., :2] - y[..., :2]).square()) + .05 * tke_rel_l2(pred, y)
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == args.updates or step % max(1, args.updates // 4) == 0:
            row = evaluate(model, builder, val_loader, device); row["iteration"] = step; history.append(row)
            if row["rel_l2"] <= best["rel_l2"]:
                best = row; torch.save({"model_state_dict": model.state_dict(), "metrics": best, "feature_set": args.feature_set}, args.out_dir / "best.pth")
            print(json.dumps(row, sort_keys=True), flush=True)
    (args.out_dir / "summary.json").write_text(json.dumps({"args": vars(args), "baseline": baseline, "best": best, "history": history}, indent=2, default=str))


if __name__ == "__main__":
    main()
