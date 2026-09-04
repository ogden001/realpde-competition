"""MF-01 output factorization and matched P0-A/N2 training runner."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

import realpde_loss_official_v9 as core
from realpde_p0_data import read_grid
from realpde_p0_features import P0FeatureBuilder, P0FeatureConfig

N2_WEIGHTS = {"mse": 1.0, "tke": 0.05, "rel": 0.027514, "mvpe": 0.009757}


def factorized_reconstruct(mean_raw: Tensor, fluct_raw: Tensor) -> tuple[Tensor, Tensor]:
    """Return mean plus a temporal zero-mean fluctuation."""
    mean_field = mean_raw.mean(dim=1, keepdim=True).expand_as(mean_raw)
    fluctuation = fluct_raw - fluct_raw.mean(dim=1, keepdim=True)
    return mean_field + fluctuation, fluctuation


class MF01CNO(nn.Module):
    """CNO with a five-channel output projection: mean uv, fluctuation uv, p."""

    def __init__(self, kit_root: Path, in_dim: int, device: torch.device):
        super().__init__()
        sys.path.insert(0, str(kit_root))
        from rpde_baselines.model.cno import CNO3d
        self.cno = CNO3d(in_dim=in_dim, out_dim=5, out_dim_mult=1, in_size=64, N_layers=3).to(device)

    def forward(self, x: Tensor) -> Tensor:
        raw = self.cno(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        uv, pressure = raw[..., :4], raw[..., 4:5]
        reconstructed, _ = factorized_reconstruct(uv[..., :2], uv[..., 2:4])
        return torch.cat((reconstructed, pressure), dim=-1)


def adapt_input_weight(model: nn.Module, checkpoint: dict, in_channels: int) -> None:
    state = checkpoint.get("model_state_dict", checkpoint)
    key = "cno.lift.inter_CNOBlock.convolution.weight" if "cno.lift.inter_CNOBlock.convolution.weight" in model.state_dict() else "lift.inter_CNOBlock.convolution.weight"
    old_key = key.removeprefix("cno.")
    old = state.get(old_key, state.get(key))
    if old is None:
        raise KeyError(f"checkpoint lacks {old_key}")
    expanded = model.state_dict()[key].clone()
    if old.shape[1] != 3 or expanded.shape[1] != in_channels:
        raise ValueError(f"expected 3-channel checkpoint and {in_channels}-channel model")
    expanded[:, :3].copy_(old)
    expanded[:, 3:].zero_()
    state = dict(state)
    state[key] = expanded
    if key != old_key:
        state.pop(old_key, None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [name for name in unexpected if name not in {old_key, key}]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def init_mf_from_direct(model: MF01CNO, checkpoint: dict, in_channels: int) -> None:
    """Copy direct projection rows to mean/fluctuation/pressure rows."""
    state = checkpoint.get("model_state_dict", checkpoint)
    target = model.cno.state_dict()
    for name, value in state.items():
        if name in {"project.convolution.weight", "project.convolution.bias"}:
            continue
        if name == "lift.inter_CNOBlock.convolution.weight":
            expanded = target[name].clone(); expanded[:, :3].copy_(value); expanded[:, 3:].zero_(); target[name] = expanded
        elif name in target:
            target[name] = value
    model.cno.load_state_dict(target, strict=True)
    target = model.cno.state_dict()
    for prefix in ("project.convolution.weight", "project.convolution.bias"):
        if prefix not in state:
            raise KeyError(f"checkpoint lacks {prefix}")
        source = state[prefix]
        expanded = target[prefix].clone()
        expanded[:2].copy_(source[:2])
        expanded[2:4].copy_(source[:2])
        expanded[4].copy_(source[2])
        target[prefix] = expanded
    model.cno.load_state_dict(target, strict=True)


def cno_direct(kit_root: Path, in_dim: int, device: torch.device) -> nn.Module:
    sys.path.insert(0, str(kit_root))
    from rpde_baselines.model.cno import CNO3d
    return CNO3d(in_dim=in_dim, out_dim=3, out_dim_mult=1, in_size=64, N_layers=3).to(device)


def build_features(paths: list[Path], device: torch.device) -> tuple[P0FeatureBuilder, P0FeatureConfig]:
    x_grid, y_grid = read_grid(paths[0], sub_sample=2)
    config = P0FeatureConfig(include_p0_a=True, include_p0_b=False,
                              dx=float(x_grid[0, 1] - x_grid[0, 0]),
                              dy=float(y_grid[1, 0] - y_grid[0, 0]))
    return P0FeatureBuilder(config).to(device), config


def forward(model: nn.Module, builder: P0FeatureBuilder, x: Tensor) -> Tensor:
    features = builder(x)
    if isinstance(model, MF01CNO):
        return model(features)
    return model(features.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)


@torch.no_grad()
def evaluate(model: nn.Module, builder: P0FeatureBuilder, paths: list[Path], args: argparse.Namespace,
             device: torch.device, kit_root: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ds, loader = core.loader(paths, args, shuffle=False)
    predictions, targets, elapsed = [], [], 0.0
    model.eval()
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda": torch.cuda.synchronize()
        started = time.perf_counter(); prediction = forward(model, builder, x)
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        predictions.append(prediction.cpu().numpy().astype(np.float32)); targets.append(y.numpy().astype(np.float32))
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    result = core.score_bundle(kit_root, prediction, target, elapsed / len(ds), out)
    rows, anatomy = core.trajectory_rows(ds, prediction, target, kit_root)
    result.update({"windows": len(ds), "trajectories": len(rows), "trajectory_anatomy": anatomy})
    with (out / "trajectory_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(out / "predictions.npz", prediction=prediction, target=target)
    return result


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, step: int, metadata: dict) -> None:
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "iteration": step, "metadata": metadata, "feature_set": "P0-A", "loss_weights": N2_WEIGHTS}, path)


def run(args: argparse.Namespace) -> None:
    if args.out_dir.exists(): raise FileExistsError(args.out_dir)
    core.set_seed(args.seed)
    args.out_dir.mkdir(parents=True)
    _, train_paths = core.read_manifest(args.manifest, "train")
    _, dev_paths = core.read_manifest(args.manifest, "dev")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    builder, config = build_features(train_paths, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = MF01CNO(args.kit_root, len(builder.feature_names), device) if args.mode == "mf01" else cno_direct(args.kit_root, len(builder.feature_names), device)
    if args.mode == "mf01": init_mf_from_direct(model, checkpoint, len(builder.feature_names))
    else: adapt_input_weight(model, checkpoint, len(builder.feature_names))
    metadata = {"experiment_id": args.experiment_id, "mode": args.mode, "seed": args.seed, "updates": args.updates,
                "milestones": args.milestones, "lr": args.lr, "batch_size": args.batch_size, "workers": args.workers,
                "manifest_sha256": core.sha256(args.manifest), "checkpoint_sha256": core.sha256(args.checkpoint),
                "scorer_sha256": core.sha256(args.kit_root / "scoring.py"), "feature_names": list(builder.feature_names),
                "feature_config": vars(config), "loss_weights": N2_WEIGHTS, "locked_final_accessed": False,
                "codabench_accessed": False, "device": str(device), "started_at": time.time()}
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    train_ds, train_loader = core.loader(train_paths, args, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    baseline = evaluate(model, builder, dev_paths, args, device, args.kit_root, args.out_dir / "eval_00000")
    save_checkpoint(args.out_dir / "model_update_00000.pth", model, optimizer, 0, metadata)
    iterator, history = iter(train_loader), [{"iteration": 0, **baseline["raw_errors"]}]
    started = time.monotonic()
    for step in range(1, args.updates + 1):
        try: x, y, _, _ = next(iterator)
        except StopIteration: iterator = iter(train_loader); x, y, _, _ = next(iterator)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        model.train(); pred = forward(model, builder, x); parts = core.loss_parts(pred, y)
        loss = sum(N2_WEIGHTS[name] * parts[name] for name in N2_WEIGHTS)
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step in args.milestones:
            evaluation = evaluate(model, builder, dev_paths, args, device, args.kit_root, args.out_dir / f"eval_{step:05d}")
            row = {"iteration": step, **evaluation["raw_errors"], "official_v9_subscores": evaluation["official_v9_subscores"], "elapsed_seconds": time.monotonic() - started}
            history.append(row); save_checkpoint(args.out_dir / f"model_update_{step:05d}.pth", model, optimizer, step, metadata)
            print(json.dumps(row, sort_keys=True), flush=True)
    save_checkpoint(args.out_dir / "model_last.pth", model, optimizer, args.updates, metadata)
    (args.out_dir / "summary.json").write_text(json.dumps({"metadata": metadata | {"train_windows": len(train_ds), "elapsed_seconds": time.monotonic() - started}, "history": history}, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("control", "mf01"), required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--data-root", type=Path, required=False)  # retained for explicit CLI provenance
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--milestones", type=int, nargs="+", default=[500, 1000, 1500])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260901)
    run(parser.parse_args())


if __name__ == "__main__": main()
