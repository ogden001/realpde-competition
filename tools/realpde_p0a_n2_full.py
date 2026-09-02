#!/usr/bin/env python3
"""Memory-capped all-data P0-A + N2 CNO training primitives.

This module intentionally keeps P0-A pure torch so the submission wrapper can
use the same numerical definition as the training runner.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import random
import signal
import time
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


LIFT_WEIGHT_KEY = "lift.inter_CNOBlock.convolution.weight"
MAX_GPU_GIB = 23.5
MAX_WALL_SECONDS = 6 * 60 * 60
MAX_TRAIN_SECONDS = 5 * 60 * 60 + 40 * 60
N2_WEIGHTS = {"mse": 1.0, "tke": 0.05, "rel": 0.027514, "mvpe": 0.009757}


@dataclass(frozen=True)
class P0AConfig:
    dx: float
    dy: float

    def __post_init__(self) -> None:
        if self.dx == 0.0 or self.dy == 0.0:
            raise ValueError("P0-A dx and dy must be non-zero")


def validate_limits(*, max_gpu_gib: float, max_train_seconds: float) -> None:
    if not 0.0 < max_gpu_gib <= MAX_GPU_GIB:
        raise ValueError(f"max_gpu_gib must be in (0, {MAX_GPU_GIB}]")
    if not 0.0 < max_train_seconds <= MAX_TRAIN_SECONDS:
        raise ValueError(f"max_train_seconds must be in (0, {MAX_TRAIN_SECONDS}]")


def validate_training_protocol(*, micro_batch: int, accumulate: int) -> None:
    if micro_batch < 1 or accumulate < 1:
        raise ValueError("micro-batch and accumulation must both be positive")


def _derivative(value: Tensor, *, dimension: int, spacing: float) -> Tensor:
    result = torch.empty_like(value)
    first = [slice(None)] * value.ndim
    second = [slice(None)] * value.ndim
    first[dimension], second[dimension] = 0, 1
    result[tuple(first)] = (value[tuple(second)] - value[tuple(first)]) / spacing
    last = [slice(None)] * value.ndim
    before_last = [slice(None)] * value.ndim
    last[dimension], before_last[dimension] = -1, -2
    result[tuple(last)] = (value[tuple(last)] - value[tuple(before_last)]) / spacing
    middle = [slice(None)] * value.ndim
    right = [slice(None)] * value.ndim
    left = [slice(None)] * value.ndim
    middle[dimension], right[dimension], left[dimension] = slice(1, -1), slice(2, None), slice(None, -2)
    result[tuple(middle)] = (value[tuple(right)] - value[tuple(left)]) / (2.0 * spacing)
    return result


def build_p0a_features(input_window: Tensor, config: P0AConfig) -> Tensor:
    """Append the frozen 17 causal P0-A features to [B,T,H,W,3] input."""
    if input_window.ndim != 5 or input_window.shape[-1] < 3:
        raise ValueError("input_window must have shape [B,T,H,W,C>=3]")
    if input_window.shape[1] < 2 or input_window.shape[2] < 3 or input_window.shape[3] < 3:
        raise ValueError("P0-A needs T>=2, H>=3, W>=3")
    raw = input_window[..., :3]
    if not torch.is_floating_point(raw) or not bool(torch.isfinite(raw).all()):
        raise ValueError("P0-A requires finite floating-point input")
    u, v = raw[..., 0], raw[..., 1]
    du_dx = _derivative(u, dimension=-1, spacing=config.dx)
    du_dy = _derivative(u, dimension=-2, spacing=config.dy)
    dv_dx = _derivative(v, dimension=-1, spacing=config.dx)
    dv_dy = _derivative(v, dimension=-2, spacing=config.dy)
    vorticity = dv_dx - du_dy
    strain = torch.sqrt(du_dx.square() + dv_dy.square() + 0.5 * (du_dy + dv_dx).square())
    delta_u, delta_v = torch.zeros_like(u), torch.zeros_like(v)
    delta_u[:, 1:] = u[:, 1:] - u[:, :-1]
    delta_v[:, 1:] = v[:, 1:] - v[:, :-1]
    u_mean, v_mean = u.mean(dim=1, keepdim=True), v.mean(dim=1, keepdim=True)
    u_std = torch.sqrt(u.var(dim=1, keepdim=True, unbiased=False))
    v_std = torch.sqrt(v.var(dim=1, keepdim=True, unbiased=False))
    history_tke = 0.5 * (u_std.square() + v_std.square())
    broadcast = lambda value: value.expand_as(u)
    extras = (
        torch.sqrt(u.square() + v.square()), du_dx, du_dy, dv_dx, dv_dy,
        vorticity, vorticity.abs(), strain, delta_u, delta_v,
        broadcast(u_mean), broadcast(v_mean), broadcast(u_std), broadcast(v_std),
        u - broadcast(u_mean), v - broadcast(v_mean), broadcast(history_tke),
    )
    return torch.cat([raw, *(feature.unsqueeze(-1) for feature in extras)], dim=-1)


def expand_cno_input_state(
    source_state: Mapping[str, Tensor], model_state: Mapping[str, Tensor], key: str = LIFT_WEIGHT_KEY
) -> dict[str, Tensor]:
    """Adapt a 3-channel CNO state to a 20-channel CNO without step-zero drift."""
    if key not in source_state or key not in model_state:
        raise KeyError(f"missing CNO lift weight {key}")
    source_weight, target_weight = source_state[key], model_state[key]
    if source_weight.shape[1] != 3 or target_weight.shape[1] != 20:
        raise ValueError(f"expected source/target input channels 3/20, got {source_weight.shape[1]}/{target_weight.shape[1]}")
    if source_weight.shape[0] != target_weight.shape[0] or source_weight.shape[2:] != target_weight.shape[2:]:
        raise ValueError("CNO lift tensor shape is incompatible apart from input channels")
    result = dict(source_state)
    expanded = torch.zeros_like(target_weight)
    expanded[:, :3].copy_(source_weight)
    result[key] = expanded
    return result


def _relative_l2(prediction: Tensor, target: Tensor) -> Tensor:
    pred_flat, target_flat = prediction.flatten(start_dim=1), target.flatten(start_dim=1)
    return (torch.linalg.vector_norm(pred_flat - target_flat, dim=1) /
            torch.linalg.vector_norm(target_flat, dim=1).clamp_min(1e-8)).mean()


def _tke_map(velocity: Tensor) -> Tensor:
    fluctuation = velocity - velocity.mean(dim=1, keepdim=True)
    return 0.5 * fluctuation.square().mean(dim=1).sum(dim=-1)


def _mvpe_loss(prediction: Tensor, target: Tensor) -> Tensor:
    _, _, height, width, _ = prediction.shape
    center_y, interval_y, spacing, center_x = height // 2, min(2, max(1, height // 10)), 16, 10
    probe_y = list(range(center_y - 4 * interval_y, center_y + 5 * interval_y, interval_y))
    probe_y = [index for index in probe_y if 0 <= index < height]
    probe_x = [int(((index + 1) * spacing + center_x) / 2) for index in range(4)]
    terms = [
        _relative_l2(prediction[:, :, probe_y, x, :2].mean(dim=1), target[:, :, probe_y, x, :2].mean(dim=1))
        for x in probe_x if 0 <= x < width
    ]
    return torch.stack(terms).mean() if terms else prediction.new_zeros(())


def n2_loss(prediction: Tensor, target: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    """Exact N2 objective on the two measured velocity channels."""
    if prediction.shape != target.shape or prediction.ndim != 5 or prediction.shape[-1] < 2:
        raise ValueError("N2 expects equal [B,T,H,W,C>=2] prediction and target tensors")
    pred_velocity, target_velocity = prediction[..., :2], target[..., :2]
    parts = {
        "mse": (pred_velocity - target_velocity).square().mean(),
        "tke": _relative_l2(_tke_map(pred_velocity), _tke_map(target_velocity)),
        "rel": _relative_l2(pred_velocity, target_velocity),
        "mvpe": _mvpe_loss(pred_velocity, target_velocity),
    }
    return sum(N2_WEIGHTS[name] * value for name, value in parts.items()), parts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def milestone_checkpoint_path(out_dir: Path, update: int) -> Path:
    if update < 1:
        raise ValueError("milestone update must be positive")
    return out_dir / f"model_update_{update:05d}.pth"


def manifest_paths(manifest: Path, data_root: Path, split: str) -> list[Path]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if split not in payload or not isinstance(payload[split], list):
        raise ValueError(f"manifest lacks list split {split!r}")
    paths = [data_root / row["file"] for row in payload[split]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"manifest files missing: {missing[:3]}")
    return paths


class RealWindowDataset(Dataset):
    """Read 20-to-20 windows lazily from every released top-level u/v H5."""
    def __init__(self, data_root: Path, *, paths: list[Path] | None = None,
                 max_trajectories: int | None = None) -> None:
        self.paths = sorted(paths) if paths is not None else sorted(data_root.glob("*.h5"))
        if max_trajectories is not None:
            self.paths = self.paths[:max_trajectories]
        if not self.paths:
            raise FileNotFoundError(f"no .h5 trajectories under {data_root}")
        self.samples: list[tuple[Path, int]] = []
        self.grid: tuple[float, float] | None = None
        for path in self.paths:
            with h5py.File(path, "r") as handle:
                if "u" not in handle or "v" not in handle:
                    raise KeyError(f"{path} must have top-level u and v datasets")
                frames, height, width = handle["u"].shape
                if handle["v"].shape != (frames, height, width) or (height, width) != (64, 128):
                    raise ValueError(f"unexpected field shapes in {path}")
                if frames < 40:
                    continue
                if "x" not in handle or "y" not in handle:
                    raise KeyError(f"{path} lacks x/y needed to freeze P0-A spacing")
                spacing = (float(handle["x"][0, 2] - handle["x"][0, 0]) / 2.0,
                           float(handle["y"][2, 0] - handle["y"][0, 0]) / 2.0)
                if spacing[0] == 0.0 or spacing[1] == 0.0:
                    raise ValueError(f"zero downsampled grid spacing in {path}")
                if self.grid is None:
                    self.grid = spacing
                elif not np.allclose(self.grid, spacing, rtol=1e-6, atol=1e-9):
                    raise ValueError(f"P0-A grid spacing differs in {path}: {spacing} vs {self.grid}")
                self.samples.extend((path, start) for start in range(0, frames - 40 + 1, 20))
        if not self.samples or self.grid is None:
            raise ValueError("no valid 20-to-20 windows")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        path, start = self.samples[index]
        with h5py.File(path, "r") as handle:
            u = np.asarray(handle["u"][start:start + 40:1, ::2, ::2], dtype=np.float32)
            v = np.asarray(handle["v"][start:start + 40:1, ::2, ::2], dtype=np.float32)
        field = np.zeros((40, 32, 64, 3), dtype=np.float32)
        field[..., 0], field[..., 1] = u, v
        return torch.from_numpy(field[:20]), torch.from_numpy(field[20:])


def _load_p0a_cno(kit_root: Path, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    import sys
    sys.path.insert(0, str(kit_root))
    from rpde_baselines.model.cno import CNO3d
    model = CNO3d(in_dim=20, out_dim=3, out_dim_mult=1, in_size=64, N_layers=3).to(device)
    loaded = torch.load(checkpoint, map_location="cpu")
    state = loaded.get("model_state_dict", loaded)
    state = expand_cno_input_state(state, model.state_dict())
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model


def _forward(model: torch.nn.Module, config: P0AConfig, inputs: Tensor) -> Tensor:
    features = build_p0a_features(inputs, config)
    return model(features.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)


def _save_checkpoint(path: Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                     update: int, config: P0AConfig, metadata: dict) -> None:
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "iteration": update, "feature_set": "P0-A", "feature_config": vars(config),
                "loss_weights": N2_WEIGHTS, "metadata": metadata}, path)


def load_resume_checkpoint(path: Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                           config: P0AConfig) -> int:
    """Restore a same-protocol P0-A run, including AdamW state, from a saved update."""
    payload = torch.load(path, map_location="cpu")
    if payload.get("feature_set") != "P0-A" or payload.get("loss_weights") != N2_WEIGHTS:
        raise ValueError("resume checkpoint is not a compatible P0-A + N2 run")
    saved_config = payload.get("feature_config", {})
    if not np.isclose(saved_config.get("dx"), config.dx) or not np.isclose(saved_config.get("dy"), config.dy):
        raise ValueError("resume checkpoint P0-A grid differs from current data")
    iteration = payload.get("iteration")
    if not isinstance(iteration, int) or iteration < 1:
        raise ValueError("resume checkpoint lacks a positive iteration")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return iteration


@torch.no_grad()
def evaluate_v9_raw(model: torch.nn.Module, config: P0AConfig, dataset: RealWindowDataset,
                    *, batch_size: int, workers: int, device: torch.device, kit_root: Path) -> dict[str, float]:
    import sys
    sys.path.insert(0, str(kit_root))
    import scoring
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        persistent_workers=workers > 0, pin_memory=True)
    predictions, targets = [], []
    model.eval()
    for inputs, target in loader:
        prediction = _forward(model, config, inputs.to(device, non_blocking=True))
        prediction[..., 2] = 0.0
        predictions.append(prediction.cpu().numpy().astype(np.float32))
        targets.append(target.numpy().astype(np.float32))
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    channels = scoring.measured_channels(target)
    return {"rel_l2": float(np.mean(scoring.rel_l2_per_sample(prediction, target, channels))),
            "tke": float(np.mean(scoring.tke_rel_l2_per_sample(prediction, target, channels))),
            "mvpe": float(scoring.mvpe_rel_l2(prediction, target))}


def train(args: argparse.Namespace) -> None:
    validate_limits(max_gpu_gib=args.max_gpu_gib, max_train_seconds=args.max_train_seconds)
    validate_training_protocol(micro_batch=args.micro_batch, accumulate=args.accumulate)
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    milestones = set(args.milestone_updates)
    if any(update < 1 or update > args.updates for update in milestones):
        raise ValueError("milestone updates must be within [1, --updates]")
    if not torch.cuda.is_available():
        raise RuntimeError("full runner requires CUDA")
    args.out_dir.mkdir(parents=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    selected_paths = manifest_paths(args.manifest, args.data_root, "train") if args.manifest else None
    dataset = RealWindowDataset(args.data_root, paths=selected_paths, max_trajectories=args.max_trajectories)
    evaluation = (RealWindowDataset(args.data_root, paths=manifest_paths(args.manifest, args.data_root, args.eval_split))
                  if args.manifest else None)
    config = P0AConfig(dx=dataset.grid[0], dy=dataset.grid[1])
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.micro_batch, shuffle=True, generator=generator,
                        num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=True,
                        drop_last=True)
    device = torch.device("cuda:0")
    # PyTorch 2.2 rejects reset_peak_memory_stats before a CUDA context exists.
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    model = _load_p0a_cno(args.kit_root, args.checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    initial_update = 0
    if args.resume_checkpoint is not None:
        initial_update = load_resume_checkpoint(args.resume_checkpoint, model=model, optimizer=optimizer, config=config)
        if args.updates <= initial_update:
            raise ValueError("--updates must exceed the resume checkpoint iteration")
        # The previous runner did not persist the DataLoader generator state.
        # Use a documented deterministic fresh shuffle rather than silently replaying it.
        generator.manual_seed(args.seed + initial_update)
    max_bytes = int(args.max_gpu_gib * 1024 ** 3)
    metadata = {"run": "T1-COMP-P0A-N2-FULL", "seed": args.seed, "data_root": str(args.data_root),
                "trajectories": len(dataset.paths), "windows": len(dataset), "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": _sha256(args.checkpoint), "kit_root": str(args.kit_root),
                "scorer_sha256": _sha256(args.kit_root / "scoring.py"), "feature_config": vars(config),
                "loss_weights": N2_WEIGHTS, "micro_batch": args.micro_batch, "accumulate": args.accumulate,
                "effective_batch": args.micro_batch * args.accumulate, "updates": args.updates,
                "max_train_seconds": args.max_train_seconds, "max_gpu_gib": args.max_gpu_gib,
                "manifest": None if args.manifest is None else str(args.manifest), "eval_split": args.eval_split,
                "resume_checkpoint": None if args.resume_checkpoint is None else str(args.resume_checkpoint),
                "initial_update": initial_update,
                "sampler_seed": args.seed + initial_update,
                "milestone_updates": sorted(milestones),
                "started_at": time.time()}
    _atomic_json(args.out_dir / "run_metadata.json", metadata)
    stop_requested = False
    def request_stop(_signal: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
    old_int, old_term = signal.signal(signal.SIGINT, request_stop), signal.signal(signal.SIGTERM, request_stop)
    started, iterator, update, stop_reason = time.monotonic(), iter(loader), initial_update, "updates_complete"
    _atomic_json(args.out_dir / "status.json", {"state": "RUNNING", "update": initial_update, "peak_gpu_bytes": 0})
    try:
        for update in range(initial_update + 1, args.updates + 1):
            if stop_requested:
                stop_reason = "signal"; break
            if time.monotonic() - started >= args.max_train_seconds:
                stop_reason = "time_cap"; break
            # Validation runs switch the model to eval(); restore training behavior.
            model.train()
            optimizer.zero_grad(set_to_none=True)
            aggregate = {name: 0.0 for name in N2_WEIGHTS}
            for _ in range(args.accumulate):
                try:
                    inputs, targets = next(iterator)
                except StopIteration:
                    iterator = iter(loader); inputs, targets = next(iterator)
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                loss, parts = n2_loss(_forward(model, config, inputs), targets)
                (loss / args.accumulate).backward()
                for name, value in parts.items(): aggregate[name] += float(value.detach()) / args.accumulate
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            peak = int(torch.cuda.max_memory_allocated(device))
            if peak > max_bytes:
                stop_reason = "memory_cap"; break
            if update % args.save_interval == 0 or update == args.updates:
                _save_checkpoint(args.out_dir / "model_resume.pth", model=model, optimizer=optimizer,
                                 update=update, config=config, metadata=metadata)
                _atomic_json(args.out_dir / "status.json", {"state": "RUNNING", "update": update,
                    "elapsed_seconds": time.monotonic() - started, "peak_gpu_bytes": peak, "loss_parts": aggregate})
                _append_jsonl(args.out_dir / "train_metrics.jsonl", {"update": update,
                    "elapsed_seconds": time.monotonic() - started, "peak_gpu_bytes": peak, "loss_parts": aggregate})
            if update in milestones:
                _save_checkpoint(milestone_checkpoint_path(args.out_dir, update), model=model,
                                 optimizer=optimizer, update=update, config=config, metadata=metadata)
            if evaluation is not None and (update % args.eval_interval == 0 or update == args.updates):
                metrics = evaluate_v9_raw(model, config, evaluation, batch_size=args.micro_batch,
                                          workers=args.workers, device=device, kit_root=args.kit_root)
                _atomic_json(args.out_dir / f"dev_{update:05d}.json", {"update": update, "raw_errors": metrics})
    finally:
        signal.signal(signal.SIGINT, old_int); signal.signal(signal.SIGTERM, old_term)
    peak = int(torch.cuda.max_memory_allocated(device))
    _save_checkpoint(args.out_dir / "model_last.pth", model=model, optimizer=optimizer,
                     update=update, config=config, metadata=metadata)
    _atomic_json(args.out_dir / "status.json", {"state": "DONE" if stop_reason == "updates_complete" else "STOPPED",
        "stop_reason": stop_reason, "update": update, "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_bytes": peak, "memory_cap_bytes": max_bytes})
    if peak > max_bytes:
        raise RuntimeError(f"GPU peak {peak} exceeded cap {max_bytes}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--eval-split", default="dev")
    parser.add_argument("--eval-interval", type=int, default=820)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--milestone-updates", type=int, nargs="*", default=())
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=6800)
    parser.add_argument("--max-train-seconds", type=float, default=MAX_TRAIN_SECONDS)
    parser.add_argument("--max-gpu-gib", type=float, default=12.0)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--max-trajectories", type=int, default=None)
    args = parser.parse_args()
    if args.updates < 1 or args.save_interval < 1 or args.eval_interval < 1 or args.workers < 0:
        parser.error("updates/save-interval/eval-interval must be positive and workers non-negative")
    train(args)


if __name__ == "__main__":
    main()
