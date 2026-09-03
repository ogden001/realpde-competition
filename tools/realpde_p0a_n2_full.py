#!/usr/bin/env python3
"""Resumable P0-A + N2 continuation runner for Track 1.

This is the compact continuation path recovered from the historical full-data
runner. It reuses the current P0-A builder and official-v9 loss/scorer helpers,
restores both model and AdamW state, then explicitly reapplies the requested LR.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import realpde_loss_official_v9 as core
from realpde_b1_p0a_n2 import evaluate, forward
from realpde_p0_data import H5WindowDataset, read_grid
from realpde_p0_features import P0FeatureBuilder, P0FeatureConfig


MAX_GPU_GIB = 23.5
MAX_TRAIN_SECONDS = 5 * 60 * 60 + 40 * 60
# Historical full-run checkpoints store exactly these four non-zero N2 terms.
# Keep this schema stable so old optimizer/model resumes remain compatible.
N2_WEIGHTS = {"mse": 1.0, "tke": 0.05, "rel": 0.027514, "mvpe": 0.009757}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def manifest_paths(manifest: Path, data_root: Path, split: str) -> list[Path]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if split not in payload or not isinstance(payload[split], list):
        raise ValueError(f"manifest lacks list split {split!r}")
    paths = [data_root / row["file"] for row in payload[split]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"manifest files missing: {missing[:3]}")
    return paths


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    update: int,
    config: P0FeatureConfig,
    metadata: dict,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": update,
            "feature_set": "P0-A",
            "feature_config": vars(config),
            "loss_weights": N2_WEIGHTS,
            "metadata": metadata,
        },
        path,
    )


def load_resume_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: P0FeatureConfig,
    lr_override: float | None = None,
) -> int:
    """Restore model/AdamW state and optionally replace only the resumed LR."""
    payload = torch.load(path, map_location="cpu")
    if payload.get("feature_set") != "P0-A" or payload.get("loss_weights") != N2_WEIGHTS:
        raise ValueError("resume checkpoint is not a compatible P0-A + N2 run")
    saved_config = payload.get("feature_config", {})
    if not np.isclose(saved_config.get("dx"), config.dx) or not np.isclose(saved_config.get("dy"), config.dy):
        raise ValueError("resume checkpoint P0-A grid differs from current data")
    iteration = payload.get("iteration")
    if not isinstance(iteration, int) or iteration < 1:
        raise ValueError("resume checkpoint lacks a positive iteration")
    if "optimizer_state_dict" not in payload:
        raise ValueError("resume checkpoint lacks optimizer state")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if lr_override is not None:
        if lr_override <= 0:
            raise ValueError("lr_override must be positive")
        for group in optimizer.param_groups:
            group["lr"] = float(lr_override)
    return iteration


def build_model(kit_root: Path, builder: P0FeatureBuilder, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(kit_root))
    from rpde_baselines.model.cno import CNO3d

    return CNO3d(
        in_dim=len(builder.feature_names),
        out_dim=3,
        out_dim_mult=1,
        in_size=64,
        N_layers=3,
    ).to(device)


def train(args: argparse.Namespace) -> None:
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("continuation runner requires CUDA")
    if not 0 < args.max_gpu_gib <= MAX_GPU_GIB:
        raise ValueError(f"max_gpu_gib must be in (0, {MAX_GPU_GIB}]")
    if not 0 < args.max_train_seconds <= MAX_TRAIN_SECONDS:
        raise ValueError(f"max_train_seconds must be in (0, {MAX_TRAIN_SECONDS}]")
    if min(args.micro_batch, args.accumulate, args.save_interval) < 1 or args.lr <= 0:
        raise ValueError("batch/accumulate/save interval and lr must be positive")
    if not (args.kit_root / "scoring.py").is_file():
        raise FileNotFoundError(args.kit_root / "scoring.py")

    train_paths = manifest_paths(args.manifest, args.data_root, "train")
    dev_paths = manifest_paths(args.manifest, args.data_root, "dev")
    x_grid, y_grid = read_grid(train_paths[0], sub_sample=2)
    config = P0FeatureConfig(
        include_p0_a=True,
        include_p0_b=False,
        dx=float(x_grid[0, 1] - x_grid[0, 0]),
        dy=float(y_grid[1, 0] - y_grid[0, 0]),
    )

    args.out_dir.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0")
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    builder = P0FeatureBuilder(config).to(device)
    model = build_model(args.kit_root, builder, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    initial_update = load_resume_checkpoint(
        args.resume_checkpoint,
        model=model,
        optimizer=optimizer,
        config=config,
        lr_override=args.lr,
    )
    if args.expected_start_update is not None and initial_update != args.expected_start_update:
        raise ValueError(
            f"resume checkpoint is update {initial_update}, expected {args.expected_start_update}"
        )
    if args.updates <= initial_update:
        raise ValueError("--updates must exceed the resume checkpoint iteration")

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
    max_bytes = int(args.max_gpu_gib * 1024**3)
    metadata = {
        "run": "T1-ID-P0A-N2-LR-CONTINUATION",
        "seed": args.seed,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "scorer_sha256": sha256(args.kit_root / "scoring.py"),
        "resume_checkpoint": str(args.resume_checkpoint.resolve()),
        "resume_checkpoint_sha256": sha256(args.resume_checkpoint),
        "initial_update": initial_update,
        "updates": args.updates,
        "lr": float(args.lr),
        "optimizer_lrs": sorted({float(group["lr"]) for group in optimizer.param_groups}),
        "micro_batch": args.micro_batch,
        "accumulate": args.accumulate,
        "effective_batch": args.micro_batch * args.accumulate,
        "workers": args.workers,
        "max_train_seconds": args.max_train_seconds,
        "max_gpu_gib": args.max_gpu_gib,
        "feature_config": vars(config),
        "loss_weights": N2_WEIGHTS,
        "train_trajectories": len(train_paths),
        "dev_trajectories": len(dev_paths),
        "train_windows": len(train_dataset),
        "sampler_seed": args.seed + initial_update,
        "started_at": time.time(),
    }
    atomic_json(args.out_dir / "run_metadata.json", metadata)
    atomic_json(
        args.out_dir / "status.json",
        {"state": "RUNNING", "update": initial_update, "peak_gpu_bytes": 0},
    )

    stop_requested = False

    def request_stop(_signal: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    old_int = signal.signal(signal.SIGINT, request_stop)
    old_term = signal.signal(signal.SIGTERM, request_stop)
    iterator = iter(loader)
    started = time.monotonic()
    update = initial_update
    stop_reason = "updates_complete"
    latest_parts: dict[str, float] = {}
    try:
        for update in range(initial_update + 1, args.updates + 1):
            if stop_requested:
                stop_reason = "signal"
                break
            if time.monotonic() - started >= args.max_train_seconds:
                stop_reason = "time_cap"
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            aggregate = {name: 0.0 for name in N2_WEIGHTS}
            for _ in range(args.accumulate):
                try:
                    inputs, targets, _, _ = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    inputs, targets, _, _ = next(iterator)
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                parts = core.loss_parts(forward(model, builder, inputs), targets)
                loss = sum(N2_WEIGHTS[name] * parts[name] for name in N2_WEIGHTS)
                (loss / args.accumulate).backward()
                for name in N2_WEIGHTS:
                    aggregate[name] += float(parts[name].detach().cpu()) / args.accumulate
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            latest_parts = aggregate
            peak = int(torch.cuda.max_memory_allocated(device))
            if peak > max_bytes:
                stop_reason = "memory_cap"
                break
            if update % args.save_interval == 0 or update == args.updates:
                save_checkpoint(
                    args.out_dir / "model_resume.pth",
                    model=model,
                    optimizer=optimizer,
                    update=update,
                    config=config,
                    metadata=metadata,
                )
                append_jsonl(
                    args.out_dir / "train_metrics.jsonl",
                    {
                        "update": update,
                        "elapsed_seconds": time.monotonic() - started,
                        "peak_gpu_bytes": peak,
                        "loss_parts": aggregate,
                    },
                )
                atomic_json(
                    args.out_dir / "status.json",
                    {
                        "state": "RUNNING",
                        "update": update,
                        "elapsed_seconds": time.monotonic() - started,
                        "peak_gpu_bytes": peak,
                    },
                )
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)

    peak = int(torch.cuda.max_memory_allocated(device))
    save_checkpoint(
        args.out_dir / "model_last.pth",
        model=model,
        optimizer=optimizer,
        update=update,
        config=config,
        metadata=metadata,
    )
    if stop_reason == "updates_complete" and update == args.updates:
        save_checkpoint(
            args.out_dir / f"model_update_{update:05d}.pth",
            model=model,
            optimizer=optimizer,
            update=update,
            config=config,
            metadata=metadata,
        )
        args.batch_size = args.micro_batch
        args.max_windows = None
        final = evaluate(model, builder, dev_paths, args, device, args.kit_root, args.out_dir / "eval_final")
        atomic_json(
            args.out_dir / f"dev_{update:05d}.json",
            {"update": update, "raw_errors": final["raw_errors"]},
        )

    final_state = "DONE" if stop_reason == "updates_complete" and update == args.updates else "STOPPED"
    atomic_json(
        args.out_dir / "status.json",
        {
            "state": final_state,
            "stop_reason": stop_reason,
            "update": update,
            "elapsed_seconds": time.monotonic() - started,
            "peak_gpu_bytes": peak,
            "memory_cap_bytes": max_bytes,
            "last_train_loss_parts": latest_parts,
        },
    )
    if final_state != "DONE":
        raise RuntimeError(f"continuation stopped before target: reason={stop_reason}, update={update}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-start-update", type=int)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--max-train-seconds", type=float, default=5400.0)
    parser.add_argument("--max-gpu-gib", type=float, default=12.0)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--max-windows-per-trajectory", type=int)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
