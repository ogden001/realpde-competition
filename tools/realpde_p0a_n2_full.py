#!/usr/bin/env python3
"""Resumable P0-A + N2 continuation runner for Track 1.

The runner supports two explicit modes:
- validation continuation from a frozen manifest; or
- all-released-data competition continuation.

It restores model and AdamW state, reapplies the requested LR, preserves exact
milestone checkpoints, and records the last truly completed optimizer update.
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

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

import realpde_loss_official_v9 as core
from realpde_b1_p0a_n2 import evaluate, forward
from realpde_p0_data import H5WindowDataset
from realpde_p0_features import P0FeatureBuilder, P0FeatureConfig


MAX_GPU_GIB = 23.5
MAX_TRAIN_SECONDS = 5 * 60 * 60 + 55 * 60
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


def released_paths(data_root: Path, max_trajectories: int | None = None) -> list[Path]:
    paths = sorted(data_root.glob("*.h5"))
    if max_trajectories is not None:
        if max_trajectories < 1:
            raise ValueError("max_trajectories must be positive")
        paths = paths[:max_trajectories]
    if not paths:
        raise FileNotFoundError(f"no released .h5 trajectories under {data_root}")
    return paths


def historical_p0a_spacing(paths: list[Path]) -> tuple[float, float]:
    """Reproduce the exact grid-spacing semantics used to train full@15,300.

    The historical full runner loaded velocity fields with ``::2`` spatial
    subsampling, but intentionally computed P0-A derivative spacing as
    ``(x[0,2]-x[0,0])/2`` and ``(y[2,0]-y[0,0])/2`` on the original grid.
    For a regular grid this equals the original-grid adjacent spacing, not the
    spacing between adjacent cells after subsampling. Continuations must keep
    this numerical convention because the checkpoint was trained with it.
    """
    if not paths:
        raise ValueError("P0-A spacing requires at least one trajectory")
    reference: tuple[float, float] | None = None
    for path in paths:
        with h5py.File(path, "r") as handle:
            if "x" not in handle or "y" not in handle:
                raise KeyError(f"{path} lacks x/y needed to restore historical P0-A spacing")
            x, y = handle["x"], handle["y"]
            if x.ndim != 2 or y.ndim != 2 or x.shape[1] < 3 or y.shape[0] < 3:
                raise ValueError(f"{path} has invalid x/y grid shape for historical P0-A spacing")
            spacing = (
                float(x[0, 2] - x[0, 0]) / 2.0,
                float(y[2, 0] - y[0, 0]) / 2.0,
            )
        if spacing[0] == 0.0 or spacing[1] == 0.0:
            raise ValueError(f"zero historical P0-A spacing in {path}")
        if reference is None:
            reference = spacing
        elif not np.allclose(reference, spacing, rtol=1e-6, atol=1e-9):
            raise ValueError(f"P0-A spacing differs in {path}: {spacing} vs {reference}")
    assert reference is not None
    return reference


def milestone_checkpoint_path(out_dir: Path, update: int) -> Path:
    if update < 1:
        raise ValueError("milestone update must be positive")
    return out_dir / f"model_update_{update:05d}.pth"


def validate_milestone_updates(values: list[int] | tuple[int, ...], *, start: int, target: int) -> tuple[int, ...]:
    milestones = tuple(sorted(set(int(value) for value in values)))
    if any(value <= start or value > target for value in milestones):
        raise ValueError(f"milestone updates must be within ({start}, {target}]")
    return milestones


def terminal_state(
    *,
    stop_reason: str,
    completed_update: int,
    target_update: int,
    initial_update: int,
    allow_time_cap: bool,
) -> str:
    if stop_reason == "updates_complete" and completed_update == target_update:
        return "DONE"
    if stop_reason == "time_cap" and allow_time_cap and completed_update > initial_update:
        return "TIME_CAPPED"
    return "STOPPED"


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
    if args.all_released_data and args.manifest is not None:
        raise ValueError("--all-released-data and --manifest are mutually exclusive")
    if not args.all_released_data and args.manifest is None:
        raise ValueError("validation continuation requires --manifest")

    if args.all_released_data:
        train_paths = released_paths(args.data_root, args.max_trajectories)
        dev_paths: list[Path] | None = None
    else:
        train_paths = manifest_paths(args.manifest, args.data_root, "train")
        dev_paths = manifest_paths(args.manifest, args.data_root, "dev")

    dx, dy = historical_p0a_spacing(train_paths)
    config = P0FeatureConfig(
        include_p0_a=True,
        include_p0_b=False,
        dx=dx,
        dy=dy,
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
    milestones = validate_milestone_updates(args.milestone_updates, start=initial_update, target=args.updates)

    train_dataset = H5WindowDataset(
        train_paths,
        in_steps=20,
        out_steps=20,
        stride=20,
        sub_sample=2,
        max_windows_per_trajectory=args.max_windows_per_trajectory,
        include_pressure=False,
    )
    if args.expected_trajectories is not None and len(train_paths) != args.expected_trajectories:
        raise ValueError(
            f"training trajectory count {len(train_paths)} != expected {args.expected_trajectories}"
        )
    if args.expected_windows is not None and len(train_dataset) != args.expected_windows:
        raise ValueError(f"training window count {len(train_dataset)} != expected {args.expected_windows}")

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
        "run": "T1-COMP-P0A-N2-FULL-CONTINUATION" if args.all_released_data else "T1-ID-P0A-N2-CONTINUATION",
        "seed": args.seed,
        "all_released_data": bool(args.all_released_data),
        "manifest": None if args.manifest is None else str(args.manifest.resolve()),
        "manifest_sha256": None if args.manifest is None else sha256(args.manifest),
        "scorer_sha256": sha256(args.kit_root / "scoring.py"),
        "resume_checkpoint": str(args.resume_checkpoint.resolve()),
        "resume_checkpoint_sha256": sha256(args.resume_checkpoint),
        "initial_update": initial_update,
        "updates": args.updates,
        "milestone_updates": list(milestones),
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
    completed_update = initial_update
    stop_reason = "updates_complete"
    latest_parts: dict[str, float] = {}
    try:
        for next_update in range(initial_update + 1, args.updates + 1):
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
            completed_update = next_update
            latest_parts = aggregate
            peak = int(torch.cuda.max_memory_allocated(device))
            if peak > max_bytes:
                stop_reason = "memory_cap"
                break
            if completed_update % args.save_interval == 0 or completed_update == args.updates:
                save_checkpoint(
                    args.out_dir / "model_resume.pth",
                    model=model,
                    optimizer=optimizer,
                    update=completed_update,
                    config=config,
                    metadata=metadata,
                )
                append_jsonl(
                    args.out_dir / "train_metrics.jsonl",
                    {
                        "update": completed_update,
                        "elapsed_seconds": time.monotonic() - started,
                        "peak_gpu_bytes": peak,
                        "loss_parts": aggregate,
                    },
                )
                atomic_json(
                    args.out_dir / "status.json",
                    {
                        "state": "RUNNING",
                        "update": completed_update,
                        "elapsed_seconds": time.monotonic() - started,
                        "peak_gpu_bytes": peak,
                    },
                )
            if completed_update in milestones:
                save_checkpoint(
                    milestone_checkpoint_path(args.out_dir, completed_update),
                    model=model,
                    optimizer=optimizer,
                    update=completed_update,
                    config=config,
                    metadata=metadata,
                )
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)

    peak = int(torch.cuda.max_memory_allocated(device))
    save_checkpoint(
        args.out_dir / "model_last.pth",
        model=model,
        optimizer=optimizer,
        update=completed_update,
        config=config,
        metadata=metadata,
    )
    if stop_reason == "updates_complete" and completed_update == args.updates:
        target_checkpoint = milestone_checkpoint_path(args.out_dir, completed_update)
        if not target_checkpoint.exists():
            save_checkpoint(
                target_checkpoint,
                model=model,
                optimizer=optimizer,
                update=completed_update,
                config=config,
                metadata=metadata,
            )
        if dev_paths is not None:
            args.batch_size = args.micro_batch
            args.max_windows = None
            final = evaluate(model, builder, dev_paths, args, device, args.kit_root, args.out_dir / "eval_final")
            atomic_json(
                args.out_dir / f"dev_{completed_update:05d}.json",
                {"update": completed_update, "raw_errors": final["raw_errors"]},
            )

    final_state = terminal_state(
        stop_reason=stop_reason,
        completed_update=completed_update,
        target_update=args.updates,
        initial_update=initial_update,
        allow_time_cap=args.allow_time_cap,
    )
    atomic_json(
        args.out_dir / "status.json",
        {
            "state": final_state,
            "stop_reason": stop_reason,
            "update": completed_update,
            "elapsed_seconds": time.monotonic() - started,
            "peak_gpu_bytes": peak,
            "memory_cap_bytes": max_bytes,
            "last_train_loss_parts": latest_parts,
        },
    )
    if final_state == "STOPPED":
        raise RuntimeError(f"continuation stopped before accepted terminal state: reason={stop_reason}, update={completed_update}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--all-released-data", action="store_true")
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-start-update", type=int)
    parser.add_argument("--expected-trajectories", type=int)
    parser.add_argument("--expected-windows", type=int)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--milestone-updates", type=int, nargs="*", default=())
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--max-train-seconds", type=float, default=5400.0)
    parser.add_argument("--max-gpu-gib", type=float, default=12.0)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--max-windows-per-trajectory", type=int)
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument("--allow-time-cap", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
