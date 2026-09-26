#!/usr/bin/env python3
"""Fast, semantics-preserving Full82 residual-corrector continuation.

The scientific recipe is intentionally frozen:
- Full82 / Dense-All stride-1 temporal starts, P00 spatial phase
- batch size 8, FP32
- frozen Full Stage-A @90k backbone
- ResidualCorrector3D(42, hidden=64, blocks=2, max_delta=0.04)
- AdamW(lr=1e-4, weight_decay=1e-5)
- CosineAnnealingLR(T_max=50_000)
- the existing residual-corrector objective and sampler seed

Runtime-only changes:
- preload sub-sampled trajectories into host RAM once
- persistent/prefetched DataLoader workers + pinned nonblocking H2D
- resume the exact Dense-All epoch offset
- synchronize/log loss only every N updates instead of every update

No Dev/AoA/final/Codabench/SPS/Joint action exists in this runner.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import realpde_sota_v2_integrated as sota
import realpde_teammate_residual_transfer as transfer
from realpde_adaptive_probe import ResidualCorrector3D, feature_config_from_checkpoint
from realpde_p0_features import P0FeatureBuilder

STATUS = "REVIEW_REQUIRED"
PROTOCOL = "REALPDE_FULL82_CORRECTOR_FAST_IO_V1"

FULL_TRAJECTORIES = 82
FULL_DENSE_WINDOWS = 66_755
BATCH_SIZE = 8
FINAL_UPDATE = 50_000
CORRECTOR_LR = 1e-4
CORRECTOR_WEIGHT_DECAY = 1e-5
CORRECTOR_SEED = 20260905
TRAIN_SAMPLER_SEED = 20260901
EXPECTED_BACKBONE_UPDATE = 90_000
EXPECTED_BACKBONE_SHA256 = "94d2d2d37110d76fd5104c7e6498b380b022e20f1f1cb621e17ba67165040e6c"
DEFAULT_MILESTONES = (25_000, 30_000, 35_000, 40_000, 45_000, 50_000)
FORBIDDEN_PATH_TOKENS = ("locked-final", "locked_final", "private_test", "private-test")


def _assert_safe_path(path: Path) -> None:
    lowered = str(path).lower().replace("\\", "/")
    if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
        raise ValueError(f"forbidden locked/private path: {path}")


def _sha(path: Path) -> str:
    return sota.sha256(path)


def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    tmp.replace(path)


def _dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _load_backbone(path: Path, kit_root: Path, device: torch.device):
    _assert_safe_path(path)
    if _sha(path) != EXPECTED_BACKBONE_SHA256:
        raise ValueError("Full82 Stage-A @90k SHA mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("iteration", -1)) != EXPECTED_BACKBONE_UPDATE:
        raise ValueError(f"backbone iteration must be {EXPECTED_BACKBONE_UPDATE}")
    if payload.get("stage") != "A" or payload.get("scope") != "full":
        raise ValueError("backbone must be the reviewed Full82 Stage-A checkpoint")
    if payload.get("feature_set") != "P0-A":
        raise ValueError("backbone feature_set must be P0-A")
    cfg = feature_config_from_checkpoint(payload)
    builder = P0FeatureBuilder(cfg).to(device)
    backbone = sota.MF01CNO(kit_root, len(builder.feature_names), device)
    backbone.load_state_dict(payload["model_state_dict"], strict=True)
    transfer.freeze_module(backbone)
    return payload, cfg, builder, backbone


def _load_corrector_resume(path: Path, *, device: torch.device, backbone_sha: str):
    _assert_safe_path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    start_update = int(payload.get("updates", -1))
    if not 0 < start_update < FINAL_UPDATE:
        raise ValueError(f"resume corrector updates must be in (0,{FINAL_UPDATE}), got {start_update}")
    if payload.get("backbone_sha256") != backbone_sha:
        raise ValueError("resume corrector is not bound to the exact Full82 Stage-A backbone")
    for key in ("corrector_state_dict", "optimizer_state_dict", "scheduler_state_dict"):
        if key not in payload:
            raise ValueError(f"resume checkpoint is not optimizer-resumable: missing {key}")
    expected_arch = {"in_channels": 42, "hidden": 64, "blocks": 2, "max_delta": 0.04}
    arch = payload.get("architecture", {})
    for key, value in expected_arch.items():
        if key in arch and arch[key] != value:
            raise ValueError(f"resume corrector architecture mismatch: {key}")

    corrector = ResidualCorrector3D(**expected_arch).to(device)
    corrector.load_state_dict(payload["corrector_state_dict"], strict=True)
    optimizer = torch.optim.AdamW(
        corrector.parameters(), lr=CORRECTOR_LR, weight_decay=CORRECTOR_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINAL_UPDATE)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])

    if int(getattr(scheduler, "T_max", -1)) != FINAL_UPDATE:
        raise ValueError(f"resume scheduler T_max must equal {FINAL_UPDATE}")
    if int(scheduler.last_epoch) != start_update:
        raise ValueError(
            f"resume scheduler last_epoch={scheduler.last_epoch} != updates={start_update}; "
            "refuse an inexact LR continuation"
        )
    return payload, corrector, optimizer, scheduler, start_update


def _sampler_position(completed_updates: int, dense_windows: int) -> tuple[int, int, int]:
    usable = (dense_windows // BATCH_SIZE) * BATCH_SIZE
    steps_per_epoch = usable // BATCH_SIZE
    epoch, step_in_epoch = divmod(int(completed_updates), steps_per_epoch)
    return epoch, step_in_epoch * BATCH_SIZE, steps_per_epoch


def _save_checkpoint(
    path: Path,
    *,
    corrector,
    optimizer,
    scheduler,
    update: int,
    sampler_epoch: int,
    sampler_offset: int,
    backbone_path: Path,
    cache_summary: dict[str, object],
) -> dict[str, object]:
    payload = {
        "protocol": PROTOCOL,
        "corrector_state_dict": corrector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "backbone_update": EXPECTED_BACKBONE_UPDATE,
        "backbone_sha256": _sha(backbone_path),
        "updates": int(update),
        "epoch": int(sampler_epoch),
        "sampler_offset": int(sampler_offset),
        "architecture": {"in_channels": 42, "hidden": 64, "blocks": 2, "max_delta": 0.04},
        "optimizer": {
            "name": "AdamW",
            "lr": CORRECTOR_LR,
            "weight_decay": CORRECTOR_WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": FINAL_UPDATE,
        },
        "loss_weights": transfer.CORRECTOR_LOSS_WEIGHTS,
        "seed": CORRECTOR_SEED,
        "train_sampler_seed": TRAIN_SAMPLER_SEED,
        "batch_size": BATCH_SIZE,
        "dense_train_windows": FULL_DENSE_WINDOWS,
        "cache_summary": cache_summary,
    }
    _atomic_torch_save(payload, path)
    return {
        "update": int(update),
        "path": str(path),
        "sha256": _sha(path),
        "sampler_epoch": int(sampler_epoch),
        "sampler_offset": int(sampler_offset),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    for p in (args.data_root, args.kit_root, args.backbone_checkpoint, args.resume_corrector, args.out_dir):
        _assert_safe_path(p)
    if not args.kit_root.joinpath("scoring.py").is_file():
        raise FileNotFoundError(args.kit_root / "scoring.py")
    train_paths = sorted(args.data_root.glob("*.h5"))
    if len(train_paths) != FULL_TRAJECTORIES:
        raise ValueError(f"Full82 requires {FULL_TRAJECTORIES} H5 files, got {len(train_paths)}")
    if args.batch_size != BATCH_SIZE:
        raise ValueError(f"batch size is frozen at {BATCH_SIZE}")
    if args.final_update != FINAL_UPDATE:
        raise ValueError(f"final update is frozen at {FINAL_UPDATE}")
    milestones = tuple(int(x) for x in args.milestones)
    if any(x <= 0 or x > FINAL_UPDATE for x in milestones):
        raise ValueError("invalid milestone")
    if args.workers < 0 or args.prefetch_factor < 1:
        raise ValueError("invalid DataLoader settings")

    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.resume_run:
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(CORRECTOR_SEED)
    np.random.seed(CORRECTOR_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CORRECTOR_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")

    _, cfg, builder, backbone = _load_backbone(args.backbone_checkpoint, args.kit_root, device)
    backbone_sha = _sha(args.backbone_checkpoint)
    resume_payload, corrector, optimizer, scheduler, start_update = _load_corrector_resume(
        args.resume_corrector, device=device, backbone_sha=backbone_sha
    )

    loader_args = argparse.Namespace(
        seed=TRAIN_SAMPLER_SEED,
        micro_batch=BATCH_SIZE,
        workers=args.workers,
        preload_to_ram=True,
        prefetch_factor=args.prefetch_factor,
    )
    dataset, sampler, loader = sota.dense_loader(train_paths, loader_args)
    if len(dataset) != FULL_DENSE_WINDOWS:
        raise ValueError(f"Dense-All windows {len(dataset)} != {FULL_DENSE_WINDOWS}")
    cache_summary = dataset.cache_summary()
    if not cache_summary["enabled"] or int(cache_summary["trajectories"]) != FULL_TRAJECTORIES:
        raise RuntimeError("RAM cache did not cover all Full82 trajectories")

    sampler_epoch, sampler_offset, steps_per_epoch = _sampler_position(start_update, len(dataset))
    sampler.set_epoch(sampler_epoch, start_index=sampler_offset)
    iterator = iter(loader)

    config = {
        "status": STATUS,
        "protocol": PROTOCOL,
        "resume_checkpoint": str(args.resume_corrector),
        "resume_checkpoint_sha256": _sha(args.resume_corrector),
        "resume_update": start_update,
        "resume_checkpoint_epoch_field": resume_payload.get("epoch"),
        "derived_sampler_epoch": sampler_epoch,
        "derived_sampler_offset": sampler_offset,
        "steps_per_dense_epoch": steps_per_epoch,
        "final_update": FINAL_UPDATE,
        "batch_size": BATCH_SIZE,
        "precision": "FP32_UNCHANGED",
        "backbone_checkpoint": str(args.backbone_checkpoint),
        "backbone_sha256": backbone_sha,
        "feature_config": vars(cfg),
        "optimizer": "AdamW",
        "corrector_lr": CORRECTOR_LR,
        "weight_decay": CORRECTOR_WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": FINAL_UPDATE,
        "sampler_seed": TRAIN_SAMPLER_SEED,
        "workers": args.workers,
        "prefetch_factor": args.prefetch_factor,
        "cache": cache_summary,
        "log_every": args.log_every,
        "runtime_only_optimization": True,
        "locked_final_accessed": False,
        "codabench_accessed": False,
        "auto_joint": False,
        "auto_sps": False,
    }
    _dump(args.out_dir / "run_config.json", config)

    # One no-step preflight batch proves the cached loader and training graph work.
    x0, y0, _, _ = next(iterator)
    x0 = x0.to(device, non_blocking=True)
    y0 = y0.to(device, non_blocking=True)
    with torch.no_grad():
        base0 = sota.forward_mf(backbone, builder, x0)
    corrector.train()
    delta0 = transfer._delta_from_corrector(corrector, x0, base0)
    loss0, _ = transfer.corrector_objective(base0, y0, delta0)
    optimizer.zero_grad(set_to_none=True)
    loss0.backward()
    grad0 = torch.nn.utils.clip_grad_norm_(corrector.parameters(), 1.0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    preflight = {
        "status": "PASS" if torch.isfinite(loss0).item() and np.isfinite(float(grad0)) else "FAIL",
        "loss": float(loss0.detach().cpu()),
        "grad_norm_preclip": float(grad0),
        "optimizer_steps": 0,
        "cache": cache_summary,
    }
    optimizer.zero_grad(set_to_none=True)
    _dump(args.out_dir / "preflight.json", preflight)
    if preflight["status"] != "PASS":
        raise RuntimeError("fast corrector preflight failed")
    if args.preflight_only:
        return {"status": STATUS, "preflight": preflight, "config": config}

    # Preflight consumed a loader batch but did not optimize. Restore the exact
    # sampler offset before the first real continuation update.
    sampler.set_epoch(sampler_epoch, start_index=sampler_offset)
    iterator = iter(loader)

    log_path = args.out_dir / "train_fast.jsonl"
    mode = "a" if args.resume_run else "w"
    started = time.monotonic()
    process_start_update = start_update
    saved: list[dict[str, object]] = []

    with log_path.open(mode, encoding="utf-8") as log:
        for update in range(start_update + 1, FINAL_UPDATE + 1):
            try:
                x, y, _, _ = next(iterator)
            except StopIteration:
                sampler_epoch += 1
                sampler_offset = 0
                sampler.set_epoch(sampler_epoch, start_index=0)
                iterator = iter(loader)
                x, y, _, _ = next(iterator)

            sampler_offset += BATCH_SIZE
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.no_grad():
                base = sota.forward_mf(backbone, builder, x)
            corrector.train()
            delta = transfer._delta_from_corrector(corrector, x, base)
            total, parts = transfer.corrector_objective(base, y, delta)

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(corrector.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            should_log = (
                update == start_update + 1
                or update % args.log_every == 0
                or update in milestones
                or update == FINAL_UPDATE
            )
            if should_log:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = max(time.monotonic() - started, 1e-9)
                completed_here = update - process_start_update
                row = {
                    "update": update,
                    "epoch": sampler_epoch,
                    "sampler_offset": sampler_offset,
                    "loss": float(total.detach().cpu()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "updates_per_second": float(completed_here / elapsed),
                    **{f"loss_{name}": float(value.detach().cpu()) for name, value in parts.items()},
                }
                log.write(json.dumps(row, sort_keys=True) + "\n")
                log.flush()
                print(json.dumps(row, sort_keys=True), flush=True)
                if not np.isfinite(row["loss"]):
                    raise FloatingPointError(f"non-finite corrector loss at update={update}")

            if update in milestones or update == FINAL_UPDATE:
                ckpt = args.out_dir / "checkpoints" / f"corrector_update_{update}.pth"
                saved.append(_save_checkpoint(
                    ckpt,
                    corrector=corrector,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    update=update,
                    sampler_epoch=sampler_epoch,
                    sampler_offset=sampler_offset,
                    backbone_path=args.backbone_checkpoint,
                    cache_summary=cache_summary,
                ))

            if args.recovery_every > 0 and update % args.recovery_every == 0:
                _save_checkpoint(
                    args.out_dir / "checkpoints" / "corrector_latest.pth",
                    corrector=corrector,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    update=update,
                    sampler_epoch=sampler_epoch,
                    sampler_offset=sampler_offset,
                    backbone_path=args.backbone_checkpoint,
                    cache_summary=cache_summary,
                )

    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.monotonic() - started
    summary = {
        "status": STATUS,
        "protocol": PROTOCOL,
        "resume_update": start_update,
        "completed_update": FINAL_UPDATE,
        "updates_this_process": FINAL_UPDATE - start_update,
        "runtime_seconds_this_process": wall,
        "updates_per_second": (FINAL_UPDATE - start_update) / max(wall, 1e-9),
        "cache": cache_summary,
        "milestones": saved,
        "next_action": "REVIEW_REQUIRED: verify final checkpoint, then start the frozen Full Joint V1 stage.",
    }
    _dump(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--kit-root", type=Path, required=True)
    p.add_argument("--backbone-checkpoint", type=Path, required=True)
    p.add_argument("--resume-corrector", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--final-update", type=int, default=FINAL_UPDATE)
    p.add_argument("--milestones", type=int, nargs="+", default=list(DEFAULT_MILESTONES))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--recovery-every", type=int, default=1000)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--resume-run", action="store_true")
    p.add_argument("--require-cuda", action="store_true")
    result = run(p.parse_args())
    print(json.dumps({"status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
