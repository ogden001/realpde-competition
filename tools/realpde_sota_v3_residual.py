#!/usr/bin/env python3
"""SOTA-V3 residual-corrector stage.

The backbone is frozen. The corrector keeps the validated teammate-style
42ch/h64/b2/max_delta=0.04 architecture and frozen objective. Dev mode trains
one 30k campaign, evaluates once after training, and fixes the final inference
path to spatial_tke_map projection. Full mode maps the validated 30k corrector
budget by Dense-All epoch exposure and performs no full-label model selection.

No alpha/projection sweep, SPS, locked-final/private data, package, or
Codabench path exists here.
"""
from __future__ import annotations

import argparse
import json
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

import realpde_residual_corrector_projection as projection
import realpde_sota_v2_full as full
import realpde_sota_v2_integrated as base
import realpde_teammate_residual_transfer as transfer
from realpde_adaptive_probe import (
    ResidualCorrector3D, feature_config_from_checkpoint
)
from realpde_p0_features import P0FeatureBuilder

DEV_UPDATES = 30_000
BATCH_SIZE = 8
LR = transfer.CORRECTOR_LR
WEIGHT_DECAY = transfer.CORRECTOR_WEIGHT_DECAY
SEED = transfer.CORRECTOR_SEED
TRAIN_SAMPLER_SEED = transfer.TRAIN_SAMPLER_SEED
RECOVERY_EVERY = 1_000


def full_updates(dev_updates: int = DEV_UPDATES) -> int:
    if dev_updates < 1:
        raise ValueError("dev_updates must be positive")
    return int(round(
        dev_updates
        * full.FULL_UPDATES_PER_EPOCH
        / full.REFERENCE_UPDATES_PER_EPOCH
    ))


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _prepare_out(out_dir: Path, resume: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not resume:
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _load_backbone(path: Path, kit_root: Path, device: torch.device):
    payload = torch.load(
        path, map_location="cpu", weights_only=False
    )
    if payload.get("feature_set") != "P0-A" or "model_state_dict" not in payload:
        raise ValueError(
            "backbone checkpoint is not a P0-A MF checkpoint"
        )
    config = feature_config_from_checkpoint(payload)
    builder = P0FeatureBuilder(config).to(device)
    model = base.MF01CNO(
        kit_root, len(builder.feature_names), device
    )
    model.load_state_dict(
        payload["model_state_dict"], strict=True
    )
    transfer.freeze_module(model)
    return payload, config, builder, model


def _save_state(
    path: Path, *, corrector, optimizer, scheduler,
    update: int, epoch: int, batches_in_epoch: int,
    backbone_checkpoint: Path, backbone_iteration: int,
    dense_windows: int, mode: str,
) -> None:
    payload = {
        "corrector_state_dict": corrector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "updates": int(update),
        "epoch": int(epoch),
        "batches_in_epoch": int(batches_in_epoch),
        "backbone_sha256": base.sha256(backbone_checkpoint),
        "backbone_iteration": int(backbone_iteration),
        "architecture": {
            "in_channels": 42,
            "hidden": 64,
            "blocks": 2,
            "max_delta": 0.04,
        },
        "loss_weights": transfer.CORRECTOR_LOSS_WEIGHTS,
        "optimizer": {
            "name": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
        },
        "seed": SEED,
        "train_sampler_seed": TRAIN_SAMPLER_SEED,
        "batch_size": BATCH_SIZE,
        "dense_train_windows": int(dense_windows),
        "mode": mode,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = (
            torch.cuda.get_rng_state_all()
        )
    torch.save(payload, path)


def _restore_state(
    path: Path, *, corrector, optimizer, scheduler
) -> dict:
    payload = torch.load(
        path, map_location="cpu", weights_only=False
    )
    corrector.load_state_dict(
        payload["corrector_state_dict"], strict=True
    )
    optimizer.load_state_dict(
        payload["optimizer_state_dict"]
    )
    scheduler.load_state_dict(
        payload["scheduler_state_dict"]
    )
    torch.set_rng_state(payload["torch_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    if (
        torch.cuda.is_available()
        and "cuda_rng_state_all" in payload
    ):
        torch.cuda.set_rng_state_all(
            payload["cuda_rng_state_all"]
        )
    return payload


def _resume_iterator(
    loader, sampler, epoch: int, batches_in_epoch: int
):
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    for _ in range(batches_in_epoch):
        try:
            next(iterator)
        except StopIteration as exc:
            raise RuntimeError(
                "saved batches_in_epoch exceeds loader length"
            ) from exc
    return iterator


def train_corrector(
    *, args, mode: str, train_paths: list[Path],
    backbone_checkpoint: Path, backbone_iteration: int,
    builder, backbone, device: torch.device,
    out_dir: Path, updates: int,
) -> Path:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    corrector = ResidualCorrector3D(
        in_channels=42, hidden=64, blocks=2, max_delta=0.04
    ).to(device)
    optimizer = torch.optim.AdamW(
        corrector.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=updates
    )

    loader_args = Namespace(
        seed=TRAIN_SAMPLER_SEED,
        micro_batch=BATCH_SIZE,
        workers=args.workers,
    )
    train_ds, sampler, loader = base.dense_loader(
        train_paths, loader_args
    )
    state_path = (
        out_dir / "checkpoints" / "corrector_runner_latest.pth"
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    start, epoch, batches_in_epoch = 1, 0, 0
    if args.resume and state_path.is_file():
        saved = _restore_state(
            state_path,
            corrector=corrector,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if saved.get("backbone_sha256") != base.sha256(
            backbone_checkpoint
        ):
            raise ValueError(
                "resume corrector belongs to another backbone"
            )
        start = int(saved["updates"]) + 1
        epoch = int(saved.get("epoch", 0))
        batches_in_epoch = int(
            saved.get("batches_in_epoch", 0)
        )

    iterator = _resume_iterator(
        loader, sampler, epoch, batches_in_epoch
    )
    log_path = out_dir / "train_corrector.jsonl"
    log_mode = "a" if start > 1 else "w"
    started = time.monotonic()

    with log_path.open(log_mode, encoding="utf-8") as log:
        for step in range(start, updates + 1):
            try:
                x, y, _, _ = next(iterator)
                batches_in_epoch += 1
            except StopIteration:
                epoch += 1
                batches_in_epoch = 0
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                x, y, _, _ = next(iterator)
                batches_in_epoch = 1

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                base_prediction = base.forward_mf(
                    backbone, builder, x
                )

            corrector.train()
            delta = transfer._delta_from_corrector(
                corrector, x, base_prediction
            )
            loss, parts = transfer.corrector_objective(
                base_prediction, y, delta
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite residual loss @ {step}"
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                corrector.parameters(), 1.0
            )
            optimizer.step()
            scheduler.step()

            if step % args.log_every == 0 or step == updates:
                row = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "lr": float(
                        optimizer.param_groups[0]["lr"]
                    ),
                    **{
                        f"loss_{k}": float(v.detach().cpu())
                        for k, v in parts.items()
                    },
                }
                log.write(
                    json.dumps(row, sort_keys=True) + "\n"
                )
                log.flush()
                print(
                    json.dumps({"mode": mode, **row}),
                    flush=True,
                )

            if (
                step % args.recovery_every == 0
                or step == updates
            ):
                _save_state(
                    state_path,
                    corrector=corrector,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    update=step,
                    epoch=epoch,
                    batches_in_epoch=batches_in_epoch,
                    backbone_checkpoint=backbone_checkpoint,
                    backbone_iteration=backbone_iteration,
                    dense_windows=len(train_ds),
                    mode=mode,
                )

    final_path = (
        out_dir
        / "checkpoints"
        / f"corrector_update_{updates}.pth"
    )
    final_payload = torch.load(
        state_path, map_location="cpu", weights_only=False
    )
    final_payload["training_wall_seconds_last_process"] = (
        time.monotonic() - started
    )
    torch.save(final_payload, final_path)
    dump(out_dir / "training_summary.json", {
        "status": "REVIEW_REQUIRED",
        "mode": mode,
        "updates": updates,
        "dense_train_windows": len(train_ds),
        "effective_dense_epochs": (
            updates * BATCH_SIZE / len(train_ds)
        ),
        "checkpoint": str(final_path),
        "checkpoint_sha256": base.sha256(final_path),
        "backbone_sha256": base.sha256(
            backbone_checkpoint
        ),
    })
    return final_path


def _load_corrector(
    path: Path, device: torch.device,
    expected_backbone_sha: str,
):
    payload = torch.load(
        path, map_location="cpu", weights_only=False
    )
    if payload.get("backbone_sha256") != expected_backbone_sha:
        raise ValueError("corrector/backbone SHA mismatch")
    model = ResidualCorrector3D(
        in_channels=42, hidden=64, blocks=2, max_delta=0.04
    ).to(device)
    model.load_state_dict(
        payload["corrector_state_dict"], strict=True
    )
    transfer.freeze_module(model)
    return model, payload


@torch.no_grad()
def evaluate_final(
    *, args, backbone_checkpoint: Path,
    corrector_checkpoint: Path, dev_paths: list[Path],
    builder, backbone, device: torch.device,
    out_dir: Path,
) -> dict:
    corrector, _ = _load_corrector(
        corrector_checkpoint,
        device,
        base.sha256(backbone_checkpoint),
    )
    ds, loader = base.dev_loader(
        dev_paths,
        Namespace(
            eval_batch_size=args.eval_batch_size,
            workers=args.workers,
        ),
    )
    chunks = {"base": [], "final": []}
    targets = []
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        base_prediction = base.forward_mf(
            backbone, builder, x
        )
        delta = transfer._delta_from_corrector(
            corrector, x, base_prediction
        )
        corrected = transfer.apply_scaled_correction(
            base_prediction, delta, 1.0
        )
        final_prediction = (
            projection.spatial_tke_map_projection(
                base_prediction, corrected
            )
        )
        chunks["base"].append(
            base_prediction.cpu().numpy().astype(np.float32)
        )
        chunks["final"].append(
            final_prediction.cpu().numpy().astype(np.float32)
        )
        targets.append(y.numpy().astype(np.float32))

    target = np.concatenate(targets)
    predictions = {
        k: np.concatenate(v)
        for k, v in chunks.items()
    }
    rows = []
    trajectory_long = []
    for name, prediction_np in predictions.items():
        raw = transfer.raw_physical_errors(
            args.kit_root, prediction_np, target
        )
        trajectory, anatomy = base.core.trajectory_rows(
            ds, prediction_np, target, args.kit_root
        )
        for row in trajectory:
            trajectory_long.append(
                {"variant": name, **row}
            )
        base.dump(
            out_dir / f"trajectory_anatomy_{name}.json",
            anatomy,
        )
        base.dump(
            out_dir / f"horizon_error_summary_{name}.json",
            base.horizon_error_summary(
                prediction_np, target
            ),
        )
        rows.append({"variant": name, **raw})

    baseline = next(
        row for row in rows if row["variant"] == "base"
    )
    final_row = next(
        row for row in rows if row["variant"] == "final"
    )
    for metric in ("rel_l2", "tke", "mvpe"):
        final_row[f"{metric}_improvement_vs_base"] = (
            1.0 - final_row[metric] / baseline[metric]
        )
    transfer.write_rows(
        out_dir / "physical_metrics.csv", rows
    )
    transfer.write_rows(
        out_dir / "trajectory_metrics_long.csv",
        trajectory_long,
    )
    result = {
        "base": baseline,
        "final": final_row,
        "projection": "spatial_tke_map",
        "dev_windows": len(ds),
        "dev_trajectories": len(
            set(ref.path.name for ref in ds.refs)
        ),
    }
    dump(out_dir / "evaluation_summary.json", result)
    return result


def run(args: argparse.Namespace) -> dict:
    _prepare_out(args.out_dir, args.resume)
    if not (args.kit_root / "scoring.py").is_file():
        raise FileNotFoundError(
            args.kit_root / "scoring.py"
        )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")

    (
        backbone_payload, config, builder, backbone
    ) = _load_backbone(
        args.backbone_checkpoint, args.kit_root, device
    )
    backbone_iteration = int(
        backbone_payload.get("iteration", -1)
    )

    if args.mode == "dev":
        if (
            args.manifest is None
            or base.sha256(args.manifest) != base.MANIFEST_SHA
        ):
            raise ValueError(
                "dev mode requires frozen 50/16 manifest"
            )
        train_paths = base.split_paths(
            args.manifest, "train", args.data_root
        )
        dev_paths = base.split_paths(
            args.manifest, "dev", args.data_root
        )
        updates = DEV_UPDATES
    else:
        train_paths = full.released_paths(args.data_root)
        dev_paths = []
        updates = full_updates()

    metadata = {
        "recipe": (
            "frozen V3 backbone + teammate residual "
            "corrector + spatial_tke_map projection"
        ),
        "mode": args.mode,
        "backbone_iteration": backbone_iteration,
        "backbone_sha256": base.sha256(
            args.backbone_checkpoint
        ),
        "feature_config": vars(config),
        "corrector_updates": updates,
        "corrector_architecture": {
            "in_channels": 42,
            "hidden": 64,
            "blocks": 2,
            "max_delta": 0.04,
        },
        "loss_weights": transfer.CORRECTOR_LOSS_WEIGHTS,
        "projection": "spatial_tke_map",
        "alpha": 1.0,
        "no_sweep": True,
        "locked_final_accessed": False,
        "codabench_accessed": False,
    }
    dump(args.out_dir / "run_metadata.json", metadata)

    corrector_checkpoint = train_corrector(
        args=args,
        mode=args.mode,
        train_paths=train_paths,
        backbone_checkpoint=args.backbone_checkpoint,
        backbone_iteration=backbone_iteration,
        builder=builder,
        backbone=backbone,
        device=device,
        out_dir=args.out_dir,
        updates=updates,
    )

    evaluation = None
    if args.mode == "dev":
        evaluation = evaluate_final(
            args=args,
            backbone_checkpoint=args.backbone_checkpoint,
            corrector_checkpoint=corrector_checkpoint,
            dev_paths=dev_paths,
            builder=builder,
            backbone=backbone,
            device=device,
            out_dir=args.out_dir,
        )

    result = {
        "status": "REVIEW_REQUIRED",
        "mode": args.mode,
        "backbone_checkpoint": str(
            args.backbone_checkpoint
        ),
        "backbone_sha256": base.sha256(
            args.backbone_checkpoint
        ),
        "corrector_checkpoint": str(
            corrector_checkpoint
        ),
        "corrector_sha256": base.sha256(
            corrector_checkpoint
        ),
        "updates": updates,
        "evaluation": evaluation,
        "projection": "spatial_tke_map",
    }
    dump(args.out_dir / "summary.json", result)
    dump(args.out_dir / "status.json", {
        "state": "DONE",
        "status": "REVIEW_REQUIRED",
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("dev", "full"), required=True
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--data-root", type=Path, required=True
    )
    parser.add_argument(
        "--kit-root", type=Path, required=True
    )
    parser.add_argument(
        "--backbone-checkpoint", type=Path, required=True
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--eval-batch-size", type=int, default=8
    )
    parser.add_argument(
        "--recovery-every",
        type=int,
        default=RECOVERY_EVERY,
    )
    parser.add_argument(
        "--log-every", type=int, default=100
    )
    parser.add_argument(
        "--resume", action="store_true"
    )
    parser.add_argument(
        "--require-cuda", action="store_true"
    )
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(
        result, indent=2, sort_keys=True, default=str
    ))


if __name__ == "__main__":
    main()
