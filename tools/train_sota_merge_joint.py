#!/usr/bin/env python3
"""Joint Stage-A + Stage-B + Residual Corrector trainer for SOTA merge.

Starts from the selected Stage-A @57k checkpoint. One optimizer step accumulates:
- existing Stage-A spatial-phase loss,
- existing Stage-B P00 Dense-All loss,
- existing residual-corrected final loss.

No new model architecture is defined here.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import realpde_sota_v2_integrated as strong
import train_sota_merge_backbone as merge
import realpde_sota_merge_joint_runtime as rt
from realpde_p0_data import H5WindowDataset
from realpde_sota_merge_spatial import SpatialPhaseExpandedDataset, SpatialPhaseExpandedSampler

STATUS = "REVIEW_REQUIRED"
SEED = merge.SEED
BATCH_SIZE = merge.BATCH_SIZE
UPDATES = 6_000
MILESTONES = (1_000, 2_000, 3_000, 4_000, 5_000, 6_000)
RECOVERY_EVERY = 500

BACKBONE_LR = 1e-6
CORRECTOR_LR = 1e-5
BACKBONE_WEIGHT_DECAY = 1e-2
CORRECTOR_WEIGHT_DECAY = 1e-5

LOSS_WEIGHT_STAGE_A = 1.0
LOSS_WEIGHT_STAGE_B = 1.0
LOSS_WEIGHT_RESIDUAL = 1.0


def next_stage_a(loader, iterator, sampler, epoch, offset):
    try:
        batch = next(iterator)
    except StopIteration:
        epoch += 1
        offset = 0
        sampler.set_epoch(epoch, start_index=0)
        iterator = iter(loader)
        batch = next(iterator)
    offset += BATCH_SIZE
    if offset == sampler.full_epoch_size:
        epoch += 1
        offset = 0
        sampler.set_epoch(epoch, start_index=0)
        iterator = iter(loader)
    elif offset > sampler.full_epoch_size:
        raise RuntimeError("Stage-A sampler offset overflow")
    return batch, iterator, epoch, offset


def next_stage_b(dataset, loader, iterator, sampler, epoch, offset):
    usable = (len(dataset) // BATCH_SIZE) * BATCH_SIZE
    try:
        batch = next(iterator)
    except StopIteration:
        epoch += 1
        offset = 0
        sampler.set_epoch(epoch, start_index=0)
        iterator = iter(loader)
        batch = next(iterator)
    offset += BATCH_SIZE
    if offset == usable:
        epoch += 1
        offset = 0
        sampler.set_epoch(epoch, start_index=0)
        iterator = iter(loader)
    elif offset > usable:
        raise RuntimeError("Stage-B sampler offset overflow")
    return batch, iterator, epoch, offset


def run(args: argparse.Namespace):
    if args.updates != UPDATES:
        raise ValueError(f"V1 fixes updates at {UPDATES}")
    if tuple(args.milestones) != MILESTONES:
        raise ValueError(f"V1 fixes milestones at {MILESTONES}")
    if args.batch_size != BATCH_SIZE:
        raise ValueError(f"V1 fixes branch batch size at {BATCH_SIZE}")

    for path in (
        args.data_root, args.manifest, args.kit_root,
        args.backbone_checkpoint, args.corrector_checkpoint,
    ):
        merge.assert_safe_path(path)

    if not (args.kit_root / "scoring.py").is_file():
        raise FileNotFoundError(args.kit_root / "scoring.py")
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.resume:
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_paths, dev_paths = merge.resolve_scope(argparse.Namespace(
        scope="clean", manifest=args.manifest, data_root=args.data_root
    ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")
    merge.core.set_seed(SEED)

    _, cfg, builder, backbone = rt.load_backbone(
        args.backbone_checkpoint, args.kit_root, device
    )
    source_backbone_sha = strong.sha256(args.backbone_checkpoint)
    corrector, corrector_source = rt.load_corrector(
        args.corrector_checkpoint, source_backbone_sha, device
    )

    optimizer = torch.optim.AdamW([
        {
            "params": backbone.parameters(),
            "lr": BACKBONE_LR,
            "weight_decay": BACKBONE_WEIGHT_DECAY,
        },
        {
            "params": corrector.parameters(),
            "lr": CORRECTOR_LR,
            "weight_decay": CORRECTOR_WEIGHT_DECAY,
        },
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UPDATES)

    stage_a_ds = SpatialPhaseExpandedDataset(
        train_paths,
        in_steps=20,
        out_steps=20,
        stride=1,
        sub_sample=2,
        include_pressure=False,
        preload_to_ram=args.preload_to_ram,
    )
    if stage_a_ds.base_window_count != merge.CLEAN_DENSE_WINDOWS:
        raise ValueError("Stage-A dense window audit mismatch")
    stage_a_sampler = SpatialPhaseExpandedSampler(stage_a_ds, seed=SEED)
    stage_a_loader = DataLoader(
        stage_a_ds,
        batch_size=BATCH_SIZE,
        sampler=stage_a_sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=bool(args.workers > 0),
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
        drop_last=True,
    )

    stage_b_ds = H5WindowDataset(
        train_paths,
        in_steps=20,
        out_steps=20,
        stride=20,
        sub_sample=2,
        include_pressure=False,
        window_mode="dense_all",
    )
    if len(stage_b_ds) != merge.CLEAN_DENSE_WINDOWS:
        raise ValueError("Stage-B dense window audit mismatch")
    stage_b_sampler = merge.P00DenseSampler(stage_b_ds, seed=SEED)
    stage_b_loader = DataLoader(
        stage_b_ds,
        batch_size=BATCH_SIZE,
        sampler=stage_b_sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=bool(args.workers > 0),
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
        drop_last=True,
    )

    history = []
    start_update = 1
    a_epoch = a_offset = b_epoch = b_offset = 0
    recovery = args.out_dir / "checkpoints" / "runner_latest.pth"

    if args.resume:
        if not recovery.is_file():
            raise FileNotFoundError(recovery)
        p = rt.restore_recovery(recovery, backbone, corrector, optimizer, scheduler)
        if p.get("source_backbone_sha256") != source_backbone_sha:
            raise ValueError("resume source backbone SHA mismatch")
        start_update = int(p["joint_update"]) + 1
        a_epoch = int(p["stage_a_sampler_epoch"])
        a_offset = int(p["stage_a_sampler_offset"])
        b_epoch = int(p["stage_b_sampler_epoch"])
        b_offset = int(p["stage_b_sampler_offset"])
        history = list(p.get("history", []))

    a_epoch, a_offset = merge.normalize_stage_a_sampler(stage_a_sampler, a_epoch, a_offset)
    b_epoch, b_offset = merge.normalize_p00_state(
        stage_b_ds, stage_b_sampler, b_epoch, b_offset
    )
    a_iter = iter(stage_a_loader)
    b_iter = iter(stage_b_loader)

    rt.dump(args.out_dir / "run_config.json", {
        "status": STATUS,
        "protocol": rt.PROTOCOL,
        "source_stage_a_checkpoint": str(args.backbone_checkpoint),
        "source_stage_a_sha256": source_backbone_sha,
        "source_stage_a_update": rt.SOURCE_STAGE_A_UPDATE,
        "corrector_source": corrector_source,
        "stage_a": {
            "sampling": "existing exhaustive P00/P01/P10/P11",
            "loss": "existing strong_loss(extra_rel=0.0)",
        },
        "stage_b": {
            "sampling": "existing P00 Dense-All",
            "loss": f"existing strong_loss(extra_rel={strong.EXTRA_REL})",
        },
        "residual": {
            "architecture": "existing ResidualCorrector3D(42,h64,b2,max_delta=0.04)",
            "projection": "existing spatial_tke_map",
            "loss": f"existing strong_loss(final_pred, extra_rel={strong.EXTRA_REL})",
            "base_pred_detached": False,
        },
        "loss_weights": {"stage_a": 1.0, "stage_b": 1.0, "residual": 1.0},
        "optimizer": {
            "backbone_lr": BACKBONE_LR,
            "corrector_lr": CORRECTOR_LR,
            "backbone_weight_decay": BACKBONE_WEIGHT_DECAY,
            "corrector_weight_decay": CORRECTOR_WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
        },
        "updates": UPDATES,
        "milestones": list(MILESTONES),
        "recovery_every": RECOVERY_EVERY,
        "batch_size_each_branch": BATCH_SIZE,
        "locked_final_accessed": False,
        "private_accessed": False,
        "codabench_accessed": False,
    })

    if not history:
        history.append(rt.evaluate_pair(
            backbone, builder, corrector, dev_paths,
            kit_root=args.kit_root,
            workers=args.workers,
            eval_batch_size=args.eval_batch_size,
            device=device,
            out_dir=args.out_dir / "eval_seen_dev_000000",
            update=0,
            split_name="seen_dev",
        ))
        rt.write_rows(args.out_dir / "aggregate_metrics.csv", history)

    started = time.monotonic()

    for update in range(start_update, UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        backbone.train()
        corrector.train()

        (xa, ya, _, _), a_iter, a_epoch, a_offset = next_stage_a(
            stage_a_loader, a_iter, stage_a_sampler, a_epoch, a_offset
        )
        xa, ya = xa.to(device, non_blocking=True), ya.to(device, non_blocking=True)
        pred_a = strong.forward_mf(backbone, builder, xa)
        loss_a, parts_a = merge.strong_loss(pred_a, ya, extra_rel=0.0)
        if not torch.isfinite(loss_a):
            raise FloatingPointError(f"non-finite Stage-A loss @{update}")
        (LOSS_WEIGHT_STAGE_A * loss_a).backward()
        del pred_a, xa, ya

        (xb, yb, _, _), b_iter, b_epoch, b_offset = next_stage_b(
            stage_b_ds, stage_b_loader, b_iter, stage_b_sampler, b_epoch, b_offset
        )
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        base_pred, final_pred = rt.forward_final(backbone, builder, corrector, xb)
        loss_b, parts_b = merge.strong_loss(base_pred, yb, extra_rel=strong.EXTRA_REL)
        loss_r, parts_r = merge.strong_loss(final_pred, yb, extra_rel=strong.EXTRA_REL)
        if not torch.isfinite(loss_b) or not torch.isfinite(loss_r):
            raise FloatingPointError(f"non-finite Stage-B/Residual loss @{update}")
        (
            LOSS_WEIGHT_STAGE_B * loss_b
            + LOSS_WEIGHT_RESIDUAL * loss_r
        ).backward()

        backbone_grad = torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
        corrector_grad = torch.nn.utils.clip_grad_norm_(corrector.parameters(), 1.0)
        if update == 1:
            if not np.isfinite(float(backbone_grad)):
                raise FloatingPointError("non-finite backbone gradient")
            if not np.isfinite(float(corrector_grad)) or float(corrector_grad) <= 0:
                raise RuntimeError("corrector received no finite non-zero gradient")

        optimizer.step()
        scheduler.step()

        if update == 1 or update % args.log_every == 0:
            print(json.dumps({
                "joint_update": update,
                "loss_a": float(loss_a.detach().cpu()),
                "loss_b": float(loss_b.detach().cpu()),
                "loss_r": float(loss_r.detach().cpu()),
                "loss_total": float(
                    loss_a.detach().cpu() + loss_b.detach().cpu() + loss_r.detach().cpu()
                ),
                "backbone_grad_norm_preclip": float(backbone_grad),
                "corrector_grad_norm_preclip": float(corrector_grad),
                "backbone_lr": float(optimizer.param_groups[0]["lr"]),
                "corrector_lr": float(optimizer.param_groups[1]["lr"]),
                "stage_a_sampler_epoch": a_epoch,
                "stage_a_sampler_offset": a_offset,
                "stage_b_sampler_epoch": b_epoch,
                "stage_b_sampler_offset": b_offset,
                "stage_a_rel_component": float(parts_a["rel"].detach().cpu()),
                "stage_b_rel_component": float(parts_b["rel"].detach().cpu()),
                "residual_rel_component": float(parts_r["rel"].detach().cpu()),
            }, sort_keys=True), flush=True)

        if update in MILESTONES:
            row = rt.evaluate_pair(
                backbone, builder, corrector, dev_paths,
                kit_root=args.kit_root,
                workers=args.workers,
                eval_batch_size=args.eval_batch_size,
                device=device,
                out_dir=args.out_dir / f"eval_seen_dev_{update:06d}",
                update=update,
                split_name="seen_dev",
            )
            row["elapsed_seconds"] = time.monotonic() - started
            history.append(row)
            rt.write_rows(args.out_dir / "aggregate_metrics.csv", history)
            rt.save_model_pair(
                args.out_dir,
                update=update,
                backbone=backbone,
                corrector=corrector,
                cfg=cfg,
                source_backbone_sha=source_backbone_sha,
                corrector_source=corrector_source,
            )

        if update % RECOVERY_EVERY == 0 or update == UPDATES:
            rt.save_recovery(
                recovery,
                backbone=backbone,
                corrector=corrector,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                source_backbone_sha=source_backbone_sha,
                corrector_source=corrector_source,
                update=update,
                stage_a_epoch=a_epoch,
                stage_a_offset=a_offset,
                stage_b_epoch=b_epoch,
                stage_b_offset=b_offset,
                history=history,
            )

    candidates = [r for r in history if int(r["update"]) in MILESTONES]
    selected = max(candidates, key=lambda r: (float(r["final_point"]), -int(r["update"])))
    result = {
        "status": STATUS,
        "protocol": rt.PROTOCOL,
        "source_stage_a_update": rt.SOURCE_STAGE_A_UPDATE,
        "source_backbone_sha256": source_backbone_sha,
        "corrector_source": corrector_source,
        "selected_seen_dev_update": int(selected["update"]),
        "selected_seen_dev": selected,
        "history": history,
        "runtime_seconds_this_process": time.monotonic() - started,
        "locked_final_accessed": False,
        "private_accessed": False,
        "codabench_accessed": False,
        "next_action": (
            "REVIEW_REQUIRED: run the existing AoA10 generalization audit on every "
            "joint milestone pair before choosing a candidate. Do not start SPS/full/final."
        ),
    }
    rt.dump(args.out_dir / "summary.json", result)
    rt.dump(args.out_dir / "status.json", {
        "state": "DONE",
        "status": STATUS,
        "selected_seen_dev_update": result["selected_seen_dev_update"],
    })
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--kit-root", type=Path, required=True)
    p.add_argument("--backbone-checkpoint", type=Path, required=True)
    p.add_argument("--corrector-checkpoint", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--updates", type=int, default=UPDATES)
    p.add_argument("--milestones", type=int, nargs="+", default=list(MILESTONES))
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--preload-to-ram", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--require-cuda", action="store_true")
    args = p.parse_args()
    result = run(args)
    print(json.dumps({
        "status": result["status"],
        "selected_seen_dev_update": result["selected_seen_dev_update"],
        "out_dir": str(args.out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
