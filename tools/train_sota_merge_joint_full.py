#!/usr/bin/env python3
"""Full82 Joint V1 adapter for the reviewed Clean Joint training recipe.

This is deliberately a thin adapter around the already-reviewed Clean Joint
components. It changes only the data scope / source checkpoints and uses an
exposure-aligned cosine horizon. It never evaluates on Seen-Dev/AoA/final and
never starts SPS or packaging automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import realpde_sota_v2_integrated as strong
import realpde_sota_merge_joint_runtime as rt
import realpde_teammate_residual_transfer as transfer
import train_sota_merge_backbone as merge
from realpde_adaptive_probe import ResidualCorrector3D, feature_config_from_checkpoint
from realpde_p0_data import H5WindowDataset
from realpde_p0_features import P0FeatureBuilder
from realpde_sota_merge_spatial import SpatialPhaseExpandedDataset, SpatialPhaseExpandedSampler

STATUS = "REVIEW_REQUIRED"
PROTOCOL = "REALPDE_SOTA_MERGE_FULL_JOINT_V1"
SEED = merge.SEED
BATCH_SIZE = merge.BATCH_SIZE
SOURCE_STAGE_A_UPDATE = 90_000
SOURCE_CORRECTOR_UPDATE = 20_000
UPDATES = 6_500
MILESTONES = (5_000, 6_000, 6_500)
RECOVERY_EVERY = 500

# Clean Joint used T_max=6000. Scale the schedule horizon by the Full82/Clean51
# Dense-All exposure ratio so Full@6500 sits at almost the same cosine progress
# as Clean@4000, the Clean curve's early optimum region.
SCHEDULER_T_MAX = 9_700
CLEAN_REFERENCE_UPDATES = 6_000
CLEAN_REFERENCE_SELECTED_UPDATE = 4_000

BACKBONE_LR = 1e-6
CORRECTOR_LR = 1e-5
BACKBONE_WEIGHT_DECAY = 1e-2
CORRECTOR_WEIGHT_DECAY = 1e-5

HOST_RSS_LIMIT_BYTES = 48 * 1024**3
GPU_PEAK_LIMIT_BYTES = 24 * 1024**3


def _dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _atomic_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{__import__('os').getpid()}")
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_full_backbone(path: Path, kit_root: Path, device: torch.device):
    merge.assert_safe_path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("iteration", -1)) != SOURCE_STAGE_A_UPDATE:
        raise ValueError(f"requires Full Stage-A @{SOURCE_STAGE_A_UPDATE}")
    if payload.get("stage") != "A" or payload.get("scope") != "full":
        raise ValueError("source must be a Full82 Stage-A checkpoint")
    if payload.get("feature_set") != "P0-A":
        raise ValueError("source must use P0-A")
    cfg = feature_config_from_checkpoint(payload)
    builder = P0FeatureBuilder(cfg).to(device)
    model = strong.MF01CNO(kit_root, len(builder.feature_names), device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return payload, cfg, builder, model


def _load_full_corrector(path: Path, source_backbone_sha: str, device: torch.device):
    merge.assert_safe_path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("updates", -1)) != SOURCE_CORRECTOR_UPDATE:
        raise ValueError(f"requires Full82 corrector @{SOURCE_CORRECTOR_UPDATE}")
    if payload.get("backbone_sha256") != source_backbone_sha:
        raise ValueError("Full82 corrector must be bound to the exact Full Stage-A checkpoint")
    arch = payload.get("architecture", {})
    expected = {"in_channels": 42, "hidden": 64, "blocks": 2, "max_delta": 0.04}
    for key, value in expected.items():
        if key in arch and arch[key] != value:
            raise ValueError(f"corrector architecture mismatch: {key}")
    model = ResidualCorrector3D(**expected).to(device)
    model.load_state_dict(payload["corrector_state_dict"], strict=True)
    source = {
        "mode": "checkpoint_same_backbone",
        "checkpoint": str(path),
        "sha256": strong.sha256(path),
        "bound_backbone_sha256": payload.get("backbone_sha256"),
        "source_backbone_sha256": source_backbone_sha,
        "updates": int(payload["updates"]),
    }
    return model, source


def _next_stage_a(loader, iterator, sampler, epoch: int, offset: int):
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


def _next_stage_b(dataset, loader, iterator, sampler, epoch: int, offset: int):
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


def _save_pair(out_dir: Path, *, update: int, backbone, corrector, cfg,
               source_backbone_sha: str, corrector_source: dict) -> tuple[Path, Path]:
    ck = out_dir / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    bp = ck / f"backbone_joint_{update:06d}.pth"
    _atomic_save({
        "model_state_dict": {k: v.detach().cpu() for k, v in backbone.state_dict().items()},
        "iteration": SOURCE_STAGE_A_UPDATE,
        "joint_update": int(update),
        "stage": "JOINT",
        "scope": "full",
        "feature_set": "P0-A",
        "feature_config": vars(cfg),
        "source_stage_a_update": SOURCE_STAGE_A_UPDATE,
        "source_backbone_sha256": source_backbone_sha,
        "source_corrector_update": SOURCE_CORRECTOR_UPDATE,
        "protocol": PROTOCOL,
        "scheduler_t_max": SCHEDULER_T_MAX,
    }, bp)
    cp = ck / f"corrector_joint_{update:06d}.pth"
    _atomic_save({
        "corrector_state_dict": {k: v.detach().cpu() for k, v in corrector.state_dict().items()},
        "architecture": {"in_channels": 42, "hidden": 64, "blocks": 2, "max_delta": 0.04},
        "backbone_sha256": strong.sha256(bp),
        "source_backbone_sha256": source_backbone_sha,
        "joint_update": int(update),
        "source_corrector": corrector_source,
        "scope": "full",
        "protocol": PROTOCOL,
    }, cp)
    return bp, cp


def _save_recovery(path: Path, *, update: int, backbone, corrector, optimizer, scheduler, cfg,
                   source_backbone_sha: str, corrector_source: dict,
                   a_epoch: int, a_offset: int, b_epoch: int, b_offset: int) -> None:
    payload: dict[str, object] = {
        "protocol": PROTOCOL,
        "joint_update": int(update),
        "source_stage_a_update": SOURCE_STAGE_A_UPDATE,
        "source_backbone_sha256": source_backbone_sha,
        "source_corrector_update": SOURCE_CORRECTOR_UPDATE,
        "corrector_source": corrector_source,
        "backbone_state_dict": backbone.state_dict(),
        "corrector_state_dict": corrector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "feature_config": vars(cfg),
        "stage_a_sampler_epoch": int(a_epoch),
        "stage_a_sampler_offset": int(a_offset),
        "stage_b_sampler_epoch": int(b_epoch),
        "stage_b_sampler_offset": int(b_offset),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    _atomic_save(payload, path)


def _restore_recovery(path: Path, backbone, corrector, optimizer, scheduler,
                      *, source_backbone_sha: str, source_corrector_sha: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("recovery protocol mismatch")
    if payload.get("source_backbone_sha256") != source_backbone_sha:
        raise ValueError("recovery source backbone SHA mismatch")
    if payload.get("corrector_source", {}).get("sha256") != source_corrector_sha:
        raise ValueError("recovery source corrector SHA mismatch")
    backbone.load_state_dict(payload["backbone_state_dict"], strict=True)
    corrector.load_state_dict(payload["corrector_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    torch.set_rng_state(payload["torch_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    return payload


def _setup(args: argparse.Namespace) -> dict[str, object]:
    if args.updates != UPDATES or tuple(args.milestones) != MILESTONES:
        raise ValueError(f"Full Joint V1 fixes updates/milestones at {UPDATES}/{MILESTONES}")
    if args.batch_size != BATCH_SIZE:
        raise ValueError(f"Full Joint V1 fixes batch size at {BATCH_SIZE}")
    if args.scheduler_t_max != SCHEDULER_T_MAX:
        raise ValueError(f"Full Joint V1 fixes scheduler T_max at {SCHEDULER_T_MAX}")
    for path in (args.data_root, args.kit_root, args.backbone_checkpoint, args.corrector_checkpoint):
        merge.assert_safe_path(path)
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.resume:
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_paths, dev_paths = merge.resolve_scope(argparse.Namespace(
        scope="full", manifest=None, data_root=args.data_root
    ))
    if dev_paths:
        raise RuntimeError("Full82 Joint must not construct a Dev split")
    if len(train_paths) != merge.FULL_TRAJECTORIES:
        raise ValueError("Full82 trajectory audit mismatch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")
    merge.core.set_seed(SEED)

    _, cfg, builder, backbone = _load_full_backbone(args.backbone_checkpoint, args.kit_root, device)
    source_backbone_sha = strong.sha256(args.backbone_checkpoint)
    corrector, corrector_source = _load_full_corrector(args.corrector_checkpoint, source_backbone_sha, device)

    optimizer = torch.optim.AdamW([
        {"params": backbone.parameters(), "lr": BACKBONE_LR, "weight_decay": BACKBONE_WEIGHT_DECAY},
        {"params": corrector.parameters(), "lr": CORRECTOR_LR, "weight_decay": CORRECTOR_WEIGHT_DECAY},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SCHEDULER_T_MAX)

    stage_a_ds = SpatialPhaseExpandedDataset(
        train_paths, in_steps=20, out_steps=20, stride=1, sub_sample=2,
        include_pressure=False, preload_to_ram=args.preload_to_ram,
    )
    if stage_a_ds.base_window_count != merge.FULL_DENSE_WINDOWS:
        raise ValueError("Full Stage-A dense window audit mismatch")
    stage_a_sampler = SpatialPhaseExpandedSampler(stage_a_ds, seed=SEED)
    stage_a_loader = DataLoader(
        stage_a_ds, batch_size=BATCH_SIZE, sampler=stage_a_sampler,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=bool(args.workers > 0),
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
        drop_last=True,
    )

    stage_b_ds = H5WindowDataset(
        train_paths, in_steps=20, out_steps=20, stride=20, sub_sample=2,
        include_pressure=False, window_mode="dense_all",
        preload_to_ram=args.preload_to_ram,
    )
    if len(stage_b_ds) != merge.FULL_DENSE_WINDOWS:
        raise ValueError("Full Stage-B dense window audit mismatch")
    stage_b_sampler = merge.P00DenseSampler(stage_b_ds, seed=SEED)
    stage_b_loader = DataLoader(
        stage_b_ds, batch_size=BATCH_SIZE, sampler=stage_b_sampler,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=bool(args.workers > 0),
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
        drop_last=True,
    )

    return locals()


def _process_tree_memory_bytes() -> tuple[int, int]:
    """Return summed RSS and summed PSS for this process tree.

    RSS double-counts fork-shared preload pages across DataLoader workers.
    Summed PSS apportions those shared pages and is the gate used for the
    physical-memory budget; RSS is retained only as a diagnostic.
    """
    rss_total = 0
    pss_total = 0
    seen: set[int] = set()
    stack = [os.getpid()]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    rss_total += int(line.split()[1]) * 1024
                    break
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
        try:
            for line in Path(f"/proc/{pid}/smaps_rollup").read_text(encoding="utf-8").splitlines():
                if line.startswith("Pss:"):
                    pss_total += int(line.split()[1]) * 1024
                    break
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
        try:
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text(encoding="utf-8").strip()
            if children:
                stack.extend(int(x) for x in children.split())
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return rss_total, pss_total


def _preflight(ctx: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    backbone, corrector, builder, device = ctx["backbone"], ctx["corrector"], ctx["builder"], ctx["device"]
    optimizer = ctx["optimizer"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    backbone.train(); corrector.train()
    xa, ya, _, _ = next(iter(ctx["stage_a_loader"]))
    xb, yb, _, _ = next(iter(ctx["stage_b_loader"]))
    xa, ya = xa.to(device), ya.to(device)
    xb, yb = xb.to(device), yb.to(device)
    pred_a = strong.forward_mf(backbone, builder, xa)
    loss_a, _ = merge.strong_loss(pred_a, ya, extra_rel=0.0)
    base_pred, final_pred = rt.forward_final(backbone, builder, corrector, xb)
    loss_b, _ = merge.strong_loss(base_pred, yb, extra_rel=strong.EXTRA_REL)
    loss_r, _ = merge.strong_loss(final_pred, yb, extra_rel=strong.EXTRA_REL)
    total = loss_a + loss_b + loss_r
    total.backward()
    bgrad = torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
    cgrad = torch.nn.utils.clip_grad_norm_(corrector.parameters(), 1.0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        gpu_peak_bytes = int(torch.cuda.max_memory_reserved(device))
    else:
        gpu_peak_bytes = 0
    host_rss_bytes, host_pss_bytes = _process_tree_memory_bytes()
    total_finite = bool(torch.isfinite(total).item())
    bgrad_f = float(bgrad)
    cgrad_f = float(cgrad)
    backbone_grad_ok = np.isfinite(bgrad_f) and bgrad_f > 0.0
    corrector_grad_ok = np.isfinite(cgrad_f) and cgrad_f > 0.0
    host_memory_ok = host_pss_bytes < HOST_RSS_LIMIT_BYTES
    gpu_peak_ok = device.type != "cuda" or gpu_peak_bytes < GPU_PEAK_LIMIT_BYTES
    passed = total_finite and backbone_grad_ok and corrector_grad_ok and host_memory_ok and gpu_peak_ok
    optimizer.zero_grad(set_to_none=True)
    report = {
        "status": "PASS" if passed else "FAIL",
        "optimizer_steps": 0,
        "loss_total": float(total.detach().cpu()),
        "loss_finite": total_finite,
        "backbone_grad_norm": bgrad_f,
        "backbone_grad_finite_nonzero": bool(backbone_grad_ok),
        "corrector_grad_norm": cgrad_f,
        "corrector_grad_finite_nonzero": bool(corrector_grad_ok),
        "host_rss_bytes": int(host_rss_bytes),
        "host_rss_gib": float(host_rss_bytes / 1024**3),
        "host_pss_bytes": int(host_pss_bytes),
        "host_pss_gib": float(host_pss_bytes / 1024**3),
        "host_memory_gate": "process_tree_pss",
        "host_memory_limit_gib": float(HOST_RSS_LIMIT_BYTES / 1024**3),
        "host_memory_ok": bool(host_memory_ok),
        "gpu_peak_reserved_bytes": int(gpu_peak_bytes),
        "gpu_peak_reserved_gib": float(gpu_peak_bytes / 1024**3),
        "gpu_peak_limit_gib": float(GPU_PEAK_LIMIT_BYTES / 1024**3),
        "gpu_peak_ok": bool(gpu_peak_ok),
        "full_trajectories": len(ctx["train_paths"]),
        "dense_windows": len(ctx["stage_b_ds"]),
        "stage_a_base_windows": ctx["stage_a_ds"].base_window_count,
    }
    _dump(args.out_dir / "preflight.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def run(args: argparse.Namespace) -> dict[str, object]:
    ctx = _setup(args)
    if args.preflight_only:
        return _preflight(ctx, args)

    backbone, corrector, builder, device = ctx["backbone"], ctx["corrector"], ctx["builder"], ctx["device"]
    optimizer, scheduler, cfg = ctx["optimizer"], ctx["scheduler"], ctx["cfg"]
    stage_a_ds, stage_a_sampler, stage_a_loader = ctx["stage_a_ds"], ctx["stage_a_sampler"], ctx["stage_a_loader"]
    stage_b_ds, stage_b_sampler, stage_b_loader = ctx["stage_b_ds"], ctx["stage_b_sampler"], ctx["stage_b_loader"]
    source_backbone_sha, corrector_source = ctx["source_backbone_sha"], ctx["corrector_source"]

    a_epoch = a_offset = b_epoch = b_offset = 0
    start_update = 1
    recovery = args.out_dir / "checkpoints" / "runner_latest.pth"
    if args.resume:
        if not recovery.is_file():
            raise FileNotFoundError(recovery)
        p = _restore_recovery(
            recovery, backbone, corrector, optimizer, scheduler,
            source_backbone_sha=source_backbone_sha,
            source_corrector_sha=corrector_source["sha256"],
        )
        start_update = int(p["joint_update"]) + 1
        a_epoch, a_offset = int(p["stage_a_sampler_epoch"]), int(p["stage_a_sampler_offset"])
        b_epoch, b_offset = int(p["stage_b_sampler_epoch"]), int(p["stage_b_sampler_offset"])

    a_epoch, a_offset = merge.normalize_stage_a_sampler(stage_a_sampler, a_epoch, a_offset)
    b_epoch, b_offset = merge.normalize_p00_state(stage_b_ds, stage_b_sampler, b_epoch, b_offset)
    a_iter, b_iter = iter(stage_a_loader), iter(stage_b_loader)

    _dump(args.out_dir / "run_config.json", {
        "status": STATUS,
        "protocol": PROTOCOL,
        "scope": "full",
        "train_trajectories": len(ctx["train_paths"]),
        "dense_windows": len(stage_b_ds),
        "source_stage_a_checkpoint": str(args.backbone_checkpoint),
        "source_stage_a_sha256": source_backbone_sha,
        "source_stage_a_update": SOURCE_STAGE_A_UPDATE,
        "corrector_source": corrector_source,
        "stage_a_sampling": "exhaustive P00/P01/P10/P11",
        "stage_b_sampling": "P00 Dense-All",
        "losses": "Clean Joint V1: Stage-A + Stage-B + residual final, equal weights",
        "backbone_lr": BACKBONE_LR,
        "corrector_lr": CORRECTOR_LR,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": SCHEDULER_T_MAX,
        "updates": UPDATES,
        "milestones": list(MILESTONES),
        "clean_reference_updates": CLEAN_REFERENCE_UPDATES,
        "clean_reference_selected_update": CLEAN_REFERENCE_SELECTED_UPDATE,
        "dev_selection": "NOT_PERFORMED",
        "aoa10_accessed": False,
        "locked_final_accessed": False,
        "codabench_accessed": False,
        "auto_sps": False,
        "auto_package": False,
        "preload_to_ram": bool(args.preload_to_ram),
        "stage_a_cache": stage_a_ds.cache_summary(),
        "stage_b_cache": stage_b_ds.cache_summary(),
        "workers": int(args.workers),
        "prefetch_factor": int(args.prefetch_factor),
    })

    started = time.monotonic()
    for update in range(start_update, UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        backbone.train(); corrector.train()

        (xa, ya, _, _), a_iter, a_epoch, a_offset = _next_stage_a(
            stage_a_loader, a_iter, stage_a_sampler, a_epoch, a_offset
        )
        xa, ya = xa.to(device, non_blocking=True), ya.to(device, non_blocking=True)
        pred_a = strong.forward_mf(backbone, builder, xa)
        loss_a, parts_a = merge.strong_loss(pred_a, ya, extra_rel=0.0)
        if not torch.isfinite(loss_a):
            raise FloatingPointError(f"non-finite Stage-A loss @{update}")
        loss_a.backward()
        del pred_a, xa, ya

        (xb, yb, _, _), b_iter, b_epoch, b_offset = _next_stage_b(
            stage_b_ds, stage_b_loader, b_iter, stage_b_sampler, b_epoch, b_offset
        )
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        base_pred, final_pred = rt.forward_final(backbone, builder, corrector, xb)
        loss_b, parts_b = merge.strong_loss(base_pred, yb, extra_rel=strong.EXTRA_REL)
        loss_r, parts_r = merge.strong_loss(final_pred, yb, extra_rel=strong.EXTRA_REL)
        if not torch.isfinite(loss_b) or not torch.isfinite(loss_r):
            raise FloatingPointError(f"non-finite Stage-B/Residual loss @{update}")
        (loss_b + loss_r).backward()

        bgrad = torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
        cgrad = torch.nn.utils.clip_grad_norm_(corrector.parameters(), 1.0)
        if update == start_update:
            if not np.isfinite(float(bgrad)) or not np.isfinite(float(cgrad)) or float(cgrad) <= 0:
                raise RuntimeError("invalid first Full Joint gradients")
        optimizer.step(); scheduler.step()

        if update == start_update or update % args.log_every == 0:
            print(json.dumps({
                "joint_update": update,
                "loss_a": float(loss_a.detach().cpu()),
                "loss_b": float(loss_b.detach().cpu()),
                "loss_r": float(loss_r.detach().cpu()),
                "backbone_grad_norm_preclip": float(bgrad),
                "corrector_grad_norm_preclip": float(cgrad),
                "backbone_lr": float(optimizer.param_groups[0]["lr"]),
                "corrector_lr": float(optimizer.param_groups[1]["lr"]),
                "stage_a_rel_component": float(parts_a["rel"].detach().cpu()),
                "stage_b_rel_component": float(parts_b["rel"].detach().cpu()),
                "residual_rel_component": float(parts_r["rel"].detach().cpu()),
                "stage_a_sampler_epoch": a_epoch,
                "stage_a_sampler_offset": a_offset,
                "stage_b_sampler_epoch": b_epoch,
                "stage_b_sampler_offset": b_offset,
            }, sort_keys=True), flush=True)

        if update in MILESTONES:
            bp, cp = _save_pair(
                args.out_dir, update=update, backbone=backbone, corrector=corrector, cfg=cfg,
                source_backbone_sha=source_backbone_sha, corrector_source=corrector_source,
            )
            print(json.dumps({"joint_update": update, "backbone_checkpoint": str(bp), "corrector_checkpoint": str(cp)}), flush=True)

        if update % RECOVERY_EVERY == 0 or update == UPDATES:
            _save_recovery(
                recovery, update=update, backbone=backbone, corrector=corrector,
                optimizer=optimizer, scheduler=scheduler, cfg=cfg,
                source_backbone_sha=source_backbone_sha, corrector_source=corrector_source,
                a_epoch=a_epoch, a_offset=a_offset, b_epoch=b_epoch, b_offset=b_offset,
            )

    final_bp = args.out_dir / "checkpoints" / f"backbone_joint_{UPDATES:06d}.pth"
    final_cp = args.out_dir / "checkpoints" / f"corrector_joint_{UPDATES:06d}.pth"
    result = {
        "status": STATUS,
        "protocol": PROTOCOL,
        "scope": "full",
        "completed_updates": UPDATES,
        "scheduler_t_max": SCHEDULER_T_MAX,
        "final_backbone_checkpoint": str(final_bp),
        "final_backbone_sha256": strong.sha256(final_bp),
        "final_corrector_checkpoint": str(final_cp),
        "final_corrector_sha256": strong.sha256(final_cp),
        "runtime_seconds_this_process": time.monotonic() - started,
        "dev_selection": "NOT_PERFORMED",
        "aoa10_accessed": False,
        "locked_final_accessed": False,
        "codabench_accessed": False,
        "next_action": "REVIEW_REQUIRED: restore the frozen Clean OOS SPS head and build/verify package; do not refit SPS on Full82.",
    }
    _dump(args.out_dir / "summary.json", result)
    _dump(args.out_dir / "status.json", {"state": "DONE", "status": STATUS, "joint_update": UPDATES})
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--kit-root", type=Path, required=True)
    p.add_argument("--backbone-checkpoint", type=Path, required=True)
    p.add_argument("--corrector-checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--updates", type=int, default=UPDATES)
    p.add_argument("--milestones", type=int, nargs="+", default=list(MILESTONES))
    p.add_argument("--scheduler-t-max", type=int, default=SCHEDULER_T_MAX)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--preload-to-ram", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--require-cuda", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()