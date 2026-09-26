#!/usr/bin/env python3
"""Frozen-backbone residual stage for the serial SOTA-merge comparison.

This is the original Strong-Backbone residual recipe, not the Joint-V1
corrector recipe:
- frozen P0-A/MF backbone selected by Seen-Dev Stage-B only
- ResidualCorrector3D hidden=96, blocks=2, max_delta=0.04
- AdamW lr=2e-4, wd=1e-5, cosine over exactly 22k updates
- scalar physics objective: point=1, mse=.05, tke=.06, temporal=.03,
  grad=.015, p_zero=.01, residual_mse=.25, delta_penalty=.02
- alpha fixed at 1.0; checkpoint selection uses point score only
- exact recovery state every 500 updates

AoA10 is intentionally absent from this trainer and cannot influence model
selection.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import realpde_sota_v2_integrated as strong
import train_sota_merge_backbone as merge
from colleague_80pt.realpde_h5_feature_adapter_train import (
    H5WindowDataset,
    RandomPhaseWindowSampler,
    paths_from_split_manifest,
    physics_loss,
)
from colleague_80pt import residual_multi as rm

STATUS = "REVIEW_REQUIRED"
PROTOCOL = "REALPDE_SOTA_MERGE_SERIAL_RESIDUAL_V1"
SEED = 41
UPDATES = 22_000
BATCH_SIZE = 8
EVAL_INTERVAL = 1_000
RECOVERY_EVERY = 500
LR = 2e-4
WEIGHT_DECAY = 1e-5
HIDDEN = 96
BLOCKS = 2
MAX_DELTA = 0.04
DROP_OUT = 0.0

LOSS_WEIGHTS = {
    "point": 1.0,
    "mse": 0.05,
    "tke": 0.06,
    "temporal": 0.03,
    "grad": 0.015,
    "p_zero": 0.01,
}
RESIDUAL_MSE = 0.25
DELTA_PENALTY = 0.02
ABS_WIDTHS = (0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.04)
REL_WIDTHS = (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.08, 0.1)


def atomic_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    tmp.replace(path)


def save_checkpoint(
    path: Path,
    *,
    model: rm.ResidualCorrectionModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    update: int,
    sampler_epoch: int,
    batches_in_epoch: int,
    best_update: int,
    best_score: float,
    best_alpha: float,
    best_bound_abs: float,
    best_bound_rel: float,
    backbone_checkpoint: Path,
    run_config: dict[str, object],
    history: list[dict[str, object]],
) -> None:
    atomic_save(
        {
            "protocol": PROTOCOL,
            "status": STATUS,
            "model_state_dict": model.state_dict(),
            "corrector_config": asdict(model.corrector.config),
            "run_config": run_config,
            "iteration": int(update),
            "best_iteration": int(best_update),
            "best_score": float(best_score),
            "best_alpha": float(best_alpha),
            "best_bound_abs": float(best_bound_abs),
            "best_bound_rel": float(best_bound_rel),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "sampler_epoch": int(sampler_epoch),
            "batches_in_epoch": int(batches_in_epoch),
            "source_backbone_sha256": strong.sha256(backbone_checkpoint),
            "history": history,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def restore_checkpoint(
    path: Path,
    *,
    model: rm.ResidualCorrectionModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    backbone_checkpoint: Path,
) -> dict[str, object]:
    p = torch.load(path, map_location="cpu", weights_only=False)
    if p.get("protocol") != PROTOCOL:
        raise ValueError("recovery protocol mismatch")
    if p.get("source_backbone_sha256") != strong.sha256(backbone_checkpoint):
        raise ValueError("recovery checkpoint belongs to another backbone")
    model.load_state_dict(p["model_state_dict"], strict=True)
    optimizer.load_state_dict(p["optimizer_state_dict"])
    scheduler.load_state_dict(p["scheduler_state_dict"])
    if "torch_rng_state" in p:
        torch.set_rng_state(p["torch_rng_state"])
    if "numpy_rng_state" in p:
        np.random.set_state(p["numpy_rng_state"])
    if torch.cuda.is_available() and p.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(p["cuda_rng_state_all"])
    return p


def make_loaders(args: argparse.Namespace, device: torch.device):
    train_paths, dev_paths = paths_from_split_manifest(args.data_root, args.manifest)
    if set(p.name for p in train_paths) & set(p.name for p in dev_paths):
        raise ValueError("train/dev overlap")
    if (len(train_paths), len(dev_paths)) != (51, 12):
        raise ValueError(f"serial clean protocol requires Train51/Seen-Dev12, got {len(train_paths)}/{len(dev_paths)}")

    base_train = H5WindowDataset(
        train_paths,
        in_steps=20,
        out_steps=20,
        stride=1,
        sub_sample=2,
        include_pressure=False,
        window_mode="random_phase",
        preload_to_ram=args.preload_to_ram,
    )
    fixed_phases = {path.name: 0 for path in train_paths}
    sampler = RandomPhaseWindowSampler(base_train, seed=SEED, equalize_phase_counts=True)
    sampler.set_epoch(0, phases=fixed_phases)

    worker_kwargs: dict[str, object] = {}
    if args.workers > 0:
        worker_kwargs = {"persistent_workers": True, "prefetch_factor": args.prefetch_factor}
    train_loader = DataLoader(
        base_train,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        **worker_kwargs,
    )

    val_dataset = H5WindowDataset(
        dev_paths,
        in_steps=20,
        out_steps=20,
        stride=20,
        sub_sample=2,
        include_pressure=False,
        preload_to_ram=args.preload_to_ram,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        **worker_kwargs,
    )
    return train_paths, dev_paths, base_train, fixed_phases, sampler, train_loader, val_dataset, val_loader


@torch.no_grad()
def evaluate_point(model, loader, device) -> dict[str, object]:
    summaries = rm.evaluate_alphas(
        model,
        loader,
        device,
        alphas=(1.0,),
        abs_widths=ABS_WIDTHS,
        rel_widths=REL_WIDTHS,
    )
    top = summaries[0]
    return {
        "point_score": float(rm.checkpoint_selection_score(top, "point_score")),
        "rel_l2": float(top["rel_l2"]),
        "tke": float(top["tke"]),
        "mvpe": float(top["mvpe"]),
        "rel_l2_score": float(top["rel_l2_score"]),
        "tke_score": float(top["tke_score"]),
        "mvpe_score": float(top["mvpe_score"]),
        "alpha": 1.0,
        "best_bound_abs_diagnostic_only": float(top["best_bound_abs"]),
        "best_bound_rel_diagnostic_only": float(top["best_bound_rel"]),
    }


def resume_iterator(loader, sampler, fixed_phases, epoch: int, batches_in_epoch: int):
    batches_per_epoch = len(loader)
    if batches_per_epoch < 1:
        raise RuntimeError("empty training loader")
    while batches_in_epoch >= batches_per_epoch:
        batches_in_epoch -= batches_per_epoch
        epoch += 1
    sampler.set_epoch(epoch, phases=fixed_phases)
    iterator = iter(loader)
    for _ in range(batches_in_epoch):
        try:
            next(iterator)
        except StopIteration as exc:
            raise RuntimeError("saved batches_in_epoch exceeds deterministic epoch") from exc
    return iterator, epoch, batches_in_epoch


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.updates != UPDATES:
        raise ValueError(f"V1 fixes residual updates at {UPDATES}")
    if args.batch_size != BATCH_SIZE:
        raise ValueError(f"V1 fixes batch size at {BATCH_SIZE}")
    if args.eval_interval != EVAL_INTERVAL:
        raise ValueError(f"V1 fixes eval interval at {EVAL_INTERVAL}")
    if args.recovery_every != RECOVERY_EVERY:
        raise ValueError(f"V1 fixes recovery interval at {RECOVERY_EVERY}")

    for path in (args.data_root, args.manifest, args.kit_root, args.backbone_checkpoint):
        merge.assert_safe_path(path)
    if not args.backbone_checkpoint.is_file():
        raise FileNotFoundError(args.backbone_checkpoint)
    if not (args.kit_root / "scoring.py").is_file():
        raise FileNotFoundError(args.kit_root / "scoring.py")

    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.resume:
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "checkpoints").mkdir(exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")

    (
        train_paths, dev_paths, base_train, fixed_phases, sampler,
        train_loader, val_dataset, val_loader,
    ) = make_loaders(args, device)
    if len(base_train) != merge.CLEAN_DENSE_WINDOWS:
        raise ValueError(f"Dense-All candidate audit mismatch: {len(base_train)} != {merge.CLEAN_DENSE_WINDOWS}")
    if len(val_dataset) != merge.CLEAN_DEV_WINDOWS:
        raise ValueError(f"Seen-Dev window audit mismatch: {len(val_dataset)} != {merge.CLEAN_DEV_WINDOWS}")

    base_model = rm.load_frozen_base("sota_v2_mf", args.backbone_checkpoint, args.kit_root, device)
    config = rm.CorrectorConfig(
        hidden=HIDDEN,
        blocks=BLOCKS,
        dropout=DROP_OUT,
        include_pressure=True,
        max_delta=MAX_DELTA,
        history_context=False,
    )
    model = rm.ResidualCorrectionModel(base_model, rm.ResidualCorrector3D(config)).to(device)
    optimizer = torch.optim.AdamW(model.corrector.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UPDATES)

    run_config = {
        "status": STATUS,
        "protocol": PROTOCOL,
        "source_backbone_checkpoint": str(args.backbone_checkpoint),
        "source_backbone_sha256": strong.sha256(args.backbone_checkpoint),
        "selection_split": "Seen-Dev12 only",
        "aoa_used_for_selection": False,
        "train_trajectories": len(train_paths),
        "seen_dev_trajectories": len(dev_paths),
        "candidate_legal_windows": len(base_train),
        "seen_dev_windows": len(val_dataset),
        "sampling": "stride1 + fixed P00 phase + global shuffle",
        "updates": UPDATES,
        "batch_size": BATCH_SIZE,
        "eval_interval": EVAL_INTERVAL,
        "recovery_every": RECOVERY_EVERY,
        "optimizer": {"name": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY, "scheduler": "CosineAnnealingLR"},
        "corrector": asdict(config),
        "loss_weights": LOSS_WEIGHTS,
        "residual_mse": RESIDUAL_MSE,
        "delta_penalty": DELTA_PENALTY,
        "gradient_mode": "scalar",
        "train_alpha": 1.0,
        "eval_alpha": 1.0,
        "selection_metric": "point_score",
        "sps_used_for_selection": False,
        "runtime_used_for_selection": False,
        "locked_final_accessed": False,
        "private_accessed": False,
        "codabench_accessed": False,
    }
    merge.dump(args.out_dir / "run_config.json", run_config)

    recovery = args.out_dir / "checkpoints" / "runner_latest.pth"
    history: list[dict[str, object]] = []
    start_update = 1
    sampler_epoch = 0
    batches_in_epoch = 0

    if args.resume:
        if not recovery.is_file():
            raise FileNotFoundError(recovery)
        state = restore_checkpoint(
            recovery,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            backbone_checkpoint=args.backbone_checkpoint,
        )
        start_update = int(state["iteration"]) + 1
        sampler_epoch = int(state.get("sampler_epoch", 0))
        batches_in_epoch = int(state.get("batches_in_epoch", 0))
        best_update = int(state["best_iteration"])
        best_score = float(state["best_score"])
        best_alpha = float(state.get("best_alpha", 1.0))
        best_bound_abs = float(state.get("best_bound_abs", 0.005))
        best_bound_rel = float(state.get("best_bound_rel", 0.0))
        history = list(state.get("history", []))
    else:
        initial = evaluate_point(model, val_loader, device)
        initial["update"] = 0
        history.append(initial)
        best_update = 0
        best_score = float(initial["point_score"])
        best_alpha = 1.0
        best_bound_abs = float(initial["best_bound_abs_diagnostic_only"])
        best_bound_rel = float(initial["best_bound_rel_diagnostic_only"])
        save_checkpoint(
            args.out_dir / "checkpoints" / "model_init.pth",
            model=model, optimizer=optimizer, scheduler=scheduler,
            update=0, sampler_epoch=0, batches_in_epoch=0,
            best_update=best_update, best_score=best_score, best_alpha=best_alpha,
            best_bound_abs=best_bound_abs, best_bound_rel=best_bound_rel,
            backbone_checkpoint=args.backbone_checkpoint, run_config=run_config, history=history,
        )
        save_checkpoint(
            args.out_dir / "checkpoints" / "model_best.pth",
            model=model, optimizer=optimizer, scheduler=scheduler,
            update=0, sampler_epoch=0, batches_in_epoch=0,
            best_update=best_update, best_score=best_score, best_alpha=best_alpha,
            best_bound_abs=best_bound_abs, best_bound_rel=best_bound_rel,
            backbone_checkpoint=args.backbone_checkpoint, run_config=run_config, history=history,
        )

    iterator, sampler_epoch, batches_in_epoch = resume_iterator(
        train_loader, sampler, fixed_phases, sampler_epoch, batches_in_epoch
    )

    started = time.monotonic()
    for step in range(start_update, UPDATES + 1):
        try:
            x, y = next(iterator)
            batches_in_epoch += 1
        except StopIteration:
            sampler_epoch += 1
            batches_in_epoch = 0
            sampler.set_epoch(sampler_epoch, phases=fixed_phases)
            iterator = iter(train_loader)
            x, y = next(iterator)
            batches_in_epoch = 1

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        model.base_model.eval()
        model.corrector.train()

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            base = model.base_predict(x)
        delta = model.predict_delta(x, base)
        pred = model.combine(base, delta, 1.0)
        loss, parts = physics_loss(pred, y, LOSS_WEIGHTS)
        residual_target = y[..., :2] - base[..., :2]
        residual_mse = torch.mean((delta[..., :2] - residual_target) ** 2)
        delta_penalty = torch.mean(delta[..., :2] ** 2)
        total = loss + RESIDUAL_MSE * residual_mse + DELTA_PENALTY * delta_penalty
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite residual loss @{step}")
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.corrector.parameters(), 1.0)
        if not np.isfinite(float(grad_norm)):
            raise FloatingPointError(f"non-finite corrector gradient @{step}")
        optimizer.step()
        scheduler.step()

        if step == 1 or step % args.log_every == 0:
            print(json.dumps({
                "update": step,
                "loss": float(total.detach().cpu()),
                "residual_mse": float(residual_mse.detach().cpu()),
                "delta_penalty": float(delta_penalty.detach().cpu()),
                "grad_norm_preclip": float(grad_norm),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "sampler_epoch": sampler_epoch,
                "batches_in_epoch": batches_in_epoch,
                **{f"loss_{k}": float(v) for k, v in parts.items()},
            }, sort_keys=True), flush=True)

        if step % EVAL_INTERVAL == 0:
            metrics = evaluate_point(model, val_loader, device)
            metrics["update"] = step
            metrics["elapsed_seconds"] = time.monotonic() - started
            history.append(metrics)
            merge.dump(args.out_dir / f"eval_seen_dev_{step:06d}.json", metrics)

            current = float(metrics["point_score"])
            if current > best_score:
                best_score = current
                best_update = step
                best_bound_abs = float(metrics["best_bound_abs_diagnostic_only"])
                best_bound_rel = float(metrics["best_bound_rel_diagnostic_only"])
                save_checkpoint(
                    args.out_dir / "checkpoints" / "model_best.pth",
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    update=step, sampler_epoch=sampler_epoch, batches_in_epoch=batches_in_epoch,
                    best_update=best_update, best_score=best_score, best_alpha=1.0,
                    best_bound_abs=best_bound_abs, best_bound_rel=best_bound_rel,
                    backbone_checkpoint=args.backbone_checkpoint, run_config=run_config, history=history,
                )

            save_checkpoint(
                args.out_dir / "checkpoints" / f"model_update_{step:06d}.pth",
                model=model, optimizer=optimizer, scheduler=scheduler,
                update=step, sampler_epoch=sampler_epoch, batches_in_epoch=batches_in_epoch,
                best_update=best_update, best_score=best_score, best_alpha=1.0,
                best_bound_abs=best_bound_abs, best_bound_rel=best_bound_rel,
                backbone_checkpoint=args.backbone_checkpoint, run_config=run_config, history=history,
            )

        if step % RECOVERY_EVERY == 0 or step == UPDATES:
            save_checkpoint(
                recovery,
                model=model, optimizer=optimizer, scheduler=scheduler,
                update=step, sampler_epoch=sampler_epoch, batches_in_epoch=batches_in_epoch,
                best_update=best_update, best_score=best_score, best_alpha=1.0,
                best_bound_abs=best_bound_abs, best_bound_rel=best_bound_rel,
                backbone_checkpoint=args.backbone_checkpoint, run_config=run_config, history=history,
            )
            merge.dump(args.out_dir / "progress.json", {
                "status": STATUS,
                "state": "RUNNING" if step < UPDATES else "TRAINING_COMPLETE",
                "current_update": step,
                "target_update": UPDATES,
                "best_update": best_update,
                "best_point_score": best_score,
                "sampler_epoch": sampler_epoch,
                "batches_in_epoch": batches_in_epoch,
                "elapsed_seconds_this_process": time.monotonic() - started,
            })

    final = {
        "status": STATUS,
        "protocol": PROTOCOL,
        "state": "DONE",
        "best_update": best_update,
        "best_point_score": best_score,
        "source_backbone_checkpoint": str(args.backbone_checkpoint),
        "source_backbone_sha256": strong.sha256(args.backbone_checkpoint),
        "history": history,
        "selection_split": "Seen-Dev12 only",
        "aoa_used_for_selection": False,
        "locked_final_accessed": False,
        "private_accessed": False,
        "codabench_accessed": False,
    }
    merge.dump(args.out_dir / "summary.json", final)
    (args.out_dir / "DONE").touch()
    return final


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--kit-root", type=Path, required=True)
    p.add_argument("--backbone-checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--updates", type=int, default=UPDATES)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    p.add_argument("--recovery-every", type=int, default=RECOVERY_EVERY)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--preload-to-ram", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--require-cuda", action="store_true")
    args = p.parse_args()
    result = run(args)
    print(json.dumps({
        "status": result["status"],
        "best_update": result["best_update"],
        "best_point_score": result["best_point_score"],
        "out_dir": str(args.out_dir),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
