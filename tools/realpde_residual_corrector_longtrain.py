#!/usr/bin/env python3
"""Full-training campaign for the SOTA-V2 residual corrector.

Scientific intent
-----------------
TMR-01 proved a strong, trajectory-stable Rel-L2 signal from the bounded 3D
residual corrector, but the original corrector saw only 2400 batches, less than
half of one Dense-All epoch. This runner keeps the backbone, corrector
architecture, features, loss, optimizer family, batch size, and alpha frozen,
and changes only training maturity: 30k corrector updates with one cosine
schedule over the full campaign.

No Dev data are evaluated until all 30k training updates finish. Checkpoints at
6k/12k/20k/30k are then replayed together against the frozen @32500 backbone.
The trajectory x horizon analysis reuses the TMR-01A diagnostic definitions.
There is no SPS, uncertainty, full-data, locked-final/private, package, or
Codabench path in this file.
"""
from __future__ import annotations

import argparse
import json
import time
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

import realpde_loss_official_v9 as core
import realpde_sota_v2_integrated as sota
import realpde_teammate_residual_transfer as transfer
import realpde_tmr01a_trajectory_horizon_audit as audit
from realpde_adaptive_probe import ResidualCorrector3D

BACKBONE_UPDATE = 32_500
BACKBONE_SHA256 = "6926722895611c38dee79cf16ce43b87c7de3d4a1305e9db431dadd4c1a7bf47"
FINAL_UPDATE = 30_000
MILESTONES = (6_000, 12_000, 20_000, 30_000)
BATCH_SIZE = 8
CORRECTOR_LR = 1e-4
CORRECTOR_WEIGHT_DECAY = 1e-5
CORRECTOR_SEED = 20260905
TRAIN_SAMPLER_SEED = 20260901
ALPHA = 1.0
DENSE_TRAIN_WINDOWS = 40_488
EXPECTED_DEV_TRAJECTORIES = 16
EXPECTED_DEV_WINDOWS = 659
REGISTERED_BASELINE = {"rel_l2": 0.09993461519479752, "tke": 0.4692927300930023, "mvpe": 0.07577798515558243}


def effective_dense_epochs(*, updates: int = FINAL_UPDATE, batch_size: int = BATCH_SIZE,
                           dense_windows: int = DENSE_TRAIN_WINDOWS) -> float:
    if min(updates, batch_size, dense_windows) < 1:
        raise ValueError("updates, batch_size, and dense_windows must be positive")
    return float(updates * batch_size / dense_windows)


def validate_recipe(args: argparse.Namespace) -> None:
    if int(args.updates) != FINAL_UPDATE:
        raise ValueError(f"long-train fixes updates at {FINAL_UPDATE}")
    if int(args.batch_size) != BATCH_SIZE:
        raise ValueError(f"long-train fixes batch size at {BATCH_SIZE}")
    if tuple(int(x) for x in args.milestones) != MILESTONES:
        raise ValueError(f"long-train fixes milestones at {MILESTONES}")


def build_optimizer_scheduler(module: nn.Module):
    optimizer = torch.optim.AdamW(
        module.parameters(), lr=CORRECTOR_LR, weight_decay=CORRECTOR_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINAL_UPDATE)
    return optimizer, scheduler


def _variant_sort_key(name: str) -> tuple[int, int | str]:
    if name == "base":
        return (0, 0)
    if name.startswith("u") and name[1:].isdigit():
        return (1, int(name[1:]))
    return (2, name)


def build_analysis_tables(*, predictions: dict[str, np.ndarray], target: np.ndarray,
                          trajectory_names: np.ndarray) -> dict[str, list[dict]]:
    """Generalize the TMR-01A diagnostics over training checkpoints.

    Returns long-form all-Dev by-horizon, trajectory x horizon, and paired
    trajectory-stability tables. All TKE/MVPE per-horizon fields remain
    diagnostics rather than official per-frame scores.
    """
    if "base" not in predictions:
        raise ValueError("predictions must include a 'base' variant")
    if len(trajectory_names) != len(target):
        raise ValueError("trajectory_names length must equal target windows")
    variants = sorted(predictions, key=_variant_sort_key)
    for name in variants:
        audit._validate_arrays(predictions[name], target)
        if len(predictions[name]) != len(target):
            raise ValueError(f"prediction windows mismatch for {name}")

    diagnostics = {
        name: audit.window_horizon_diagnostics(predictions[name], target)
        for name in variants
    }
    names = np.asarray(trajectory_names).astype(str)
    groups = {
        name: np.flatnonzero(names == name)
        for name in sorted(set(names.tolist()))
    }

    by_horizon: list[dict] = []
    for h in range(20):
        base_values = {
            metric: float(diagnostics["base"][metric][:, h].mean())
            for metric in diagnostics["base"]
        }
        for variant in variants:
            values = {
                metric: float(diagnostics[variant][metric][:, h].mean())
                for metric in diagnostics[variant]
            }
            row = {"variant": variant, "horizon": h + 1, **values}
            for metric in audit.ERROR_METRICS:
                row[f"{metric}_improvement_vs_base"] = audit._relative_improvement(
                    base_values[metric], values[metric]
                )
            row["tke_contrib_ratio_delta_vs_base"] = (
                values["tke_contrib_ratio"] - base_values["tke_contrib_ratio"]
            )
            by_horizon.append(row)

    by_trajectory_horizon: list[dict] = []
    for trajectory_id, indices in groups.items():
        for h in range(20):
            base_values = {
                metric: float(diagnostics["base"][metric][indices, h].mean())
                for metric in diagnostics["base"]
            }
            for variant in variants:
                values = {
                    metric: float(diagnostics[variant][metric][indices, h].mean())
                    for metric in diagnostics[variant]
                }
                row = {
                    "trajectory_id": trajectory_id,
                    "horizon": h + 1,
                    "variant": variant,
                    "windows": int(len(indices)),
                    **values,
                }
                for metric in audit.ERROR_METRICS:
                    row[f"{metric}_improvement_vs_base"] = audit._relative_improvement(
                        base_values[metric], values[metric]
                    )
                row["tke_contrib_ratio_delta_vs_base"] = (
                    values["tke_contrib_ratio"] - base_values["tke_contrib_ratio"]
                )
                by_trajectory_horizon.append(row)

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in by_trajectory_horizon:
        if row["variant"] != "base":
            grouped[(str(row["variant"]), int(row["horizon"]))].append(row)

    stability: list[dict] = []
    for (variant, horizon), rows in sorted(grouped.items(), key=lambda item: (_variant_sort_key(item[0][0]), item[0][1])):
        result: dict[str, float | int | str] = {
            "variant": variant,
            "horizon": horizon,
            "trajectory_count": len(rows),
        }
        for metric in audit.ERROR_METRICS:
            key = f"{metric}_improvement_vs_base"
            values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            result[f"{metric}_improved_count"] = int((finite > 0.0).sum())
            result[f"{metric}_improved_fraction"] = float((finite > 0.0).mean()) if len(finite) else float("nan")
            result[f"{metric}_mean_improvement"] = float(finite.mean()) if len(finite) else float("nan")
            result[f"{metric}_median_improvement"] = float(np.median(finite)) if len(finite) else float("nan")
            result[f"{metric}_worst_improvement"] = float(finite.min()) if len(finite) else float("nan")
            result[f"{metric}_best_improvement"] = float(finite.max()) if len(finite) else float("nan")
        stability.append(result)

    return {
        "by_horizon": by_horizon,
        "by_trajectory_horizon": by_trajectory_horizon,
        "horizon_trajectory_stability": stability,
    }


def _save_corrector_checkpoint(*, path: Path, corrector: ResidualCorrector3D,
                               optimizer, scheduler, step: int, epoch: int,
                               backbone_checkpoint: Path) -> dict:
    payload = {
        "corrector_state_dict": corrector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "backbone_update": BACKBONE_UPDATE,
        "backbone_sha256": sota.sha256(backbone_checkpoint),
        "updates": step,
        "epoch": epoch,
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
        "dense_train_windows": DENSE_TRAIN_WINDOWS,
        "effective_dense_epochs": effective_dense_epochs(updates=step),
    }
    torch.save(payload, path)
    return {
        "update": step,
        "path": str(path),
        "sha256": sota.sha256(path),
        "effective_dense_epochs": payload["effective_dense_epochs"],
        "lr": float(optimizer.param_groups[0]["lr"]),
        "epoch": epoch,
    }


def train_long_corrector(*, checkpoint: Path, train_paths: list[Path], kit_root: Path,
                         builder, device: torch.device, out_dir: Path,
                         workers: int) -> dict:
    torch.manual_seed(CORRECTOR_SEED)
    np.random.seed(CORRECTOR_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(CORRECTOR_SEED)

    backbone, _ = transfer.load_backbone(checkpoint, BACKBONE_UPDATE, kit_root, builder, device)
    corrector = ResidualCorrector3D(in_channels=42, hidden=64, blocks=2, max_delta=0.04).to(device)
    optimizer, scheduler = build_optimizer_scheduler(corrector)

    loader_args = Namespace(seed=TRAIN_SAMPLER_SEED, micro_batch=BATCH_SIZE, workers=workers)
    train_ds, sampler, loader = sota.dense_loader(train_paths, loader_args)
    if len(train_ds) != DENSE_TRAIN_WINDOWS:
        raise ValueError(f"Dense-All train windows {len(train_ds)} != {DENSE_TRAIN_WINDOWS}")

    iterator = iter(loader)
    epoch = 0
    checkpoints: list[dict] = []
    log_path = out_dir / "train_corrector_long.jsonl"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        for step in range(1, FINAL_UPDATE + 1):
            try:
                x, y, _, _ = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                x, y, _, _ = next(iterator)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                base = sota.forward_mf(backbone, builder, x)

            corrector.train()
            delta = transfer._delta_from_corrector(corrector, x, base)
            total, parts = transfer.corrector_objective(base, y, delta)
            if not torch.isfinite(total):
                raise FloatingPointError(f"nonfinite corrector loss at step={step}")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(corrector.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            row = {
                "step": step,
                "epoch": epoch,
                "loss": float(total.detach().cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                **{f"loss_{name}": float(value.detach().cpu()) for name, value in parts.items()},
            }
            log.write(json.dumps(row, sort_keys=True) + "\n")
            if step % 100 == 0 or step in MILESTONES:
                log.flush()
                print(json.dumps(row, sort_keys=True), flush=True)

            if step in MILESTONES:
                ckpt = out_dir / f"corrector_update_{step}.pth"
                checkpoints.append(_save_corrector_checkpoint(
                    path=ckpt,
                    corrector=corrector,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    epoch=epoch,
                    backbone_checkpoint=checkpoint,
                ))

    summary = {
        "updates": FINAL_UPDATE,
        "batch_size": BATCH_SIZE,
        "dense_train_windows": len(train_ds),
        "effective_dense_epochs": effective_dense_epochs(),
        "training_wall_seconds": time.monotonic() - started,
        "milestones": checkpoints,
    }
    transfer.dump(out_dir / "training_summary.json", summary)
    del backbone, corrector, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def _load_frozen_corrector(path: Path, expected_update: int, device: torch.device) -> ResidualCorrector3D:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("updates", -1)) != expected_update:
        raise ValueError(f"corrector checkpoint {path} updates != {expected_update}")
    if payload.get("backbone_sha256") != BACKBONE_SHA256:
        raise ValueError(f"corrector checkpoint {path} backbone hash mismatch")
    model = ResidualCorrector3D(in_channels=42, hidden=64, blocks=2, max_delta=0.04).to(device)
    model.load_state_dict(payload["corrector_state_dict"], strict=True)
    transfer.freeze_module(model)
    return model


@torch.no_grad()
def replay_milestones(*, checkpoint: Path, corrector_paths: dict[int, Path],
                      dev_paths: list[Path], train_paths: list[Path], kit_root: Path,
                      builder, device: torch.device, out_dir: Path,
                      eval_batch_size: int, workers: int) -> dict:
    backbone, _ = transfer.load_backbone(checkpoint, BACKBONE_UPDATE, kit_root, builder, device)
    correctors = {
        update: _load_frozen_corrector(path, update, device)
        for update, path in corrector_paths.items()
    }

    ds, loader = sota.dev_loader(dev_paths, Namespace(eval_batch_size=eval_batch_size, workers=workers))
    if len(ds) != EXPECTED_DEV_WINDOWS:
        raise ValueError(f"Dev windows {len(ds)} != {EXPECTED_DEV_WINDOWS}")

    variants = ["base", *(f"u{update}" for update in MILESTONES)]
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in variants}
    targets: list[np.ndarray] = []
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        base = sota.forward_mf(backbone, builder, x)
        chunks["base"].append(base.cpu().numpy().astype(np.float32))
        for update in MILESTONES:
            delta = transfer._delta_from_corrector(correctors[update], x, base)
            pred = transfer.apply_scaled_correction(base, delta, ALPHA)
            chunks[f"u{update}"].append(pred.cpu().numpy().astype(np.float32))
        targets.append(y.numpy().astype(np.float32))

    predictions = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    target = np.concatenate(targets, axis=0)
    trajectory_names = np.asarray([ref.path.name for ref in ds.refs])
    if len(set(trajectory_names.tolist())) != EXPECTED_DEV_TRAJECTORIES:
        raise RuntimeError("trajectory mapping does not contain exactly 16 Dev trajectories")

    physical_rows: list[dict] = []
    trajectory_long: list[dict] = []
    anatomy: dict[str, object] = {}
    base_raw = transfer.raw_physical_errors(kit_root, predictions["base"], target)
    for metric, expected in REGISTERED_BASELINE.items():
        if abs(base_raw[metric] - expected) > 5e-6:
            raise RuntimeError(
                f"registered @32500 baseline mismatch {metric}: observed={base_raw[metric]} expected={expected}"
            )
    target_energy = transfer._fluctuation_energy(target)

    for variant in variants:
        pred = predictions[variant]
        raw = transfer.raw_physical_errors(kit_root, pred, target)
        trajectory, variant_anatomy = core.trajectory_rows(ds, pred, target, kit_root)
        anatomy[variant] = variant_anatomy
        for row in trajectory:
            trajectory_long.append({"variant": variant, **row})
        horizon = sota.horizon_error_summary(pred, target)
        energy = transfer._fluctuation_energy(pred)
        physical_rows.append({
            "variant": variant,
            "corrector_update": 0 if variant == "base" else int(variant[1:]),
            "rel_l2": raw["rel_l2"],
            "tke": raw["tke"],
            "mvpe": raw["mvpe"],
            "rel_l2_improvement_vs_base": audit._relative_improvement(base_raw["rel_l2"], raw["rel_l2"]),
            "tke_improvement_vs_base": audit._relative_improvement(base_raw["tke"], raw["tke"]),
            "mvpe_improvement_vs_base": audit._relative_improvement(base_raw["mvpe"], raw["mvpe"]),
            "fluctuation_energy": energy,
            "target_fluctuation_energy": target_energy,
            "fluctuation_energy_ratio": energy / max(target_energy, 1e-30),
            "h19_fraction": horizon["h19"]["fraction"],
            "h20_fraction": horizon["h20"]["fraction"],
            "h19_h20_fraction": horizon["h19_h20"]["fraction"],
        })

    tables = build_analysis_tables(
        predictions=predictions,
        target=target,
        trajectory_names=trajectory_names,
    )
    expected_by_horizon = 20 * len(variants)
    expected_by_traj = EXPECTED_DEV_TRAJECTORIES * 20 * len(variants)
    expected_stability = 20 * (len(variants) - 1)
    if len(tables["by_horizon"]) != expected_by_horizon:
        raise RuntimeError(f"by_horizon rows {len(tables['by_horizon'])} != {expected_by_horizon}")
    if len(tables["by_trajectory_horizon"]) != expected_by_traj:
        raise RuntimeError(f"by_trajectory_horizon rows {len(tables['by_trajectory_horizon'])} != {expected_by_traj}")
    if len(tables["horizon_trajectory_stability"]) != expected_stability:
        raise RuntimeError(f"stability rows {len(tables['horizon_trajectory_stability'])} != {expected_stability}")

    transfer.write_rows(out_dir / "physical_metrics.csv", physical_rows)
    transfer.write_rows(out_dir / "trajectory_metrics_long.csv", trajectory_long)
    transfer.write_rows(out_dir / "by_horizon.csv", tables["by_horizon"])
    transfer.write_rows(out_dir / "by_trajectory_horizon.csv", tables["by_trajectory_horizon"])
    transfer.write_rows(out_dir / "horizon_trajectory_stability.csv", tables["horizon_trajectory_stability"])
    transfer.dump(out_dir / "trajectory_anatomy.json", anatomy)

    result = {
        "variants": variants,
        "physical_metrics": physical_rows,
        "by_horizon_rows": len(tables["by_horizon"]),
        "by_trajectory_horizon_rows": len(tables["by_trajectory_horizon"]),
        "stability_rows": len(tables["horizon_trajectory_stability"]),
    }
    transfer.dump(out_dir / "replay_summary.json", result)
    del backbone
    for model in correctors.values():
        del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run(args: argparse.Namespace) -> dict:
    validate_recipe(args)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if sota.sha256(args.manifest) != sota.MANIFEST_SHA:
        raise ValueError("frozen 50/16 manifest SHA mismatch")
    if sota.sha256(args.checkpoint_32500) != BACKBONE_SHA256:
        raise ValueError("frozen @32500 backbone SHA mismatch")
    if not (args.kit_root / "scoring.py").is_file():
        raise FileNotFoundError(args.kit_root / "scoring.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")

    # Train side only. Dev paths/loaders are deliberately not constructed here.
    train_paths = sota.split_paths(args.manifest, "train", args.data_root)
    if len(train_paths) != 50:
        raise ValueError("expected frozen Train50")
    builder, feature_config = sota.build_features(train_paths, device)

    metadata = {
        "experiment": "TMR-02_residual_corrector_full_training",
        "status": "RUNNING",
        "manifest_sha256": sota.MANIFEST_SHA,
        "backbone_update": BACKBONE_UPDATE,
        "backbone_sha256": BACKBONE_SHA256,
        "train_trajectories": 50,
        "dense_train_windows": DENSE_TRAIN_WINDOWS,
        "feature_config": vars(feature_config),
        "corrector": {
            "source": "realpde_adaptive_probe.ResidualCorrector3D + adaptive_features + corrector_loss",
            "in_channels": 42,
            "hidden": 64,
            "blocks": 2,
            "max_delta": 0.04,
            "updates": FINAL_UPDATE,
            "milestones": list(MILESTONES),
            "batch_size": BATCH_SIZE,
            "lr": CORRECTOR_LR,
            "weight_decay": CORRECTOR_WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": FINAL_UPDATE,
            "loss_weights": transfer.CORRECTOR_LOSS_WEIGHTS,
            "seed": CORRECTOR_SEED,
            "train_sampler_seed": TRAIN_SAMPLER_SEED,
            "effective_dense_epochs": effective_dense_epochs(),
        },
        "alpha_evaluated": ALPHA,
        "dev_policy": "no Dev access until the complete 30k corrector training finishes",
        "scope": {
            "backbone_training": "NOT_PERFORMED",
            "sps": "NOT_ACCESSED",
            "uncertainty": "NOT_ACCESSED",
            "full_data": "NOT_ACCESSED",
            "locked_final_private": "NOT_ACCESSED",
            "package": "NOT_BUILT",
            "codabench": "NOT_ACCESSED",
        },
    }
    transfer.dump(args.out_dir / "run_metadata.json", metadata)

    training = train_long_corrector(
        checkpoint=args.checkpoint_32500,
        train_paths=train_paths,
        kit_root=args.kit_root,
        builder=builder,
        device=device,
        out_dir=args.out_dir,
        workers=args.workers,
    )

    # Only after the full 30k campaign is complete do we resolve and load Dev16.
    dev_paths = sota.split_paths(args.manifest, "dev", args.data_root)
    if len(dev_paths) != EXPECTED_DEV_TRAJECTORIES:
        raise ValueError(f"Dev trajectories {len(dev_paths)} != {EXPECTED_DEV_TRAJECTORIES}")
    corrector_paths = {
        update: args.out_dir / f"corrector_update_{update}.pth"
        for update in MILESTONES
    }
    replay = replay_milestones(
        checkpoint=args.checkpoint_32500,
        corrector_paths=corrector_paths,
        dev_paths=dev_paths,
        train_paths=train_paths,
        kit_root=args.kit_root,
        builder=builder,
        device=device,
        out_dir=args.out_dir,
        eval_batch_size=args.eval_batch_size,
        workers=args.workers,
    )

    summary = {
        "status": "REVIEW_REQUIRED",
        "experiment": metadata["experiment"],
        "training": training,
        "replay": replay,
        "interpretation_owner": "ChatGPT/Sol",
        "automatic_go_no_go": False,
        "notes": "One full-training campaign only. No follow-on experiment is authorized automatically.",
    }
    transfer.dump(args.out_dir / "summary.json", summary)
    transfer.dump(args.out_dir / "status.json", {"state": "DONE", "status": "REVIEW_REQUIRED"})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--checkpoint-32500", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=FINAL_UPDATE)
    parser.add_argument("--milestones", type=int, nargs="+", default=list(MILESTONES))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
