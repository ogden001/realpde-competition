#!/usr/bin/env python3
"""TMR-01: transfer the frozen teammate-style residual corrector onto SOTA-V2.

Scientific question
-------------------
Does the already-implemented 42-channel h64/b2 residual corrector add physical
metric value on top of the strong SOTA-V2 backbone, and is that value still
present after Stage-B?

This runner deliberately trains BOTH correctors first, without reading Dev,
and evaluates everything only after both fixed 2400-update trainings finish.
There is no SPS, uncertainty, package, full-data, locked-final, or Codabench
path in this file.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

import realpde_loss_official_v9 as core
import realpde_sota_v2_integrated as sota
from realpde_adaptive_probe import ResidualCorrector3D, adaptive_features, corrector_loss

CORRECTOR_UPDATES = 2400
CORRECTOR_LR = 1e-4
CORRECTOR_WEIGHT_DECAY = 1e-5
CORRECTOR_SEED = 20260905
TRAIN_SAMPLER_SEED = 20260901
ALPHAS = (0.0, 0.5, 1.0)
BACKBONE_UPDATES = (30_000, 32_500)
REGISTERED_BASELINES = {
    30_000: {"rel_l2": 0.105989, "tke": 0.474373, "mvpe": 0.090337},
    32_500: {"rel_l2": 0.099935, "tke": 0.469293, "mvpe": 0.075778},
}
CORRECTOR_LOSS_WEIGHTS = {
    "point": 1.0,
    "mse": 0.05,
    "tke": 0.12,
    "temporal": 0.04,
    "grad": 0.02,
    "p_zero": 0.01,
    "residual_mse": 0.15,
    "delta_penalty": 0.05,
}


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_rows(path: Path, values: list[dict]) -> None:
    if not values:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in values for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def apply_scaled_correction(base: Tensor, delta: Tensor, alpha: float) -> Tensor:
    if alpha not in ALPHAS:
        raise ValueError(f"alpha must be one of {ALPHAS}")
    if base.shape != delta.shape or base.ndim != 5 or base.shape[-1] != 3:
        raise ValueError("base/delta must match [B,T,H,W,3]")
    out = base + float(alpha) * delta
    out = out.clone()
    out[..., 2] = base[..., 2]
    return out


def corrector_objective(base: Tensor, target: Tensor, delta: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    parts = corrector_loss(base, target, delta)
    total = sum(CORRECTOR_LOSS_WEIGHTS[name] * value for name, value in parts.items())
    return total, parts


def raw_physical_errors(kit_root: Path, pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute only the three physical raw errors from the official scorer.

    Deliberately bypasses final-score, time, SPS, bounds, and uncertainty code.
    """
    scorer_path = kit_root / "scoring.py"
    spec = importlib.util.spec_from_file_location("tmr01_official_scoring", scorer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official scorer from {scorer_path}")
    scoring = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scoring)
    measured_channels = scoring.measured_channels(target)
    return {
        "rel_l2": float(np.mean(scoring.rel_l2_per_sample(pred, target, measured_channels))),
        "tke": float(np.mean(scoring.tke_rel_l2_per_sample(pred, target, measured_channels))),
        "mvpe": float(scoring.mvpe_rel_l2(pred, target)),
    }


def _checkpoint_payload(path: Path, expected_update: int) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("iteration", -1)) != expected_update:
        raise ValueError(f"checkpoint {path} iteration != {expected_update}")
    if payload.get("feature_set") != "P0-A":
        raise ValueError(f"checkpoint {path} is not P0-A")
    return payload


def load_backbone(path: Path, expected_update: int, kit_root: Path, builder, device: torch.device):
    payload = _checkpoint_payload(path, expected_update)
    model = sota.MF01CNO(kit_root, len(builder.feature_names), device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    freeze_module(model)
    return model, payload


def _delta_from_corrector(corrector: ResidualCorrector3D, x: Tensor, base: Tensor) -> Tensor:
    features = adaptive_features(x, base).permute(0, 4, 1, 2, 3).contiguous()
    return corrector(features).permute(0, 2, 3, 4, 1).contiguous()


def train_one_corrector(*, update: int, checkpoint: Path, train_paths: list[Path], kit_root: Path,
                        builder, device: torch.device, out_dir: Path, batch_size: int, workers: int) -> dict:
    """Train exactly one fixed-budget corrector. Dev is never touched here."""
    torch.manual_seed(CORRECTOR_SEED)
    np.random.seed(CORRECTOR_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(CORRECTOR_SEED)

    backbone, payload = load_backbone(checkpoint, update, kit_root, builder, device)
    corrector = ResidualCorrector3D(in_channels=42, hidden=64, blocks=2, max_delta=0.04).to(device)
    optimizer = torch.optim.AdamW(corrector.parameters(), lr=CORRECTOR_LR, weight_decay=CORRECTOR_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CORRECTOR_UPDATES)

    loader_args = Namespace(seed=TRAIN_SAMPLER_SEED, micro_batch=batch_size, workers=workers)
    train_ds, sampler, loader = sota.dense_loader(train_paths, loader_args)
    if len(train_ds) != sota.EXPECTED["dense_train"]:
        raise ValueError(f"Dense-All train windows {len(train_ds)} != {sota.EXPECTED['dense_train']}")

    iterator = iter(loader)
    epoch = 0
    log_path = out_dir / f"train_corrector_{update}.jsonl"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        for step in range(1, CORRECTOR_UPDATES + 1):
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
            delta = _delta_from_corrector(corrector, x, base)
            total, parts = corrector_objective(base, y, delta)
            if not torch.isfinite(total):
                raise FloatingPointError(f"nonfinite corrector loss @ backbone={update} step={step}")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(corrector.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            row = {
                "step": step,
                "loss": float(total.detach().cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                **{f"loss_{name}": float(value.detach().cpu()) for name, value in parts.items()},
            }
            log.write(json.dumps(row, sort_keys=True) + "\n")
            if step % 100 == 0 or step == CORRECTOR_UPDATES:
                log.flush()
                print(json.dumps({"backbone_update": update, **row}), flush=True)

    ckpt = out_dir / f"corrector_backbone_{update}.pth"
    torch.save({
        "corrector_state_dict": corrector.state_dict(),
        "backbone_update": update,
        "backbone_sha256": sota.sha256(checkpoint),
        "updates": CORRECTOR_UPDATES,
        "architecture": {"in_channels": 42, "hidden": 64, "blocks": 2, "max_delta": 0.04},
        "optimizer": {"name": "AdamW", "lr": CORRECTOR_LR, "weight_decay": CORRECTOR_WEIGHT_DECAY,
                      "scheduler": "CosineAnnealingLR"},
        "loss_weights": CORRECTOR_LOSS_WEIGHTS,
        "seed": CORRECTOR_SEED,
        "train_sampler_seed": TRAIN_SAMPLER_SEED,
    }, ckpt)
    result = {
        "backbone_update": update,
        "backbone_sha256": sota.sha256(checkpoint),
        "corrector_checkpoint": str(ckpt),
        "corrector_sha256": sota.sha256(ckpt),
        "updates": CORRECTOR_UPDATES,
        "dense_train_windows": len(train_ds),
        "training_wall_seconds": time.monotonic() - started,
    }
    dump(out_dir / f"training_summary_{update}.json", result)
    del backbone, corrector, optimizer, scheduler, payload
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _fluctuation_energy(array: np.ndarray) -> float:
    uv = array[..., :2].astype(np.float64)
    fluct = uv - uv.mean(axis=1, keepdims=True)
    return float(0.5 * np.square(fluct).mean(axis=(0, 1, 2, 3)).sum())


def _relative_improvement(base: float, candidate: float) -> float:
    return 1.0 - candidate / base


@torch.no_grad()
def evaluate_one_backbone(*, update: int, checkpoint: Path, corrector_checkpoint: Path,
                          dev_paths: list[Path], kit_root: Path, builder, device: torch.device,
                          out_dir: Path, eval_batch_size: int, workers: int) -> list[dict]:
    backbone, _ = load_backbone(checkpoint, update, kit_root, builder, device)
    corrector_payload = torch.load(corrector_checkpoint, map_location="cpu", weights_only=False)
    corrector = ResidualCorrector3D(in_channels=42, hidden=64, blocks=2, max_delta=0.04).to(device)
    corrector.load_state_dict(corrector_payload["corrector_state_dict"], strict=True)
    freeze_module(corrector)

    ds, loader = sota.dev_loader(dev_paths, Namespace(eval_batch_size=eval_batch_size, workers=workers))
    preds: dict[float, list[np.ndarray]] = {alpha: [] for alpha in ALPHAS}
    targets: list[np.ndarray] = []
    delta_sq_sum = base_sq_sum = 0.0
    delta_count = base_count = 0
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        base = sota.forward_mf(backbone, builder, x)
        delta = _delta_from_corrector(corrector, x, base)
        delta_sq_sum += float(delta[..., :2].double().square().sum().cpu())
        base_sq_sum += float(base[..., :2].double().square().sum().cpu())
        delta_count += delta[..., :2].numel()
        base_count += base[..., :2].numel()
        for alpha in ALPHAS:
            preds[alpha].append(apply_scaled_correction(base, delta, alpha).cpu().numpy())
        targets.append(y.numpy())

    target = np.concatenate(targets).astype(np.float32)
    target_energy = _fluctuation_energy(target)
    full_delta_rms = (delta_sq_sum / max(delta_count, 1)) ** 0.5
    base_rms = (base_sq_sum / max(base_count, 1)) ** 0.5
    results: list[dict] = []
    trajectory_long: list[dict] = []

    for alpha in ALPHAS:
        pred = np.concatenate(preds[alpha]).astype(np.float32)
        if not np.isfinite(pred).all() or float(np.abs(pred[..., 2]).max()) != 0.0:
            raise FloatingPointError(f"invalid prediction backbone={update} alpha={alpha}")
        variant = f"a{str(alpha).replace('.', 'p')}"
        variant_dir = out_dir / f"eval_{update}_{variant}"
        variant_dir.mkdir(parents=True, exist_ok=True)
        raw = raw_physical_errors(kit_root, pred, target)
        trajectory, anatomy = core.trajectory_rows(ds, pred, target, kit_root)
        for row in trajectory:
            trajectory_long.append({"backbone_update": update, "alpha": alpha, **row})
        horizon = sota.horizon_error_summary(pred, target)
        dump(variant_dir / "horizon_error_summary.json", horizon)
        dump(variant_dir / "trajectory_anatomy.json", anatomy)
        write_rows(variant_dir / "trajectory_metrics.csv", trajectory)
        energy = _fluctuation_energy(pred)
        result = {
            "backbone_update": update,
            "alpha": alpha,
            "rel_l2": raw["rel_l2"],
            "tke": raw["tke"],
            "mvpe": raw["mvpe"],
            "fluctuation_energy": energy,
            "target_fluctuation_energy": target_energy,
            "fluctuation_energy_ratio": energy / max(target_energy, 1e-30),
            "h19_fraction": horizon["h19"]["fraction"],
            "h20_fraction": horizon["h20"]["fraction"],
            "h19_h20_fraction": horizon["h19_h20"]["fraction"],
            "full_delta_rms": full_delta_rms,
            "delta_to_base_rms": full_delta_rms / max(base_rms, 1e-30),
        }
        results.append(result)
        dump(variant_dir / "physical_summary.json", result)

    baseline = next(row for row in results if row["alpha"] == 0.0)
    expected = REGISTERED_BASELINES[update]
    for metric in ("rel_l2", "tke", "mvpe"):
        if abs(baseline[metric] - expected[metric]) > 5e-4:
            raise RuntimeError(
                f"registered SOTA-V2 baseline mismatch @{update} {metric}: "
                f"observed={baseline[metric]} expected={expected[metric]}"
            )
    for row in results:
        row["rel_l2_improvement_vs_base"] = _relative_improvement(baseline["rel_l2"], row["rel_l2"])
        row["tke_improvement_vs_base"] = _relative_improvement(baseline["tke"], row["tke"])
        row["mvpe_improvement_vs_base"] = _relative_improvement(baseline["mvpe"], row["mvpe"])

    write_rows(out_dir / f"trajectory_metrics_long_{update}.csv", trajectory_long)
    del backbone, corrector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def run(args: argparse.Namespace) -> dict:
    if args.updates != CORRECTOR_UPDATES:
        raise ValueError(f"TMR-01 fixes corrector updates at {CORRECTOR_UPDATES}")
    if args.batch_size != 8:
        raise ValueError("TMR-01 fixes corrector batch size at 8")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if sota.sha256(args.manifest) != sota.MANIFEST_SHA:
        raise ValueError("frozen 50/16 manifest SHA mismatch")
    if not (args.kit_root / "scoring.py").is_file():
        raise FileNotFoundError(args.kit_root / "scoring.py")

    train_paths = sota.split_paths(args.manifest, "train", args.data_root)
    dev_paths = sota.split_paths(args.manifest, "dev", args.data_root)
    if (len(train_paths), len(dev_paths)) != (50, 16):
        raise ValueError("expected frozen 50/16 split")
    canonical = sota.H5WindowDataset(train_paths, window_mode="fixed")
    dev_ds = sota.H5WindowDataset(dev_paths, window_mode="fixed")
    if (len(canonical), len(dev_ds)) != (2052, 659):
        raise ValueError("canonical train/dev window audit mismatch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")
    builder, feature_config = sota.build_features(train_paths, device)

    checkpoints = {30_000: args.checkpoint_30000, 32_500: args.checkpoint_32500}
    for update, path in checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        _checkpoint_payload(path, update)

    metadata = {
        "experiment": "TMR-01_SOTA-V2_x_teammate_residual_corrector",
        "status": "RUNNING",
        "manifest_sha256": sota.MANIFEST_SHA,
        "train_trajectories": 50,
        "dev_trajectories": 16,
        "canonical_train_windows": 2052,
        "dense_train_windows": 40488,
        "dev_windows": 659,
        "backbones": {str(k): {"path": str(v), "sha256": sota.sha256(v)} for k, v in checkpoints.items()},
        "feature_config": vars(feature_config),
        "corrector_source": "realpde_adaptive_probe.ResidualCorrector3D + adaptive_features + corrector_loss",
        "corrector": {"in_channels": 42, "hidden": 64, "blocks": 2, "max_delta": 0.04,
                      "updates": CORRECTOR_UPDATES, "batch_size": 8, "lr": CORRECTOR_LR,
                      "weight_decay": CORRECTOR_WEIGHT_DECAY, "loss_weights": CORRECTOR_LOSS_WEIGHTS,
                      "seed": CORRECTOR_SEED},
        "alphas_evaluated": list(ALPHAS),
        "dev_policy": "no Dev access until both fixed-budget correctors finish training",
        "physical_metric_path_only": True,
        "sps_accessed": False,
        "uncertainty_accessed": False,
        "full_data_accessed": False,
        "locked_final_accessed": False,
        "codabench_accessed": False,
    }
    dump(args.out_dir / "run_metadata.json", metadata)

    # Train both fixed-budget correctors first. No Dev evaluation occurs in this loop.
    training_summaries = []
    for update in BACKBONE_UPDATES:
        training_summaries.append(train_one_corrector(
            update=update,
            checkpoint=checkpoints[update],
            train_paths=train_paths,
            kit_root=args.kit_root,
            builder=builder,
            device=device,
            out_dir=args.out_dir,
            batch_size=args.batch_size,
            workers=args.workers,
        ))

    # Only after BOTH trainings are complete do we touch Dev and collect all six points.
    physical_rows: list[dict] = []
    for update in BACKBONE_UPDATES:
        physical_rows.extend(evaluate_one_backbone(
            update=update,
            checkpoint=checkpoints[update],
            corrector_checkpoint=args.out_dir / f"corrector_backbone_{update}.pth",
            dev_paths=dev_paths,
            kit_root=args.kit_root,
            builder=builder,
            device=device,
            out_dir=args.out_dir,
            eval_batch_size=args.eval_batch_size,
            workers=args.workers,
        ))

    write_rows(args.out_dir / "physical_metrics.csv", physical_rows)
    summary = {
        "status": "REVIEW_REQUIRED",
        "experiment": metadata["experiment"],
        "training": training_summaries,
        "physical_metrics": physical_rows,
        "interpretation_owner": "ChatGPT/Sol",
        "automatic_go_no_go": False,
        "notes": "Evidence only. No SPS/full-data/package/Codabench action is authorized by this run.",
    }
    dump(args.out_dir / "summary.json", summary)
    dump(args.out_dir / "status.json", {"state": "DONE", "status": "REVIEW_REQUIRED"})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--checkpoint-30000", type=Path, required=True)
    parser.add_argument("--checkpoint-32500", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=CORRECTOR_UPDATES)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
