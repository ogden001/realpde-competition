#!/usr/bin/env python3
"""Replicate the teammate-style stride=1 SPS recipe on frozen SOTA-V2.

Phase 1 is a strict 50/16 matched experiment:
- frozen SOTA-V2 validation backbone @32500
- same h32/b2 adaptive uncertainty head, Gaussian NLL, 1400 updates
- only head-training windows change from canonical stride=20 to every legal
  stride=1 window (``window_mode='dense_all'``)
- calibration remains the frozen 28-row floor x multiplier grid on the same
  16-trajectory Dev split

If Phase 1 passes the pre-registered SPS/width gate, Phase 2 trains a fresh
full-specific uncertainty head on residuals from the frozen all-82 SOTA-V2
backbone @53582, again using every legal stride=1 window.  The clean Phase-1
calibration parameters are reused; full-data calibration is forbidden.

This runner never accesses locked-final/private data and never submits to
Codabench.  It may optionally build a candidate package for later review.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

import realpde_sota_v2_adaptive as legacy
from sota_v2_adaptive_runtime import AdaptiveUncertaintyHead

SEED = legacy.SEED
HEAD_UPDATES = legacy.HEAD_UPDATES
HEAD_LR = legacy.HEAD_LR
HEAD_WEIGHT_DECAY = legacy.HEAD_WEIGHT_DECAY
EXPECTED_VALIDATION_ITERATION = legacy.EXPECTED_ITERATION
EXPECTED_FULL_ITERATION = 53_582
EXPECTED_FULL_CHECKPOINT_SHA = "f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce"
EXPECTED_PHASE1_DENSE_WINDOWS = 40_488
EXPECTED_FULL_DENSE_WINDOWS = 66_755
EXPECTED_FULL_TRAJECTORIES = 82

CURRENT_DEV_SPS = 45.07008160038756
CURRENT_DEV_COVERAGE = 0.8558713772975425
CURRENT_DEV_MEAN_WIDTH = 0.02358330972492695
MIN_DEV_SPS_GAIN = 1.5
MAX_MEAN_WIDTH_RATIO = 1.15
MAX_CORRELATION_POINTS = 200_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_replica_gate(
    *,
    candidate_sps: float,
    candidate_mean_width: float,
    baseline_sps: float = CURRENT_DEV_SPS,
    baseline_mean_width: float = CURRENT_DEV_MEAN_WIDTH,
    min_sps_gain: float = MIN_DEV_SPS_GAIN,
    max_width_ratio: float = MAX_MEAN_WIDTH_RATIO,
) -> dict[str, object]:
    """Apply the pre-registered Phase-1 gate.

    SPS must improve by at least 1.5 points and the gain may not be purchased by
    more than a 15% increase in mean UV interval width.
    """
    values = (candidate_sps, candidate_mean_width, baseline_sps, baseline_mean_width)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("gate inputs must be finite")
    if min(candidate_mean_width, baseline_mean_width) < 0:
        raise ValueError("mean widths must be non-negative")
    if baseline_mean_width == 0:
        raise ValueError("baseline mean width must be positive")
    delta = float(candidate_sps) - float(baseline_sps)
    ratio = float(candidate_mean_width) / float(baseline_mean_width)
    sps_gain_ok = delta >= float(min_sps_gain) - 1e-12
    width_guard_ok = ratio <= float(max_width_ratio) + 1e-12
    return {
        "status": "SPS_REPLICA_GO" if sps_gain_ok and width_guard_ok else "SPS_REPLICA_NO_GO",
        "baseline_sps": float(baseline_sps),
        "candidate_sps": float(candidate_sps),
        "delta_sps": delta,
        "min_sps_gain": float(min_sps_gain),
        "sps_gain_ok": bool(sps_gain_ok),
        "baseline_mean_width_uv": float(baseline_mean_width),
        "candidate_mean_width_uv": float(candidate_mean_width),
        "mean_width_ratio": ratio,
        "max_mean_width_ratio": float(max_width_ratio),
        "width_guard_ok": bool(width_guard_ok),
    }


def _dense_dataset(paths: list[Path]):
    from realpde_p0_data import H5WindowDataset

    return H5WindowDataset(
        paths,
        in_steps=20,
        out_steps=20,
        stride=20,
        sub_sample=2,
        include_pressure=False,
        window_mode="dense_all",
    )


def _dense_loader(dataset, *, batch_size: int, workers: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
    )


def _train_head(
    *,
    model,
    builder,
    dataset,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> tuple[AdaptiveUncertaintyHead, list[dict[str, float]], float]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    loader = _dense_loader(dataset, batch_size=batch_size, workers=workers)
    head = AdaptiveUncertaintyHead(in_channels=15, hidden=32, blocks=2).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=HEAD_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HEAD_UPDATES)
    iterator = iter(loader)
    losses: list[dict[str, float]] = []
    started = time.monotonic()
    head.train()
    for update in range(1, HEAD_UPDATES + 1):
        try:
            x, y, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y, _, _ = next(iterator)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.no_grad():
            pred = legacy._forward(model, builder, x)
            features = legacy.uncertainty_features(x, pred).permute(0, 4, 1, 2, 3)
        sigma = head(features).permute(0, 2, 3, 4, 1)
        loss = legacy.gaussian_nll(y[..., :2], pred[..., :2], sigma)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite head loss @ {update}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if update == 1 or update % 100 == 0 or update == HEAD_UPDATES:
            losses.append(
                {
                    "update": float(update),
                    "loss": float(loss.detach().cpu()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
            )
    head.eval()
    return head, losses, time.monotonic() - started


def _save_head(path: Path, head: AdaptiveUncertaintyHead, metadata: dict[str, object]) -> None:
    torch.save(
        {
            "head_state_dict": {name: value.detach().cpu() for name, value in head.state_dict().items()},
            "metadata": metadata,
        },
        path,
    )


def _subsample_pair(left: np.ndarray, right: np.ndarray, max_points: int = MAX_CORRELATION_POINTS) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError("correlation arrays must have the same size")
    if x.size == 0:
        return x, y
    step = max(1, math.ceil(x.size / max_points))
    return x[::step], y[::step]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("rank input must be 1-D")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def correlation_diagnostics(sigma_uv: np.ndarray, abs_error_uv: np.ndarray) -> dict[str, float | int]:
    sigma, error = _subsample_pair(sigma_uv, abs_error_uv)
    return {
        "sampled_points": int(sigma.size),
        "pearson": _corr(sigma, error),
        "spearman": _corr(_average_ranks(sigma), _average_ranks(error)),
    }


def horizon_diagnostics(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    sigma_uv: np.ndarray,
    floor: float,
    mult: float,
) -> list[dict[str, float | int]]:
    if prediction.shape != target.shape or sigma_uv.shape != prediction.shape[:-1] + (2,):
        raise ValueError("horizon diagnostic shapes are incompatible")
    half_uv = floor + mult * sigma_uv
    abs_error = np.abs(target[..., :2] - prediction[..., :2])
    rows: list[dict[str, float | int]] = []
    for horizon in range(prediction.shape[1]):
        err = abs_error[:, horizon]
        half = half_uv[:, horizon]
        rows.append(
            {
                "horizon": horizon + 1,
                "coverage": float(np.mean(err <= half)),
                "mean_width_uv": float(np.mean(2.0 * half)),
                "mae_uv": float(np.mean(err)),
                "sigma_mean": float(np.mean(sigma_uv[:, horizon])),
            }
        )
    return rows


def channel_diagnostics(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    sigma_uv: np.ndarray,
    floor: float,
    mult: float,
) -> list[dict[str, float | str]]:
    half_uv = floor + mult * sigma_uv
    abs_error = np.abs(target[..., :2] - prediction[..., :2])
    rows = []
    for channel, index in (("u", 0), ("v", 1)):
        rows.append(
            {
                "channel": channel,
                "coverage": float(np.mean(abs_error[..., index] <= half_uv[..., index])),
                "mean_width": float(np.mean(2.0 * half_uv[..., index])),
                "mae": float(np.mean(abs_error[..., index])),
                "sigma_mean": float(np.mean(sigma_uv[..., index])),
            }
        )
    return rows


def _load_full_backbone(checkpoint: Path, kit_root: Path, device: torch.device):
    from realpde_mf01 import MF01CNO
    from realpde_p0_features import P0FeatureBuilder

    actual_sha = sha256(checkpoint)
    if actual_sha != EXPECTED_FULL_CHECKPOINT_SHA:
        raise ValueError(f"full checkpoint SHA mismatch: {actual_sha}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("iteration") != EXPECTED_FULL_ITERATION or payload.get("feature_set") != "P0-A":
        raise ValueError("full checkpoint is not frozen SOTA-V2 @53582 P0-A")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not state or not all(name.startswith("cno.") for name in state):
        raise ValueError("full checkpoint is not MF01CNO state")
    config = legacy._feature_config(payload.get("feature_config"))
    builder = P0FeatureBuilder(config).to(device)
    model = MF01CNO(kit_root, len(builder.feature_names), device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return payload, config, builder, model, actual_sha


def _calibrate(
    *,
    prediction: np.ndarray,
    sigma_uv: np.ndarray,
    target: np.ndarray,
    scoring,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    channels = scoring.measured_channels(target)
    rows: list[dict[str, float]] = []
    for floor, mult in legacy.FIXED_GRID:
        half_uv = floor + mult * sigma_uv
        half = np.concatenate(
            [half_uv, np.zeros(half_uv.shape[:-1] + (1,), dtype=np.float32)],
            axis=-1,
        )
        raw_sps, coverage = scoring.aggregate_sps(
            prediction,
            target,
            channels,
            prediction - half,
            prediction + half,
        )
        rows.append(
            {
                "floor": float(floor),
                "mult": float(mult),
                "sps": legacy._score_sps(float(raw_sps), scoring),
                "coverage": float(coverage),
                "mean_width_uv": float(np.mean(2.0 * half_uv)),
            }
        )
    return rows, legacy.choose_best_calibration(rows)


def run_phase1(args: argparse.Namespace, *, device: torch.device, scoring) -> dict[str, object]:
    phase_dir = args.out_dir / "phase1_50_16_stride1"
    phase_dir.mkdir(parents=True, exist_ok=False)
    if sha256(args.manifest) != legacy.EXPECTED_MANIFEST_SHA:
        raise ValueError("frozen 50/16 manifest SHA mismatch")
    train_paths, dev_paths = legacy._split_paths(args.manifest, args.data_root)
    payload, config, builder, model = legacy._load_backbone(args.validation_checkpoint, args.kit_root, device)

    replay_head = AdaptiveUncertaintyHead().to(device).eval()
    dev_ds, dev_pred, _, dev_target = legacy._collect(
        model,
        builder,
        replay_head,
        dev_paths,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    raw = legacy._raw_metrics(dev_pred, dev_target, scoring)
    legacy.validate_replay_metrics(raw, atol=args.replay_tolerance)
    if len(dev_ds) != legacy.EXPECTED_DEV_WINDOWS:
        raise ValueError("Dev window count differs from frozen 50/16 protocol")

    train_ds = _dense_dataset(train_paths)
    if len(train_ds) != EXPECTED_PHASE1_DENSE_WINDOWS:
        raise ValueError(
            f"Phase-1 dense window audit mismatch: {len(train_ds)} != {EXPECTED_PHASE1_DENSE_WINDOWS}"
        )
    head, losses, elapsed = _train_head(
        model=model,
        builder=builder,
        dataset=train_ds,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    validation_sha = sha256(args.validation_checkpoint)
    head_path = phase_dir / "adaptive_head_stride1_1400.pth"
    _save_head(
        head_path,
        head,
        {
            "head_scope": "validation_stride1",
            "seed": SEED,
            "updates": HEAD_UPDATES,
            "lr": HEAD_LR,
            "weight_decay": HEAD_WEIGHT_DECAY,
            "architecture": {"in_channels": 15, "hidden": 32, "blocks": 2},
            "validation_checkpoint_sha256": validation_sha,
            "validation_checkpoint_iteration": EXPECTED_VALIDATION_ITERATION,
            "manifest_sha256": legacy.EXPECTED_MANIFEST_SHA,
            "train_trajectories": len(train_paths),
            "train_windows": len(train_ds),
            "window_mode": "dense_all",
            "effective_window_stride": 1,
            "feature_config": vars(config),
        },
    )
    dump_json(
        phase_dir / "head_training_summary.json",
        {
            "head": str(head_path),
            "head_sha256": sha256(head_path),
            "elapsed_seconds": elapsed,
            "loss_curve": losses,
            "train_windows": len(train_ds),
            "train_trajectories": len(train_paths),
        },
    )

    dev_ds2, dev_pred2, dev_sigma, dev_target2 = legacy._collect(
        model,
        builder,
        head,
        dev_paths,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    raw2 = legacy._raw_metrics(dev_pred2, dev_target2, scoring)
    legacy.validate_replay_metrics(raw2, atol=args.replay_tolerance)
    if not np.array_equal(dev_target, dev_target2):
        raise RuntimeError("Dev targets changed or reordered after head training")
    prediction_parity = float(np.max(np.abs(dev_pred - dev_pred2)))
    if prediction_parity > 1e-7:
        raise RuntimeError(f"uncertainty head altered backbone prediction: {prediction_parity}")

    rows, best = _calibrate(prediction=dev_pred2, sigma_uv=dev_sigma, target=dev_target2, scoring=scoring)
    gate = evaluate_replica_gate(
        candidate_sps=float(best["sps"]),
        candidate_mean_width=float(best["mean_width_uv"]),
    )
    best_floor, best_mult = float(best["floor"]), float(best["mult"])
    abs_error_uv = np.abs(dev_target2[..., :2] - dev_pred2[..., :2])
    corr = correlation_diagnostics(dev_sigma, abs_error_uv)
    horizon = horizon_diagnostics(
        prediction=dev_pred2,
        target=dev_target2,
        sigma_uv=dev_sigma,
        floor=best_floor,
        mult=best_mult,
    )
    channel = channel_diagnostics(
        prediction=dev_pred2,
        target=dev_target2,
        sigma_uv=dev_sigma,
        floor=best_floor,
        mult=best_mult,
    )
    write_csv(phase_dir / "calibration_grid.csv", [dict(row) for row in rows])
    dump_json(phase_dir / "calibration_grid.json", rows)
    write_csv(phase_dir / "by_horizon.csv", [dict(row) for row in horizon])
    write_csv(phase_dir / "by_channel.csv", [dict(row) for row in channel])
    summary = {
        "status": "REVIEW_REQUIRED",
        "gate": gate["status"],
        "gate_detail": gate,
        "baseline": {
            "sps": CURRENT_DEV_SPS,
            "coverage": CURRENT_DEV_COVERAGE,
            "mean_width_uv": CURRENT_DEV_MEAN_WIDTH,
            "training_windows": legacy.EXPECTED_TRAIN_WINDOWS,
            "training_window_mode": "fixed_stride20",
        },
        "best": best,
        "raw_errors": raw2,
        "prediction_parity_max_abs": prediction_parity,
        "sigma_error_correlation": corr,
        "head": str(head_path),
        "head_sha256": sha256(head_path),
        "train_windows": len(train_ds),
        "dev_windows": len(dev_ds2),
        "validation_checkpoint_sha256": validation_sha,
        "scorer_sha256": sha256(args.kit_root / "scoring.py"),
        "private_test_accessed": False,
        "codabench_accessed": False,
    }
    dump_json(phase_dir / "calibration_summary.json", summary)
    return summary


def run_phase2(args: argparse.Namespace, *, device: torch.device, phase1: dict[str, object]) -> dict[str, object]:
    if phase1.get("gate") != "SPS_REPLICA_GO":
        return {"status": "SKIPPED_PHASE1_NO_GO"}
    phase_dir = args.out_dir / "phase2_full_specific_stride1"
    phase_dir.mkdir(parents=True, exist_ok=False)
    import realpde_sota_v2_full as full_recipe

    full_paths = full_recipe.released_paths(args.data_root)
    if len(full_paths) != EXPECTED_FULL_TRAJECTORIES:
        raise ValueError("full released trajectory count mismatch")
    _, config, builder, model, full_sha = _load_full_backbone(args.full_checkpoint, args.kit_root, device)
    train_ds = _dense_dataset(full_paths)
    if len(train_ds) != EXPECTED_FULL_DENSE_WINDOWS:
        raise ValueError(
            f"Phase-2 dense window audit mismatch: {len(train_ds)} != {EXPECTED_FULL_DENSE_WINDOWS}"
        )
    head, losses, elapsed = _train_head(
        model=model,
        builder=builder,
        dataset=train_ds,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    best = phase1.get("best")
    if not isinstance(best, dict):
        raise ValueError("Phase-1 summary lacks best calibration")
    head_path = phase_dir / "adaptive_head_full_stride1_1400.pth"
    _save_head(
        head_path,
        head,
        {
            "head_scope": "full_specific",
            "seed": SEED,
            "updates": HEAD_UPDATES,
            "lr": HEAD_LR,
            "weight_decay": HEAD_WEIGHT_DECAY,
            "architecture": {"in_channels": 15, "hidden": 32, "blocks": 2},
            "backbone_checkpoint_sha256": full_sha,
            "backbone_checkpoint_iteration": EXPECTED_FULL_ITERATION,
            "train_trajectories": len(full_paths),
            "train_windows": len(train_ds),
            "window_mode": "dense_all",
            "effective_window_stride": 1,
            "feature_config": vars(config),
            "calibration_source": "clean fixed 50/16 Phase-1 Dev only",
            "bound_floor": float(best["floor"]),
            "bound_mult": float(best["mult"]),
            "private_test_accessed": False,
            "codabench_accessed": False,
        },
    )
    summary = {
        "status": "FULL_HEAD_READY / REVIEW_REQUIRED",
        "head": str(head_path),
        "head_sha256": sha256(head_path),
        "full_checkpoint": str(args.full_checkpoint),
        "full_checkpoint_sha256": full_sha,
        "full_checkpoint_iteration": EXPECTED_FULL_ITERATION,
        "train_windows": len(train_ds),
        "train_trajectories": len(full_paths),
        "elapsed_seconds": elapsed,
        "loss_curve": losses,
        "frozen_bounds": {"floor": float(best["floor"]), "mult": float(best["mult"])},
        "calibration_source": str(args.out_dir / "phase1_50_16_stride1" / "calibration_summary.json"),
        "private_test_accessed": False,
        "codabench_accessed": False,
    }
    dump_json(phase_dir / "full_head_summary.json", summary)
    return summary


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    if not (args.kit_root / "scoring.py").is_file():
        raise FileNotFoundError(args.kit_root / "scoring.py")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")
    sys.path.insert(0, str(args.kit_root.resolve()))
    import scoring

    phase1 = run_phase1(args, device=device, scoring=scoring)
    phase2 = run_phase2(args, device=device, phase1=phase1)
    result: dict[str, object] = {
        "status": "REVIEW_REQUIRED",
        "phase1": phase1,
        "phase2": phase2,
        "candidate_package": None,
        "codabench_accessed": False,
    }
    if phase1.get("gate") == "SPS_REPLICA_GO" and args.package_out_root is not None:
        if not isinstance(phase2, dict) or "head" not in phase2:
            raise RuntimeError("Phase-2 head is unavailable for packaging")
        from build_sota_v2_adaptive_package import build

        package = build(
            full_checkpoint=args.full_checkpoint,
            head_checkpoint=Path(str(phase2["head"])),
            calibration_summary=args.out_dir / "phase1_50_16_stride1" / "calibration_summary.json",
            kit_root=args.kit_root,
            out_root=args.package_out_root,
            execution_commit=args.execution_commit,
        )
        result["candidate_package"] = package
    dump_json(args.out_dir / "run_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-checkpoint", type=Path, required=True)
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--package-out-root", type=Path)
    parser.add_argument("--execution-commit", default="UNSET")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--replay-tolerance", type=float, default=5e-6)
    parser.add_argument("--require-cuda", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
