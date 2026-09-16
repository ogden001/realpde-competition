#!/usr/bin/env python3
"""Two-phase teammate-style SPS experiment for frozen SOTA-V2.

Phase A keeps the SOTA-V2 @32500 point predictor frozen and trains the exact
35-channel uncertainty architecture/feature recipe extracted from the
teammate's 2026-09-15 Codabench package.  It trains for a 2000-update budget,
evaluates every 200 updates on the frozen 16-trajectory Dev split, and selects
the best SPS checkpoint on the same frozen 28-row calibration grid.

Only if Phase A passes the preregistered gate does Phase B train a full-specific
head against the frozen all-82 SOTA-V2 @53582 predictor.  Phase B reuses the
Phase-A selected update count and calibration; it does not tune on full-data
labels.  The runner may build a candidate package but never submits Codabench.
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

import realpde_sota_v2_adaptive as legacy
from sps_teammate_uncertainty_runtime import (
    TeammateUncertaintyHead,
    gaussian_nll_from_log_std,
    sigma_from_log_std,
)

SEED = legacy.SEED
HEAD_MAX_UPDATES = 2000
EVAL_INTERVAL = 200
HEAD_LR = 1e-3
HEAD_WEIGHT_DECAY = 1e-5
SIGMA0 = 0.02
EXPECTED_VALIDATION_ITERATION = legacy.EXPECTED_ITERATION
EXPECTED_FULL_ITERATION = 53_582
EXPECTED_FULL_CHECKPOINT_SHA = "f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce"
EXPECTED_FULL_CANONICAL_WINDOWS = 3_383
EXPECTED_FULL_TRAJECTORIES = 82

CURRENT_DEV_SPS = 45.07008160038756
CURRENT_DEV_MEAN_WIDTH = 0.02358330972492695
MIN_DEV_SPS_GAIN = 1.5
MAX_MEAN_WIDTH_RATIO = 1.15
MAX_PREDICTION_PARITY = 1e-7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
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


def evaluate_phase_a_gate(
    *,
    candidate_sps: float,
    candidate_mean_width: float,
    prediction_parity_max_abs: float,
) -> dict[str, object]:
    values = (candidate_sps, candidate_mean_width, prediction_parity_max_abs)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("gate inputs must be finite")
    if candidate_mean_width < 0:
        raise ValueError("candidate mean width must be non-negative")
    delta = float(candidate_sps) - CURRENT_DEV_SPS
    width_ratio = float(candidate_mean_width) / CURRENT_DEV_MEAN_WIDTH
    sps_ok = delta >= MIN_DEV_SPS_GAIN - 1e-12
    width_ok = width_ratio <= MAX_MEAN_WIDTH_RATIO + 1e-12
    parity_ok = float(prediction_parity_max_abs) <= MAX_PREDICTION_PARITY + 1e-15
    return {
        "status": "SPS_TEAMMATE_GO" if sps_ok and width_ok and parity_ok else "SPS_TEAMMATE_NO_GO",
        "baseline_sps": CURRENT_DEV_SPS,
        "candidate_sps": float(candidate_sps),
        "delta_sps": delta,
        "min_sps_gain": MIN_DEV_SPS_GAIN,
        "sps_gain_ok": sps_ok,
        "baseline_mean_width_uv": CURRENT_DEV_MEAN_WIDTH,
        "candidate_mean_width_uv": float(candidate_mean_width),
        "mean_width_ratio": width_ratio,
        "max_mean_width_ratio": MAX_MEAN_WIDTH_RATIO,
        "width_guard_ok": width_ok,
        "prediction_parity_max_abs": float(prediction_parity_max_abs),
        "max_prediction_parity": MAX_PREDICTION_PARITY,
        "prediction_parity_ok": parity_ok,
    }


def _canonical_dataset(paths: list[Path]):
    from realpde_p0_data import H5WindowDataset

    return H5WindowDataset(
        paths,
        in_steps=20,
        out_steps=20,
        stride=20,
        sub_sample=2,
        include_pressure=False,
        window_mode="fixed",
    )


def _init_head(device: torch.device) -> TeammateUncertaintyHead:
    head = TeammateUncertaintyHead(hidden=32, blocks=2, dropout=0.0, include_pressure=True).to(device)
    # The teammate artifact records sigma0=0.02.  Reproduce the corresponding
    # output-layer initialization for training; inference checkpoints load
    # their learned weights strictly.
    torch.nn.init.zeros_(head.net[-1].weight)
    torch.nn.init.constant_(head.net[-1].bias, math.log(SIGMA0))
    return head


def _collect(
    model,
    builder,
    head,
    paths: list[Path],
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
):
    loader = torch.utils.data.DataLoader(
        _canonical_dataset(paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
    )
    predictions, sigmas, targets = [], [], []
    with torch.inference_mode():
        for x, y, _, _ in loader:
            x = x.to(device, non_blocking=True)
            prediction = legacy._forward(model, builder, x)
            sigma = sigma_from_log_std(head(x, prediction))
            predictions.append(prediction.cpu().numpy().astype(np.float32))
            sigmas.append(sigma.cpu().numpy().astype(np.float32))
            targets.append(y.numpy().astype(np.float32))
    return np.concatenate(predictions), np.concatenate(sigmas), np.concatenate(targets)


def _calibrate(prediction: np.ndarray, sigma_uv: np.ndarray, target: np.ndarray, scoring):
    channels = scoring.measured_channels(target)
    rows: list[dict[str, float]] = []
    for floor, mult in legacy.FIXED_GRID:
        half_uv = floor + mult * sigma_uv
        half = np.concatenate(
            [half_uv, np.zeros(half_uv.shape[:-1] + (1,), dtype=np.float32)],
            axis=-1,
        )
        raw_sps, coverage = scoring.aggregate_sps(
            prediction, target, channels, prediction - half, prediction + half
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


def _train_with_selection(
    *,
    model,
    builder,
    train_paths: list[Path],
    dev_paths: list[Path],
    batch_size: int,
    workers: int,
    device: torch.device,
    scoring,
):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_ds = _canonical_dataset(train_paths)
    if len(train_ds) != legacy.EXPECTED_TRAIN_WINDOWS:
        raise ValueError(f"Phase-A train windows {len(train_ds)} != {legacy.EXPECTED_TRAIN_WINDOWS}")
    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
    )
    head = _init_head(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=HEAD_WEIGHT_DECAY)
    iterator = iter(loader)
    losses: list[dict[str, float]] = []
    evals: list[dict[str, object]] = []
    best_state = None
    best_eval = None
    started = time.monotonic()
    head.train()
    for update in range(1, HEAD_MAX_UPDATES + 1):
        try:
            x, y, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y, _, _ = next(iterator)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.no_grad():
            prediction = legacy._forward(model, builder, x)
        loss = gaussian_nll_from_log_std(y[..., :2], prediction[..., :2], head(x, prediction))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite head loss @ {update}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        if update == 1 or update % 100 == 0:
            losses.append({"update": update, "loss": float(loss.detach().cpu())})
        if update % EVAL_INTERVAL == 0:
            head.eval()
            pred_dev, sigma_dev, target_dev = _collect(
                model,
                builder,
                head,
                dev_paths,
                batch_size=batch_size,
                workers=workers,
                device=device,
            )
            rows, best = _calibrate(pred_dev, sigma_dev, target_dev, scoring)
            record = {"iteration": update, "best": best, "grid": rows}
            evals.append(record)
            if best_eval is None or float(best["sps"]) > float(best_eval["best"]["sps"]):
                best_eval = record
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in head.state_dict().items()
                }
            head.train()
    if best_state is None or best_eval is None:
        raise RuntimeError("no Phase-A checkpoint was selected")
    head.load_state_dict(best_state, strict=True)
    head.eval()
    return head, losses, evals, best_eval, time.monotonic() - started, len(train_ds)


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


def _train_fixed_updates(
    *,
    model,
    builder,
    paths: list[Path],
    updates: int,
    batch_size: int,
    workers: int,
    device: torch.device,
):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dataset = _canonical_dataset(paths)
    if len(dataset) != EXPECTED_FULL_CANONICAL_WINDOWS:
        raise ValueError(f"full canonical windows {len(dataset)} != {EXPECTED_FULL_CANONICAL_WINDOWS}")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
    )
    head = _init_head(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=HEAD_WEIGHT_DECAY)
    iterator = iter(loader)
    losses: list[dict[str, float]] = []
    started = time.monotonic()
    head.train()
    for update in range(1, updates + 1):
        try:
            x, y, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y, _, _ = next(iterator)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.no_grad():
            prediction = legacy._forward(model, builder, x)
        loss = gaussian_nll_from_log_std(y[..., :2], prediction[..., :2], head(x, prediction))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite full-head loss @ {update}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        if update == 1 or update % 100 == 0 or update == updates:
            losses.append({"update": update, "loss": float(loss.detach().cpu())})
    head.eval()
    return head, losses, time.monotonic() - started, len(dataset)


def run_phase_a(args: argparse.Namespace, *, device: torch.device, scoring):
    phase_dir = args.out_dir / "phase_a_50_16_teammate35"
    phase_dir.mkdir(parents=True, exist_ok=False)
    if sha256(args.manifest) != legacy.EXPECTED_MANIFEST_SHA:
        raise ValueError("frozen 50/16 manifest SHA mismatch")
    train_paths, dev_paths = legacy._split_paths(args.manifest, args.data_root)
    _, config, builder, model = legacy._load_backbone(args.validation_checkpoint, args.kit_root, device)

    baseline_head = _init_head(device).eval()
    base_pred, _, base_target = _collect(
        model,
        builder,
        baseline_head,
        dev_paths,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    raw = legacy._raw_metrics(base_pred, base_target, scoring)
    legacy.validate_replay_metrics(raw, atol=args.replay_tolerance)

    head, losses, evals, selected, elapsed, train_windows = _train_with_selection(
        model=model,
        builder=builder,
        train_paths=train_paths,
        dev_paths=dev_paths,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        scoring=scoring,
    )
    final_pred, final_sigma, final_target = _collect(
        model,
        builder,
        head,
        dev_paths,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    if not np.array_equal(final_target, base_target):
        raise RuntimeError("Dev targets changed or reordered")
    parity = float(np.max(np.abs(final_pred - base_pred)))
    rows, best = _calibrate(final_pred, final_sigma, final_target, scoring)
    expected_selected = max(evals, key=lambda row: float(row["best"]["sps"]))
    if int(selected["iteration"]) != int(expected_selected["iteration"]):
        raise RuntimeError("selected checkpoint is inconsistent with recorded evals")
    gate = evaluate_phase_a_gate(
        candidate_sps=float(best["sps"]),
        candidate_mean_width=float(best["mean_width_uv"]),
        prediction_parity_max_abs=parity,
    )

    head_path = phase_dir / "teammate35_head_best.pth"
    metadata = {
        "head_scope": "validation_teammate35",
        "recipe": "teammate35",
        "seed": SEED,
        "max_updates": HEAD_MAX_UPDATES,
        "eval_interval": EVAL_INTERVAL,
        "selected_updates": int(selected["iteration"]),
        "lr": HEAD_LR,
        "weight_decay": HEAD_WEIGHT_DECAY,
        "sigma0": SIGMA0,
        "architecture": {"in_channels": 35, "hidden": 32, "blocks": 2, "dropout": 0.0},
        "validation_checkpoint_iteration": EXPECTED_VALIDATION_ITERATION,
        "validation_checkpoint_sha256": sha256(args.validation_checkpoint),
        "manifest_sha256": legacy.EXPECTED_MANIFEST_SHA,
        "train_trajectories": len(train_paths),
        "train_windows": train_windows,
        "window_mode": "fixed",
        "feature_config": vars(config),
        "source_package": "submission_all81_probe_fp16_20260915.zip",
    }
    torch.save(
        {
            "head_state_dict": {name: value.detach().cpu() for name, value in head.state_dict().items()},
            "metadata": metadata,
        },
        head_path,
    )
    write_csv(phase_dir / "calibration_grid.csv", rows)
    dump_json(phase_dir / "calibration_grid.json", rows)
    dump_json(phase_dir / "checkpoint_evals.json", evals)
    dump_json(
        phase_dir / "head_training_summary.json",
        {
            "head": str(head_path),
            "head_sha256": sha256(head_path),
            "loss_curve": losses,
            "elapsed_seconds": elapsed,
            "selected_iteration": int(selected["iteration"]),
        },
    )
    summary = {
        "status": "REVIEW_REQUIRED",
        "gate": gate["status"],
        "gate_detail": gate,
        "best": best,
        "baseline": {"sps": CURRENT_DEV_SPS, "mean_width_uv": CURRENT_DEV_MEAN_WIDTH},
        "raw_errors": raw,
        "prediction_parity_max_abs": parity,
        "selected_iteration": int(selected["iteration"]),
        "head": str(head_path),
        "head_sha256": sha256(head_path),
        "train_windows": train_windows,
        "dev_windows": legacy.EXPECTED_DEV_WINDOWS,
        "validation_checkpoint_sha256": sha256(args.validation_checkpoint),
        "scorer_sha256": sha256(args.kit_root / "scoring.py"),
    }
    dump_json(phase_dir / "calibration_summary.json", summary)
    return summary, config


def run_phase_b(
    args: argparse.Namespace,
    phase_a: dict[str, object],
    validation_config,
    *,
    device: torch.device,
):
    if phase_a["gate"] != "SPS_TEAMMATE_GO":
        return {"status": "SKIPPED_PHASE_A_NO_GO"}
    phase_dir = args.out_dir / "phase_b_full_specific_teammate35"
    phase_dir.mkdir(parents=True, exist_ok=False)
    from realpde_sota_v2_full import released_paths

    paths = released_paths(args.data_root)
    _, full_config, builder, model, full_sha = _load_full_backbone(
        args.full_checkpoint, args.kit_root, device
    )
    if vars(full_config) != vars(validation_config):
        raise ValueError("validation/full feature configs differ")
    selected_updates = int(phase_a["selected_iteration"])
    head, losses, elapsed, train_windows = _train_fixed_updates(
        model=model,
        builder=builder,
        paths=paths,
        updates=selected_updates,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    if len(paths) != EXPECTED_FULL_TRAJECTORIES:
        raise ValueError("full trajectory count mismatch")
    head_path = phase_dir / "teammate35_full_head.pth"
    metadata = {
        "head_scope": "full_specific_teammate35",
        "recipe": "teammate35",
        "seed": SEED,
        "selected_updates": selected_updates,
        "lr": HEAD_LR,
        "weight_decay": HEAD_WEIGHT_DECAY,
        "sigma0": SIGMA0,
        "architecture": {"in_channels": 35, "hidden": 32, "blocks": 2, "dropout": 0.0},
        "backbone_checkpoint_iteration": EXPECTED_FULL_ITERATION,
        "backbone_checkpoint_sha256": full_sha,
        "train_trajectories": len(paths),
        "train_windows": train_windows,
        "window_mode": "fixed",
        "feature_config": vars(full_config),
        "calibration_source": "Phase-A frozen 50/16 Dev",
        "bound_floor": float(phase_a["best"]["floor"]),
        "bound_mult": float(phase_a["best"]["mult"]),
    }
    torch.save(
        {
            "head_state_dict": {name: value.detach().cpu() for name, value in head.state_dict().items()},
            "metadata": metadata,
        },
        head_path,
    )
    summary = {
        "status": "REVIEW_REQUIRED",
        "head": str(head_path),
        "head_sha256": sha256(head_path),
        "selected_updates": selected_updates,
        "train_trajectories": len(paths),
        "train_windows": train_windows,
        "elapsed_seconds": elapsed,
        "loss_curve": losses,
        "full_checkpoint_sha256": full_sha,
        "bounds": {"floor": metadata["bound_floor"], "mult": metadata["bound_mult"]},
    }
    dump_json(phase_dir / "full_head_summary.json", summary)
    return summary


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    sys.path.insert(0, str(args.kit_root.resolve()))
    import scoring

    phase_a, validation_config = run_phase_a(args, device=device, scoring=scoring)
    result: dict[str, object] = {
        "phase_a": phase_a,
        "phase_b": {"status": "SKIPPED_PHASE_A_NO_GO"},
        "package": None,
    }
    if phase_a["gate"] == "SPS_TEAMMATE_GO":
        phase_b = run_phase_b(args, phase_a, validation_config, device=device)
        result["phase_b"] = phase_b
        if args.package_out_root is not None:
            from build_sota_v2_teammate_package import build

            result["package"] = build(
                full_checkpoint=args.full_checkpoint,
                head_checkpoint=Path(phase_b["head"]),
                calibration_summary=(
                    args.out_dir / "phase_a_50_16_teammate35" / "calibration_summary.json"
                ),
                kit_root=args.kit_root,
                out_root=args.package_out_root,
                execution_commit=args.execution_commit,
            )
    result["status"] = "REVIEW_REQUIRED"
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
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--replay-tolerance", type=float, default=5e-6)
    parser.add_argument("--require-cuda", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
