#!/usr/bin/env python3
"""Single-variable SPS A/B on the frozen full SOTA-V2 predictor.

Question: with the exact same full point predictor (@53582), teammate35 uncertainty
architecture, optimizer, update count and fixed interval rule, does training the
uncertainty head on only the frozen 50 Train trajectories outperform the existing
all-82-trained full-specific head on the frozen 16 Dev trajectories?

Important: the full @53582 point predictor was fitted on all released trajectories,
including the 16 Dev trajectories. Therefore this experiment isolates uncertainty-
head training scope only. It is NOT a pristine unseen-residual test for the point
predictor and must not be described as OOF or held-out point-model validation.

This runner never trains or changes the point predictor and never submits Codabench.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import realpde_sota_v2_adaptive as legacy
import realpde_sps_teammate_final as teammate

SEED = teammate.SEED
PRIMARY_UPDATES = 1600
BOUND_FLOOR = 0.0025
BOUND_MULT = 1.0


def _dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_existing_full82_head(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "head_state_dict" not in payload:
        raise ValueError("full82 head checkpoint missing head_state_dict")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("full82 head checkpoint missing metadata")
    if metadata.get("head_scope") != "full_specific_teammate35":
        raise ValueError("baseline head is not the frozen full-specific teammate35 head")
    if int(metadata.get("selected_updates", -1)) != PRIMARY_UPDATES:
        raise ValueError("baseline full82 head was not trained for the frozen 1600 updates")
    if metadata.get("backbone_checkpoint_sha256") != teammate.EXPECTED_FULL_CHECKPOINT_SHA:
        raise ValueError("baseline head backbone SHA does not match frozen full @53582")
    head = teammate._init_head(device)
    head.load_state_dict(payload["head_state_dict"], strict=True)
    head.eval()
    return head, metadata


def _train_full50_head(
    *,
    model,
    builder,
    train_paths: list[Path],
    batch_size: int,
    workers: int,
    device: torch.device,
):
    """Train the teammate35 head on exactly the frozen 50 Train split for 1600 updates."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dataset = teammate._canonical_dataset(train_paths)
    if len(dataset) != legacy.EXPECTED_TRAIN_WINDOWS:
        raise ValueError(f"50-Train windows {len(dataset)} != {legacy.EXPECTED_TRAIN_WINDOWS}")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
    )
    head = teammate._init_head(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=teammate.HEAD_LR, weight_decay=teammate.HEAD_WEIGHT_DECAY
    )
    iterator = iter(loader)
    losses: list[dict[str, float]] = []
    started = time.monotonic()
    head.train()
    for update in range(1, PRIMARY_UPDATES + 1):
        try:
            x, y, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y, _, _ = next(iterator)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.no_grad():
            prediction = legacy._forward(model, builder, x)
        loss = teammate.gaussian_nll_from_log_std(
            y[..., :2], prediction[..., :2], head(x, prediction)
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite full50 head loss @ {update}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        if update == 1 or update % 100 == 0 or update == PRIMARY_UPDATES:
            losses.append({"update": update, "loss": float(loss.detach().cpu())})
    head.eval()
    return head, losses, time.monotonic() - started, len(dataset)


def _score_fixed(
    prediction: np.ndarray,
    sigma_uv: np.ndarray,
    target: np.ndarray,
    scoring,
) -> dict[str, float]:
    half_uv = BOUND_FLOOR + BOUND_MULT * sigma_uv
    half = np.concatenate(
        [half_uv, np.zeros(half_uv.shape[:-1] + (1,), dtype=np.float32)], axis=-1
    )
    channels = scoring.measured_channels(target)
    raw_sps, coverage = scoring.aggregate_sps(
        prediction, target, channels, prediction - half, prediction + half
    )
    return {
        "sps": float(legacy._score_sps(float(raw_sps), scoring)),
        "raw_sps": float(raw_sps),
        "coverage": float(coverage),
        "mean_width_uv": float(np.mean(2.0 * half_uv)),
        "floor": BOUND_FLOOR,
        "mult": BOUND_MULT,
    }


def paired_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("paired trajectory rows are empty")
    deltas = np.asarray([float(row["delta_sps"]) for row in rows], dtype=np.float64)
    return {
        "trajectories": len(rows),
        "candidate_wins": int(np.sum(deltas > 0.0)),
        "ties": int(np.sum(deltas == 0.0)),
        "candidate_losses": int(np.sum(deltas < 0.0)),
        "median_delta_sps": float(np.median(deltas)),
        "mean_delta_sps": float(np.mean(deltas)),
        "min_delta_sps": float(np.min(deltas)),
        "max_delta_sps": float(np.max(deltas)),
    }


def _collect_per_trajectory(
    *,
    model,
    builder,
    candidate_head,
    baseline_head,
    dev_paths: list[Path],
    batch_size: int,
    workers: int,
    device: torch.device,
    scoring,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in dev_paths:
        pred_c, sigma_c, target_c = teammate._collect(
            model,
            builder,
            candidate_head,
            [path],
            batch_size=batch_size,
            workers=workers,
            device=device,
        )
        pred_b, sigma_b, target_b = teammate._collect(
            model,
            builder,
            baseline_head,
            [path],
            batch_size=batch_size,
            workers=workers,
            device=device,
        )
        if not np.array_equal(target_c, target_b):
            raise RuntimeError(f"target mismatch for {path.name}")
        parity = float(np.max(np.abs(pred_c - pred_b)))
        if parity > teammate.MAX_PREDICTION_PARITY:
            raise RuntimeError(f"point prediction parity failed for {path.name}: {parity}")
        candidate = _score_fixed(pred_c, sigma_c, target_c, scoring)
        baseline = _score_fixed(pred_b, sigma_b, target_b, scoring)
        rows.append(
            {
                "trajectory": path.name,
                "candidate_sps": candidate["sps"],
                "baseline_sps": baseline["sps"],
                "delta_sps": candidate["sps"] - baseline["sps"],
                "candidate_coverage": candidate["coverage"],
                "baseline_coverage": baseline["coverage"],
                "candidate_mean_width_uv": candidate["mean_width_uv"],
                "baseline_mean_width_uv": baseline["mean_width_uv"],
                "prediction_parity_max_abs": parity,
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    if teammate.sha256(args.manifest) != legacy.EXPECTED_MANIFEST_SHA:
        raise ValueError("frozen 50/16 manifest SHA mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    sys.path.insert(0, str(args.kit_root.resolve()))
    import scoring

    train_paths, dev_paths = legacy._split_paths(args.manifest, args.data_root)
    if len(train_paths) != 50 or len(dev_paths) != 16:
        raise ValueError("frozen manifest is not 50 Train / 16 Dev")
    _, full_config, builder, model, full_sha = teammate._load_full_backbone(
        args.full_checkpoint, args.kit_root, device
    )
    baseline_head, baseline_metadata = _load_existing_full82_head(
        args.full82_head_checkpoint, device
    )

    candidate_head, losses, elapsed, train_windows = _train_full50_head(
        model=model,
        builder=builder,
        train_paths=train_paths,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )

    pred_c, sigma_c, target_c = teammate._collect(
        model,
        builder,
        candidate_head,
        dev_paths,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    pred_b, sigma_b, target_b = teammate._collect(
        model,
        builder,
        baseline_head,
        dev_paths,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    if not np.array_equal(target_c, target_b):
        raise RuntimeError("Dev targets changed or reordered")
    parity = float(np.max(np.abs(pred_c - pred_b)))
    if parity > teammate.MAX_PREDICTION_PARITY:
        raise RuntimeError(f"point prediction parity failed: {parity}")

    candidate = _score_fixed(pred_c, sigma_c, target_c, scoring)
    baseline = _score_fixed(pred_b, sigma_b, target_b, scoring)
    raw = legacy._raw_metrics(pred_c, target_c, scoring)
    trajectory_rows = _collect_per_trajectory(
        model=model,
        builder=builder,
        candidate_head=candidate_head,
        baseline_head=baseline_head,
        dev_paths=dev_paths,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        scoring=scoring,
    )
    paired = paired_summary(trajectory_rows)

    candidate_path = args.out_dir / "teammate35_full53582_train50_head.pth"
    candidate_metadata = {
        "head_scope": "full53582_train50_teammate35",
        "recipe": "teammate35",
        "seed": SEED,
        "updates": PRIMARY_UPDATES,
        "lr": teammate.HEAD_LR,
        "weight_decay": teammate.HEAD_WEIGHT_DECAY,
        "sigma0": teammate.SIGMA0,
        "architecture": {"in_channels": 35, "hidden": 32, "blocks": 2, "dropout": 0.0},
        "backbone_checkpoint_iteration": teammate.EXPECTED_FULL_ITERATION,
        "backbone_checkpoint_sha256": full_sha,
        "manifest_sha256": legacy.EXPECTED_MANIFEST_SHA,
        "train_trajectories": len(train_paths),
        "dev_trajectories": len(dev_paths),
        "train_windows": train_windows,
        "dev_windows": legacy.EXPECTED_DEV_WINDOWS,
        "bounds": {"floor": BOUND_FLOOR, "mult": BOUND_MULT},
        "note": "Dev is held out from uncertainty-head training only; full point predictor saw all released trajectories.",
    }
    torch.save(
        {
            "head_state_dict": {
                name: value.detach().cpu() for name, value in candidate_head.state_dict().items()
            },
            "metadata": candidate_metadata,
        },
        candidate_path,
    )

    _write_csv(args.out_dir / "per_trajectory_paired.csv", trajectory_rows)
    _dump_json(args.out_dir / "per_trajectory_paired.json", trajectory_rows)
    _dump_json(
        args.out_dir / "head_training_summary.json",
        {
            "candidate_head": str(candidate_path),
            "candidate_head_sha256": teammate.sha256(candidate_path),
            "loss_curve": losses,
            "elapsed_seconds": elapsed,
            "train_windows": train_windows,
        },
    )
    result = {
        "status": "REVIEW_REQUIRED",
        "experiment": "SPS-FULL50-01",
        "scientific_question": "Does uncertainty-head Train50 beat the existing Train82 head when both use the same frozen full @53582 point predictor?",
        "primary_variable": "uncertainty_head_training_scope_only",
        "primary_updates": PRIMARY_UPDATES,
        "bounds": {"floor": BOUND_FLOOR, "mult": BOUND_MULT},
        "candidate_train_scope": "50 frozen Train trajectories / 2052 canonical windows",
        "baseline_train_scope": "82 released trajectories / 3383 canonical windows",
        "candidate": candidate,
        "baseline": baseline,
        "delta_sps": candidate["sps"] - baseline["sps"],
        "paired_trajectory_summary": paired,
        "prediction_parity_max_abs": parity,
        "raw_point_errors": raw,
        "full_checkpoint_sha256": full_sha,
        "baseline_head_checkpoint": str(args.full82_head_checkpoint),
        "baseline_head_sha256": teammate.sha256(args.full82_head_checkpoint),
        "baseline_head_metadata": baseline_metadata,
        "candidate_head": str(candidate_path),
        "candidate_head_sha256": teammate.sha256(candidate_path),
        "scorer_sha256": teammate.sha256(args.kit_root / "scoring.py"),
        "manifest_sha256": teammate.sha256(args.manifest),
        "interpretation_limit": "The 16 Dev trajectories are held out only from uncertainty-head training. The full @53582 point predictor itself was fitted on all released trajectories, so this is not pristine unseen-residual/OOF validation.",
        "codabench_accessed": False,
        "locked_final_accessed": False,
    }
    _dump_json(args.out_dir / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    parser.add_argument("--full82-head-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--require-cuda", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
