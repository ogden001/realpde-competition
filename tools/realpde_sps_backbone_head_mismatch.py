#!/usr/bin/env python3
"""Audit uncertainty-head portability between two frozen SOTA-V2 backbones.

Only the uncertainty heads are trained.  Backbone A is the preferred @30000
checkpoint from the same validation run (or the nearest earlier checkpoint),
while Backbone B is the frozen @32500 validation checkpoint.  The existing
@32500 uncertainty head is reused when supplied; it is only retrained as an
explicit fallback when no compatible artifact is found.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor

import realpde_sota_v2_adaptive as legacy
from sota_v2_adaptive_runtime import AdaptiveUncertaintyHead

SEED = legacy.SEED
HEAD_UPDATES = legacy.HEAD_UPDATES
HEAD_LR = legacy.HEAD_LR
HEAD_WEIGHT_DECAY = legacy.HEAD_WEIGHT_DECAY
EXPECTED_BACKBONE_B_ITERATION = 32_500
PREFERRED_BACKBONE_A_ITERATION = 30_000
EXPECTED_MANIFEST_SHA = legacy.EXPECTED_MANIFEST_SHA
EXPECTED_TRAIN_WINDOWS = legacy.EXPECTED_TRAIN_WINDOWS
EXPECTED_DEV_WINDOWS = legacy.EXPECTED_DEV_WINDOWS
FIXED_GRID = legacy.FIXED_GRID
CHECKPOINT_RE = re.compile(r"^model_update_(\d+)\.pth$")
COMBINATION_KEYS = (
    "A_headA_Acal",
    "B_headB_Bcal",
    "B_headA_Acal",
    "B_headA_Bcal",
)
MAX_CORRELATION_POINTS = 200_000


@dataclass(frozen=True)
class CheckpointSelection:
    path: Path
    iteration: int
    selection_rule: str


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
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_iteration(path: Path) -> int | None:
    match = CHECKPOINT_RE.match(path.name)
    return None if match is None else int(match.group(1))


def resolve_backbone_a(checkpoint_dir: Path, *, backbone_b_iteration: int) -> CheckpointSelection:
    """Prefer @30000, otherwise select the nearest checkpoint before B."""
    candidates = []
    for path in checkpoint_dir.glob("model_update_*.pth"):
        iteration = _checkpoint_iteration(path)
        if iteration is not None and iteration < backbone_b_iteration:
            candidates.append((iteration, path))
    if not candidates:
        raise ValueError(f"no earlier checkpoint before {backbone_b_iteration} in {checkpoint_dir}")
    exact = [item for item in candidates if item[0] == PREFERRED_BACKBONE_A_ITERATION]
    if exact:
        return CheckpointSelection(exact[0][1], exact[0][0], "exact_30000")
    iteration, path = max(candidates, key=lambda item: item[0])
    return CheckpointSelection(path, iteration, "nearest_earlier")


def compute_mismatch(rows: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Compare both cross-backbone combinations against matched B + Head-B."""
    missing = [key for key in COMBINATION_KEYS if key not in rows]
    if missing:
        raise ValueError(f"missing combination rows: {missing}")
    values = {key: float(rows[key]["sps"]) for key in COMBINATION_KEYS}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("combination SPS values must be finite")
    matched_b = values["B_headB_Bcal"]
    return {
        "mismatch_total": matched_b - values["B_headA_Acal"],
        "mismatch_after_recalibration": matched_b - values["B_headA_Bcal"],
    }


def _feature_config(raw: object):
    from realpde_p0_features import P0FeatureConfig

    if not isinstance(raw, dict) or "dx" not in raw or "dy" not in raw:
        raise ValueError("checkpoint lacks P0-A feature_config")
    return P0FeatureConfig(
        include_p0_a=bool(raw.get("include_p0_a", True)),
        include_p0_b=bool(raw.get("include_p0_b", False)),
        dx=float(raw["dx"]),
        dy=float(raw["dy"]),
        dt=None if raw.get("dt") is None else float(raw["dt"]),
        re_center=float(raw.get("re_center", 0.0)),
        re_scale=float(raw.get("re_scale", 1.0)),
    )


def _load_backbone(checkpoint: Path, kit_root: Path, device: torch.device, expected_iteration: int):
    from realpde_mf01 import MF01CNO
    from realpde_p0_features import P0FeatureBuilder

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("iteration") != expected_iteration or payload.get("feature_set") != "P0-A":
        raise ValueError(f"checkpoint is not P0-A iteration {expected_iteration}")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not state or not all(name.startswith("cno.") for name in state):
        raise ValueError("checkpoint is not MF01CNO state")
    config = _feature_config(payload.get("feature_config"))
    builder = P0FeatureBuilder(config).to(device)
    model = MF01CNO(kit_root, len(builder.feature_names), device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return payload, config, builder, model


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


def _train_head(*, model, builder, dataset, batch_size: int, workers: int, device: torch.device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
    )
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
            losses.append({
                "update": float(update),
                "loss": float(loss.detach().cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
            })
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


def _load_frozen_head(path: Path, *, expected_backbone_sha: str) -> tuple[AdaptiveUncertaintyHead, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata")
    state = payload.get("head_state_dict")
    if not isinstance(metadata, dict) or not isinstance(state, dict) or not state:
        raise ValueError("frozen head artifact lacks state or metadata")
    if metadata.get("updates") != HEAD_UPDATES:
        raise ValueError("frozen head updates do not match 1400")
    if metadata.get("validation_checkpoint_iteration") != EXPECTED_BACKBONE_B_ITERATION:
        raise ValueError("frozen head was not trained for validation @32500")
    if metadata.get("train_windows") != EXPECTED_TRAIN_WINDOWS:
        raise ValueError("frozen head does not use canonical 50-train windows")
    recorded_sha = metadata.get("validation_checkpoint_sha256")
    if recorded_sha is not None and recorded_sha != expected_backbone_sha:
        raise ValueError("frozen head validation checkpoint SHA does not match Backbone B")
    head = AdaptiveUncertaintyHead(in_channels=15, hidden=32, blocks=2)
    head.load_state_dict(state, strict=True)
    head.eval()
    return head, metadata


def _find_frozen_head(search_root: Path | None, *, expected_backbone_sha: str) -> Path | None:
    if search_root is None:
        return None
    candidates = []
    for path in search_root.rglob("adaptive_head_1400.pth"):
        try:
            _load_frozen_head(path, expected_backbone_sha=expected_backbone_sha)
        except (OSError, RuntimeError, ValueError):
            continue
        candidates.append(path)
    if len(candidates) > 1:
        raise ValueError(f"multiple compatible frozen heads found: {candidates}")
    return candidates[0] if candidates else None


def _collect(model, builder, head, paths: list[Path], *, batch_size: int, workers: int, device: torch.device):
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
            features = legacy.uncertainty_features(x, prediction).permute(0, 4, 1, 2, 3)
            sigma = head(features).permute(0, 2, 3, 4, 1)
            predictions.append(prediction.cpu().numpy().astype(np.float32))
            sigmas.append(sigma.cpu().numpy().astype(np.float32))
            targets.append(y.numpy().astype(np.float32))
    return np.concatenate(predictions), np.concatenate(sigmas), np.concatenate(targets)


def _calibrate(prediction: np.ndarray, sigma_uv: np.ndarray, target: np.ndarray, scoring):
    channels = scoring.measured_channels(target)
    rows: list[dict[str, float]] = []
    for floor, mult in FIXED_GRID:
        half_uv = floor + mult * sigma_uv
        half = np.concatenate([half_uv, np.zeros(half_uv.shape[:-1] + (1,), dtype=np.float32)], axis=-1)
        raw_sps, coverage = scoring.aggregate_sps(
            prediction, target, channels, prediction - half, prediction + half
        )
        rows.append({
            "floor": float(floor),
            "mult": float(mult),
            "sps": float(scoring.score_sps(float(raw_sps))),
            "coverage": float(coverage),
            "mean_width_uv": float(np.mean(2.0 * half_uv)),
        })
    best = max(rows, key=lambda row: (row["sps"], -row["mean_width_uv"]))
    return rows, best


def _row_at(rows: list[dict[str, float]], floor: float, mult: float) -> dict[str, float]:
    for row in rows:
        if row["floor"] == float(floor) and row["mult"] == float(mult):
            return row
    raise ValueError(f"bounds ({floor}, {mult}) are not in frozen grid")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _correlation(sigma_uv: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    sigma = np.asarray(sigma_uv, dtype=np.float64).reshape(-1)
    error = np.abs(target[..., :2] - prediction[..., :2]).reshape(-1)
    step = max(1, math.ceil(sigma.size / MAX_CORRELATION_POINTS))
    sigma, error = sigma[::step], error[::step]
    if sigma.size < 2 or np.std(sigma) == 0.0 or np.std(error) == 0.0:
        pearson = spearman = 0.0
    else:
        pearson = float(np.corrcoef(sigma, error)[0, 1])
        spearman = float(np.corrcoef(_average_ranks(sigma), _average_ranks(error))[0, 1])
    return {"sampled_points": int(sigma.size), "pearson": pearson, "spearman": spearman}


def _combination_row(name: str, backbone: str, head: str, calibration_source: str,
                     row: dict[str, float], corr: dict[str, float | int]) -> dict[str, object]:
    return {
        "combination": name,
        "backbone": backbone,
        "head": head,
        "calibration_source": calibration_source,
        **row,
        "sigma_error_pearson": corr["pearson"],
        "sigma_error_spearman": corr["spearman"],
        "sigma_error_sampled_points": corr["sampled_points"],
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if sha256(args.manifest) != EXPECTED_MANIFEST_SHA:
        raise ValueError("frozen 50/16 manifest SHA mismatch")
    if not args.backbone_b.is_file():
        raise FileNotFoundError(args.backbone_b)
    args.out_dir.mkdir(parents=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")
    sys.path.insert(0, str(args.kit_root.resolve()))
    import scoring

    train_paths, dev_paths = legacy._split_paths(args.manifest, args.data_root)
    if args.backbone_a is None:
        selection = resolve_backbone_a(args.backbone_b.parent, backbone_b_iteration=EXPECTED_BACKBONE_B_ITERATION)
        backbone_a = selection.path
        a_selection = {"rule": selection.selection_rule, "iteration": selection.iteration}
    else:
        backbone_a = args.backbone_a
        iteration = _checkpoint_iteration(backbone_a)
        if iteration is None:
            raise ValueError("explicit Backbone A must be named model_update_<iteration>.pth")
        a_selection = {"rule": "explicit", "iteration": iteration}
    if not backbone_a.is_file():
        raise FileNotFoundError(backbone_a)
    if _checkpoint_iteration(args.backbone_b) != EXPECTED_BACKBONE_B_ITERATION:
        raise ValueError("Backbone B must be model_update_32500.pth")

    payload_a, config_a, builder_a, model_a = _load_backbone(
        backbone_a, args.kit_root, device, int(a_selection["iteration"])
    )
    payload_b, config_b, builder_b, model_b = _load_backbone(
        args.backbone_b, args.kit_root, device, EXPECTED_BACKBONE_B_ITERATION
    )
    if vars(config_a) != vars(config_b):
        raise ValueError("Backbone A/B feature configs differ; mismatch audit is not comparable")
    backbone_a_sha, backbone_b_sha = sha256(backbone_a), sha256(args.backbone_b)
    train_ds = _canonical_dataset(train_paths)
    dev_ds = _canonical_dataset(dev_paths)
    if len(train_ds) != EXPECTED_TRAIN_WINDOWS or len(dev_ds) != EXPECTED_DEV_WINDOWS:
        raise ValueError("canonical 50/16 window counts differ from frozen protocol")

    head_a, losses_a, elapsed_a = _train_head(
        model=model_a, builder=builder_a, dataset=train_ds,
        batch_size=args.batch_size, workers=args.workers, device=device,
    )
    head_a_path = args.out_dir / "head_a_1400.pth"
    _save_head(head_a_path, head_a, {
        "head_scope": "backbone_a_matched",
        "seed": SEED,
        "updates": HEAD_UPDATES,
        "lr": HEAD_LR,
        "weight_decay": HEAD_WEIGHT_DECAY,
        "architecture": {"in_channels": 15, "hidden": 32, "blocks": 2},
        "backbone_checkpoint_sha256": backbone_a_sha,
        "backbone_checkpoint_iteration": int(a_selection["iteration"]),
        "manifest_sha256": EXPECTED_MANIFEST_SHA,
        "train_trajectories": len(train_paths),
        "train_windows": len(train_ds),
        "window_mode": "fixed",
        "effective_window_stride": 20,
        "feature_config": vars(config_a),
    })

    frozen_head_path = args.frozen_head
    if frozen_head_path is None:
        frozen_head_path = _find_frozen_head(args.frozen_head_search_root, expected_backbone_sha=backbone_b_sha)
    head_b_reused = frozen_head_path is not None
    if head_b_reused:
        head_b, head_b_meta = _load_frozen_head(frozen_head_path, expected_backbone_sha=backbone_b_sha)
        head_b = head_b.to(device).eval()
        losses_b, elapsed_b = [], 0.0
    else:
        head_b, losses_b, elapsed_b = _train_head(
            model=model_b, builder=builder_b, dataset=train_ds,
            batch_size=args.batch_size, workers=args.workers, device=device,
        )
        head_b_path = args.out_dir / "head_b_1400_fallback.pth"
        _save_head(head_b_path, head_b, {
            "head_scope": "backbone_b_fallback",
            "seed": SEED,
            "updates": HEAD_UPDATES,
            "lr": HEAD_LR,
            "weight_decay": HEAD_WEIGHT_DECAY,
            "architecture": {"in_channels": 15, "hidden": 32, "blocks": 2},
            "backbone_checkpoint_sha256": backbone_b_sha,
            "backbone_checkpoint_iteration": EXPECTED_BACKBONE_B_ITERATION,
            "manifest_sha256": EXPECTED_MANIFEST_SHA,
            "train_trajectories": len(train_paths),
            "train_windows": len(train_ds),
            "window_mode": "fixed",
            "effective_window_stride": 20,
            "feature_config": vars(config_b),
        })
        head_b_meta = {"reused_frozen_head": False, "fallback_head": str(head_b_path)}

    pred_a, sigma_a, target_a = _collect(
        model_a, builder_a, head_a, dev_paths,
        batch_size=args.batch_size, workers=args.workers, device=device,
    )
    pred_b, sigma_b, target_b = _collect(
        model_b, builder_b, head_b, dev_paths,
        batch_size=args.batch_size, workers=args.workers, device=device,
    )
    pred_b_cross, sigma_b_cross, target_b_cross = _collect(
        model_b, builder_b, head_a, dev_paths,
        batch_size=args.batch_size, workers=args.workers, device=device,
    )
    if not np.array_equal(target_a, target_b) or not np.array_equal(target_b, target_b_cross):
        raise RuntimeError("Dev targets changed or reordered between combinations")
    if np.max(np.abs(pred_b - pred_b_cross)) > 1e-7:
        raise RuntimeError("Head-A changed Backbone B prediction")

    rows_a, best_a = _calibrate(pred_a, sigma_a, target_a, scoring)
    rows_b, best_b = _calibrate(pred_b, sigma_b, target_b, scoring)
    rows_b_cross, best_b_cross = _calibrate(pred_b_cross, sigma_b_cross, target_b_cross, scoring)
    combo_rows = [
        _combination_row("A_headA_Acal", "A@%d" % a_selection["iteration"], "Head-A", "A", best_a,
                         _correlation(sigma_a, target_a, pred_a)),
        _combination_row("B_headB_Bcal", "B@32500", "Head-B", "B", best_b,
                         _correlation(sigma_b, target_b, pred_b)),
        _combination_row("B_headA_Acal", "B@32500", "Head-A", "A", _row_at(rows_b_cross, best_a["floor"], best_a["mult"]),
                         _correlation(sigma_b_cross, target_b_cross, pred_b_cross)),
        _combination_row("B_headA_Bcal", "B@32500", "Head-A", "B-on-Head-A", best_b_cross,
                         _correlation(sigma_b_cross, target_b_cross, pred_b_cross)),
    ]
    combo_map = {row["combination"]: row for row in combo_rows}
    mismatch = compute_mismatch(combo_map)
    head_a_summary = {
        "path": str(head_a_path),
        "sha256": sha256(head_a_path),
        "reused_frozen_head": False,
        "loss_curve": losses_a,
        "elapsed_seconds": elapsed_a,
    }
    if head_b_reused:
        head_b_summary = {
            "path": str(frozen_head_path),
            "sha256": sha256(frozen_head_path),
            "reused_frozen_head": True,
            "metadata": head_b_meta,
            "loss_curve": [],
            "elapsed_seconds": 0.0,
        }
    else:
        head_b_summary = {
            "path": str(head_b_path),
            "sha256": sha256(head_b_path),
            "reused_frozen_head": False,
            "metadata": head_b_meta,
            "loss_curve": losses_b,
            "elapsed_seconds": elapsed_b,
        }
    grid_rows = []
    for source, rows in (("A_headA", rows_a), ("B_headB", rows_b), ("B_headA", rows_b_cross)):
        grid_rows.extend({"grid_source": source, **row} for row in rows)
    write_csv(args.out_dir / "calibration_grids.csv", grid_rows)
    write_csv(args.out_dir / "combinations.csv", combo_rows)
    provenance = {
        "execution_commit": args.execution_commit,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "scorer_sha256": sha256(args.kit_root / "scoring.py"),
        "backbone_a": {"path": str(backbone_a), "iteration": a_selection["iteration"],
                       "selection_rule": a_selection["rule"], "sha256": backbone_a_sha},
        "backbone_b": {"path": str(args.backbone_b), "iteration": EXPECTED_BACKBONE_B_ITERATION,
                       "sha256": backbone_b_sha},
        "head_a": head_a_summary,
        "head_b": head_b_summary,
        "head_training": {"seed": SEED, "updates": HEAD_UPDATES, "lr": HEAD_LR,
                           "weight_decay": HEAD_WEIGHT_DECAY, "window_mode": "fixed", "stride": 20,
                           "train_windows": len(train_ds)},
        "locked_final_accessed": False,
        "private_test_accessed": False,
        "codabench_accessed": False,
    }
    dump_json(args.out_dir / "head_provenance.json", provenance)
    summary = {
        "status": "REVIEW_REQUIRED",
        "combinations": combo_rows,
        "mismatch": mismatch,
        "backbone_b_prediction_parity_head_swap_max_abs": float(np.max(np.abs(pred_b - pred_b_cross))),
        "dev_windows": len(dev_ds),
        "dev_trajectories": len(dev_paths),
        "grid_rows_per_source": len(FIXED_GRID),
        "head_b_reused_frozen": head_b_reused,
        "provenance": str(args.out_dir / "head_provenance.json"),
        "locked_final_accessed": False,
        "private_test_accessed": False,
        "codabench_accessed": False,
    }
    dump_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backbone-b", type=Path, required=True)
    parser.add_argument("--backbone-a", type=Path)
    parser.add_argument("--frozen-head", type=Path)
    parser.add_argument("--frozen-head-search-root", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--execution-commit", default="UNSET")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--require-cuda", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
