#!/usr/bin/env python3
"""Dev-only audit of adaptive sigma against realized prediction error.

This diagnostic never trains a model, reads only the frozen Dev split, and
keeps the current SOTA bounds fixed at ``0.0025 + 1.0 * sigma``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent

CURRENT_FLOOR = 0.0025
CURRENT_MULT = 1.0
CHANNEL_NAMES = ("u", "v")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rankdata_average(values: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks with deterministic tie handling."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    boundaries = np.concatenate(
        ([0], np.flatnonzero(sorted_x[1:] != sorted_x[:-1]) + 1, [x.size])
    )
    counts = np.diff(boundaries)
    average_ranks = 0.5 * ((boundaries[:-1] + 1) + boundaries[1:])
    sorted_ranks = np.repeat(average_ranks, counts)
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = sorted_ranks
    return ranks


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, returning NaN for degenerate inputs."""
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("correlation inputs must have identical shapes")
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denom <= np.finfo(np.float64).eps:
        return float("nan")
    return float(np.dot(x, y) / denom)


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation using average ranks for ties."""
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("correlation inputs must have identical shapes")
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2:
        return float("nan")
    return safe_pearson(rankdata_average(x), rankdata_average(y))


def sigma_decile_rows(
    sigma: np.ndarray,
    abs_error: np.ndarray,
    inside: np.ndarray,
    *,
    bins: int = 10,
) -> list[dict[str, float | int]]:
    """Summarize realized errors by sigma quantile without changing bounds."""
    s = np.asarray(sigma, dtype=np.float64).reshape(-1)
    e = np.asarray(abs_error, dtype=np.float64).reshape(-1)
    covered = np.asarray(inside, dtype=bool).reshape(-1)
    if not (s.shape == e.shape == covered.shape):
        raise ValueError("sigma/error/inside must have identical flattened shapes")
    finite = np.isfinite(s) & np.isfinite(e)
    s, e, covered = s[finite], e[finite], covered[finite]
    if s.size == 0:
        raise ValueError("sigma audit received no finite elements")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    edges = np.quantile(s, np.linspace(0.0, 1.0, bins + 1))
    rows: list[dict[str, float | int]] = []
    for idx in range(bins):
        if idx == 0:
            mask = s <= edges[1]
        elif idx == bins - 1:
            mask = s > edges[idx]
        else:
            mask = (s > edges[idx]) & (s <= edges[idx + 1])
        count = int(mask.sum())
        rows.append(
            {
                "decile": idx + 1,
                "sigma_q_lo": float(edges[idx]),
                "sigma_q_hi": float(edges[idx + 1]),
                "mean_sigma": float(np.mean(s[mask])) if count else float("nan"),
                "mean_abs_error": float(np.mean(e[mask])) if count else float("nan"),
                "coverage": float(np.mean(covered[mask])) if count else float("nan"),
                "count": count,
            }
        )
    return rows


def _normalized_penalties(pred: np.ndarray, target: np.ndarray, c: int, scoring: Any) -> dict[str, np.ndarray]:
    normalize_factor = 0.5
    dm = scoring.rel_l2_per_sample(pred, target, c)
    tke = scoring.tke_rel_l2_per_sample(pred, target, c)
    mvpe = scoring.mvpe_rel_l2_per_sample(pred, target)
    return {
        "dm": dm / (normalize_factor + dm),
        "tke": tke / (normalize_factor + tke),
        "mvpe": mvpe / (normalize_factor + mvpe),
    }


def _slice_sps_score(
    abs_error: np.ndarray,
    half_width: np.ndarray,
    penalties: dict[str, np.ndarray],
    scoring: Any,
) -> float:
    """Official-SPS decomposition restricted to one horizon/channel slice."""
    error = np.asarray(abs_error, dtype=np.float64)
    half = np.asarray(half_width, dtype=np.float64)
    if error.shape != half.shape or error.ndim != 3:
        raise ValueError("slice error/half_width must both be [N,H,W]")
    inside = error <= half
    nil = (2.0 * half) / float(scoring.SIGMA_GLOBAL)

    def branch(pm: np.ndarray) -> float:
        shaped = np.asarray(pm, dtype=np.float64).reshape(-1, 1, 1)
        elem = (1.0 - shaped) * np.exp(-nil)
        elem = np.where(inside, elem, 0.0)
        elem = np.where(np.isfinite(shaped), elem, np.nan)
        value = float(np.nanmean(elem))
        return value if np.isfinite(value) else 0.0

    raw = 0.5 * branch(penalties["dm"]) + 0.3 * branch(penalties["tke"]) + 0.2 * branch(penalties["mvpe"])
    return float(scoring.score_sps(raw))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(
    *,
    data_root: Path,
    kit_root: Path,
    checkpoint: Path,
    manifest: Path,
    validation_probe: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Run the fixed current-SOTA sigma/error audit on frozen Dev only."""
    import torch

    sys.path.insert(0, str(HERE))
    import realpde_b1_p0a_n2 as base_api
    from realpde_adaptive_probe import (
        AdaptiveUncertaintyHead,
        assert_feature_config_matches_checkpoint,
        feature_config_from_checkpoint,
        flow_features,
    )
    from realpde_p0_data import H5WindowDataset
    from realpde_p0_features import P0FeatureBuilder

    spec = json.loads(manifest.read_text(encoding="utf-8"))
    dev = [data_root / row["file"] for row in spec["dev"]]
    if not dev:
        raise ValueError("manifest dev split is empty")

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = feature_config_from_checkpoint(checkpoint_payload)
    assert_feature_config_matches_checkpoint(config, checkpoint_payload)
    probe_payload = torch.load(validation_probe, map_location="cpu", weights_only=False)
    head_state = probe_payload.get("base_head_state_dict")
    if not isinstance(head_state, dict) or not head_state:
        raise ValueError("validation probe lacks base_head_state_dict")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    builder = P0FeatureBuilder(config).to(device)
    base = base_api.load_model(kit_root, checkpoint, builder, device).eval()
    head = AdaptiveUncertaintyHead(hidden=32, blocks=2).to(device)
    head.load_state_dict(head_state)
    head.eval()

    dataset = H5WindowDataset(dev, include_pressure=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    predictions: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.inference_mode():
        for x, y, _, _ in loader:
            x = x.to(device)
            prediction = base_api.forward(base, builder, x)
            uncertainty_input = torch.cat([x, flow_features(prediction[..., :2])], dim=-1)
            sigma = head(uncertainty_input.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
            predictions.append(prediction.cpu().numpy().astype(np.float32))
            sigmas.append(sigma.cpu().numpy().astype(np.float32))
            targets.append(y.numpy().astype(np.float32))

    pred = np.concatenate(predictions, axis=0)
    sigma = np.concatenate(sigmas, axis=0)
    target = np.concatenate(targets, axis=0)
    if pred.shape != target.shape or sigma.shape != pred.shape[:-1] + (2,):
        raise ValueError(f"unexpected audit shapes pred={pred.shape} target={target.shape} sigma={sigma.shape}")

    sys.path.insert(0, str(kit_root.resolve()))
    import scoring

    c = scoring.measured_channels(target)
    if c != 2:
        raise ValueError(f"expected exactly two measured PIV channels, got {c}")
    penalties = _normalized_penalties(pred, target, c, scoring)

    horizon_rows: list[dict[str, Any]] = []
    for channel_index, channel_name in enumerate(CHANNEL_NAMES):
        for horizon_index in range(pred.shape[1]):
            error_slice = np.abs(
                pred[:, horizon_index, :, :, channel_index] - target[:, horizon_index, :, :, channel_index]
            )
            sigma_slice = sigma[:, horizon_index, :, :, channel_index]
            half_slice = CURRENT_FLOOR + CURRENT_MULT * sigma_slice
            inside_slice = error_slice <= half_slice
            horizon_rows.append(
                {
                    "channel": channel_name,
                    "horizon": horizon_index + 1,
                    "mean_abs_error": float(np.mean(error_slice)),
                    "mean_sigma": float(np.mean(sigma_slice)),
                    "coverage": float(np.mean(inside_slice)),
                    "sps": _slice_sps_score(error_slice, half_slice, penalties, scoring),
                    "pearson_sigma_abs_error": safe_pearson(sigma_slice, error_slice),
                    "spearman_sigma_abs_error": safe_spearman(sigma_slice, error_slice),
                    "elements": int(error_slice.size),
                }
            )

    abs_error_uv = np.abs(pred[..., :2] - target[..., :2])
    inside_uv = abs_error_uv <= (CURRENT_FLOOR + CURRENT_MULT * sigma)
    decile_rows = sigma_decile_rows(sigma, abs_error_uv, inside_uv, bins=10)

    finite_spearman = np.array(
        [row["spearman_sigma_abs_error"] for row in horizon_rows], dtype=np.float64
    )
    finite_spearman = finite_spearman[np.isfinite(finite_spearman)]
    worst_coverage = sorted(horizon_rows, key=lambda row: row["coverage"])[:5]
    decile_errors = np.array([row["mean_abs_error"] for row in decile_rows], dtype=np.float64)
    result: dict[str, Any] = {
        "status": "REVIEW_REQUIRED",
        "scope": "frozen_dev_only_no_training_no_final_no_package_no_submission",
        "bounds": {"floor": CURRENT_FLOOR, "mult": CURRENT_MULT},
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_iteration": checkpoint_payload.get("iteration"),
        "validation_probe": str(validation_probe.resolve()),
        "validation_probe_sha256": _sha256(validation_probe),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": _sha256(manifest),
        "scorer_sha256": _sha256(kit_root / "scoring.py"),
        "trajectories": len(dev),
        "windows": len(dataset),
        "horizon_channel_rows": horizon_rows,
        "sigma_deciles": decile_rows,
        "summary": {
            "median_spearman_sigma_abs_error": float(np.median(finite_spearman)) if finite_spearman.size else float("nan"),
            "decile_mean_abs_error_monotonic_non_decreasing": bool(np.all(np.diff(decile_errors) >= 0.0)),
            "worst_5_horizon_channels_by_coverage": [
                {
                    "channel": row["channel"],
                    "horizon": row["horizon"],
                    "coverage": row["coverage"],
                    "sps": row["sps"],
                    "spearman_sigma_abs_error": row["spearman_sigma_abs_error"],
                }
                for row in worst_coverage
            ],
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "horizon_channel_stats.csv", horizon_rows)
    _write_csv(out_dir / "sigma_deciles.csv", decile_rows)
    (out_dir / "audit_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "windows": len(dataset), "summary": result["summary"]}, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("data-root", "kit-root", "checkpoint", "manifest", "validation-probe", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run(
        data_root=args.data_root,
        kit_root=args.kit_root,
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        validation_probe=args.validation_probe,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
