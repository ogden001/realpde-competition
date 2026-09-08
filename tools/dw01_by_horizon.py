"""Prediction-only Future20 frame and horizon diagnostics.

All TKE and MVPE quantities emitted here are diagnostics, not official
per-frame scorer values. Inputs are ``[N,20,H,W,3]`` and only u/v channels
participate in the requested metrics.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _validate(prediction: np.ndarray, target: np.ndarray, trajectories: Sequence[str], starts: Sequence[int]) -> None:
    if prediction.shape != target.shape or prediction.ndim != 5 or prediction.shape[-1] < 2:
        raise ValueError(f"expected matching [N,T,H,W,C>=2], got {prediction.shape}/{target.shape}")
    if len(trajectories) != prediction.shape[0] or len(starts) != prediction.shape[0]:
        raise ValueError("window metadata length does not match predictions")


def official_probe_geometry(height: int, width: int) -> tuple[list[int], list[int]]:
    """Return the v9 scorer's 32x64 probe rows and columns."""
    center_y, interval_y, distance, center_x = 16, min(2, max(1, height // 10)), 16, 10
    ys = [z for z in range(center_y - interval_y * 4, center_y + interval_y * 5, interval_y) if 0 <= z < height]
    if 21 < width:
        xs = [int(((i + 1) * distance + center_x) / 2) for i in range(4)]
    else:
        xs = [int(0.5 * (i + 2) * distance + center_x) for i in range(4)]
    return ys, [x for x in xs if 0 <= x < width]


def _relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    numerator = np.linalg.norm((prediction - target).reshape(-1))
    denominator = np.linalg.norm(target.reshape(-1))
    return float(numerator / max(float(denominator), 1e-12))


def _energy_ratio(prediction: np.ndarray, target: np.ndarray) -> float:
    numerator, denominator = float(prediction.sum()), float(target.sum())
    if denominator <= 1e-12:
        return 1.0 if numerator <= 1e-12 else float("inf")
    return numerator / denominator


def compute_window_horizon_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    trajectories: Sequence[str],
    starts: Sequence[int],
) -> list[dict[str, object]]:
    """Compute one diagnostic row for every window and Future20 horizon."""
    _validate(prediction, target, trajectories, starts)
    pred, truth = prediction[..., :2].astype(np.float64), target[..., :2].astype(np.float64)
    n, horizon, height, width, _ = pred.shape
    ys, xs = official_probe_geometry(height, width)
    pred_mean, truth_mean = pred.mean(axis=1), truth.mean(axis=1)
    pred_energy = 0.5 * np.sum((pred - pred_mean[:, None]) ** 2, axis=-1)
    truth_energy = 0.5 * np.sum((truth - truth_mean[:, None]) ** 2, axis=-1)
    rows: list[dict[str, object]] = []
    for i in range(n):
        for h in range(horizon):
            p, y = pred[i, h], truth[i, h]
            energy_p, energy_y = pred_energy[i, h], truth_energy[i, h]
            probe_errors = [_relative_l2(p[ys, x], y[ys, x]) for x in xs]
            rows.append({
                "trajectory": str(trajectories[i]), "window_start": int(starts[i]), "horizon": h + 1,
                "frame_rel_l2": _relative_l2(p, y),
                "frame_rmse": float(np.sqrt(np.mean((p - y) ** 2))),
                "tke_contrib_rel_l2": _relative_l2(energy_p, energy_y),
                "tke_contrib_ratio": _energy_ratio(energy_p, energy_y),
                "mvpe_probe_rel_l2": float(np.mean(probe_errors)) if probe_errors else float("nan"),
            })
    return rows


def aggregate_by_horizon(rows: Sequence[dict[str, object]], *, experiment: str, trajectories: int) -> list[dict[str, object]]:
    """Average window-level diagnostic rows into exactly one row per horizon."""
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["horizon"])].append(row)
    if not grouped:
        raise ValueError("cannot aggregate an empty row set")
    horizons = sorted(grouped)
    if horizons != list(range(1, max(horizons) + 1)):
        raise ValueError("horizon rows must be contiguous and start at 1")
    fields = ("frame_rel_l2", "frame_rmse", "tke_contrib_rel_l2", "tke_contrib_ratio", "mvpe_probe_rel_l2")
    result = []
    for horizon in horizons:
        group = grouped[horizon]
        result.append({"experiment": experiment, "horizon": horizon,
                       **{field: float(np.mean([float(row[field]) for row in group])) for field in fields},
                       "windows": len(group), "trajectories": int(trajectories)})
    return result


def write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
