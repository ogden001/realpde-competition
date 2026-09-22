#!/usr/bin/env python3
"""Standard post-train diagnostics for RealPDE Track 1 predictors.

Consumes prediction arrays shaped [N,T,H,W,C>=2] and writes a stable bundle
used after every training experiment: horizon, trajectory, trajectory-by-horizon,
spatial maps, temporal mean/fluctuation decomposition, and residual before/after
analysis when base predictions are available.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

EPS = 1e-12


def _validate(prediction, target, trajectories, starts, base_prediction):
    if prediction.shape != target.shape or prediction.ndim != 5 or prediction.shape[-1] < 2:
        raise ValueError(f"expected matching [N,T,H,W,C>=2], got {prediction.shape}/{target.shape}")
    if base_prediction is not None and base_prediction.shape != prediction.shape:
        raise ValueError("base prediction shape must match prediction")
    if len(trajectories) != prediction.shape[0] or len(starts) != prediction.shape[0]:
        raise ValueError("window metadata length does not match predictions")


def _velocity(x: np.ndarray) -> np.ndarray:
    return np.asarray(x[..., :2], dtype=np.float64)


def _rel_l2_per_sample(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    n = pred.shape[0]
    num = np.linalg.norm((pred - target).reshape(n, -1), axis=1)
    den = np.linalg.norm(target.reshape(n, -1), axis=1)
    return num / np.maximum(den, EPS)


def _tke_field(x: np.ndarray) -> np.ndarray:
    vel = _velocity(x)
    mean = vel.mean(axis=1, keepdims=True)
    fluct = vel - mean
    return 0.5 * np.mean(np.sum(fluct * fluct, axis=-1), axis=1)


def _mean_fluctuation_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = _velocity(prediction)
    truth = _velocity(target)
    pred_mean = pred.mean(axis=1)
    truth_mean = truth.mean(axis=1)
    pred_fluct = pred - pred_mean[:, None]
    truth_fluct = truth - truth_mean[:, None]
    pred_tke = _tke_field(prediction)
    truth_tke = _tke_field(target)
    return {
        "mean_field_rel_l2": float(_rel_l2_per_sample(pred_mean, truth_mean).mean()),
        "fluctuation_rel_l2": float(_rel_l2_per_sample(pred_fluct, truth_fluct).mean()),
        "tke_field_rel_l2": float(_rel_l2_per_sample(pred_tke, truth_tke).mean()),
        "pred_tke_mean": float(pred_tke.mean()),
        "target_tke_mean": float(truth_tke.mean()),
        "tke_energy_ratio": float(pred_tke.sum() / max(float(truth_tke.sum()), EPS)),
    }


def horizon_rows(prediction, target, *, experiment: str, base_prediction=None):
    pred = _velocity(prediction)
    truth = _velocity(target)
    base = _velocity(base_prediction) if base_prediction is not None else None
    rows = []
    for h in range(pred.shape[1]):
        p, y = pred[:, h], truth[:, h]
        row = {
            "experiment": experiment,
            "horizon": h + 1,
            "frame_rel_l2": float(_rel_l2_per_sample(p, y).mean()),
            "u_rmse": float(np.sqrt(np.mean((p[..., 0] - y[..., 0]) ** 2))),
            "v_rmse": float(np.sqrt(np.mean((p[..., 1] - y[..., 1]) ** 2))),
            "velocity_rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        }
        if base is not None:
            b = base[:, h]
            base_err = np.sum((b - y) ** 2, axis=-1)
            pred_err = np.sum((p - y) ** 2, axis=-1)
            row.update({
                "base_frame_rel_l2": float(_rel_l2_per_sample(b, y).mean()),
                "delta_rms": float(np.sqrt(np.mean((p - b) ** 2))),
                "correction_help_fraction": float(np.mean(pred_err < base_err)),
            })
        rows.append(row)
    return rows


def trajectory_rows(prediction, target, trajectories, *, experiment: str):
    pred = _velocity(prediction)
    truth = _velocity(target)
    tke_pred = _tke_field(prediction)
    tke_truth = _tke_field(target)
    names = np.asarray([str(name) for name in trajectories], dtype=object)
    rows = []
    for name in sorted(set(names.tolist())):
        idx = np.flatnonzero(names == name)
        rows.append({
            "experiment": experiment,
            "trajectory": name,
            "windows": int(idx.size),
            "rel_l2": float(_rel_l2_per_sample(pred[idx], truth[idx]).mean()),
            "tke_rel_l2": float(_rel_l2_per_sample(tke_pred[idx], tke_truth[idx]).mean()),
            **_mean_fluctuation_metrics(prediction[idx], target[idx]),
        })
    return rows


def trajectory_horizon_rows(prediction, target, trajectories, starts, *, experiment: str):
    pred = _velocity(prediction)
    truth = _velocity(target)
    rows = []
    for i, (trajectory, start) in enumerate(zip(trajectories, starts, strict=True)):
        for h in range(pred.shape[1]):
            p = pred[i:i + 1, h]
            y = truth[i:i + 1, h]
            rows.append({
                "experiment": experiment,
                "trajectory": str(trajectory),
                "window_start": int(start),
                "horizon": h + 1,
                "frame_rel_l2": float(_rel_l2_per_sample(p, y)[0]),
                "velocity_rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            })
    return rows


def spatial_maps(prediction: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    pred = _velocity(prediction)
    truth = _velocity(target)
    err = pred - truth
    pred_mean = pred.mean(axis=1)
    truth_mean = truth.mean(axis=1)
    pred_fluct = pred - pred_mean[:, None]
    truth_fluct = truth - truth_mean[:, None]
    pred_tke = _tke_field(prediction)
    truth_tke = _tke_field(target)
    return {
        "u_rmse": np.sqrt(np.mean(err[..., 0] ** 2, axis=(0, 1))).astype(np.float32),
        "v_rmse": np.sqrt(np.mean(err[..., 1] ** 2, axis=(0, 1))).astype(np.float32),
        "velocity_rmse": np.sqrt(np.mean(err ** 2, axis=(0, 1, 4))).astype(np.float32),
        "mean_field_rmse": np.sqrt(np.mean((pred_mean - truth_mean) ** 2, axis=(0, 3))).astype(np.float32),
        "fluctuation_rmse": np.sqrt(np.mean((pred_fluct - truth_fluct) ** 2, axis=(0, 1, 4))).astype(np.float32),
        "tke_abs_error": np.mean(np.abs(pred_tke - truth_tke), axis=0).astype(np.float32),
        "target_tke_mean": np.mean(truth_tke, axis=0).astype(np.float32),
    }


def residual_summary(base_prediction, prediction, target) -> dict[str, float]:
    base = _velocity(base_prediction)
    pred = _velocity(prediction)
    truth = _velocity(target)
    before = np.sum((base - truth) ** 2, axis=-1)
    after = np.sum((pred - truth) ** 2, axis=-1)
    delta = pred - base
    delta_mean = delta.mean(axis=1, keepdims=True)
    delta_fluct = delta - delta_mean
    base_diag = _mean_fluctuation_metrics(base_prediction, target)
    pred_diag = _mean_fluctuation_metrics(prediction, target)
    return {
        "base_rel_l2": float(_rel_l2_per_sample(base, truth).mean()),
        "corrected_rel_l2": float(_rel_l2_per_sample(pred, truth).mean()),
        "correction_help_fraction": float(np.mean(after < before)),
        "correction_hurt_fraction": float(np.mean(after > before)),
        "delta_rms": float(np.sqrt(np.mean(delta ** 2))),
        "delta_mean_rms": float(np.sqrt(np.mean(delta_mean ** 2))),
        "delta_fluctuation_rms": float(np.sqrt(np.mean(delta_fluct ** 2))),
        "base_mean_field_rel_l2": base_diag["mean_field_rel_l2"],
        "corrected_mean_field_rel_l2": pred_diag["mean_field_rel_l2"],
        "base_fluctuation_rel_l2": base_diag["fluctuation_rel_l2"],
        "corrected_fluctuation_rel_l2": pred_diag["fluctuation_rel_l2"],
        "base_tke_field_rel_l2": base_diag["tke_field_rel_l2"],
        "corrected_tke_field_rel_l2": pred_diag["tke_field_rel_l2"],
        "base_tke_energy_ratio": base_diag["tke_energy_ratio"],
        "corrected_tke_energy_ratio": pred_diag["tke_energy_ratio"],
    }


def _write_csv(path: Path, rows):
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_post_train_diagnostics(*, out_dir: Path, experiment: str, prediction: np.ndarray,
                                 target: np.ndarray, trajectories: Sequence[str], starts: Sequence[int],
                                 base_prediction: np.ndarray | None = None) -> dict[str, object]:
    _validate(prediction, target, trajectories, starts, base_prediction)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "by_horizon.csv", horizon_rows(
        prediction, target, experiment=experiment, base_prediction=base_prediction))
    _write_csv(out_dir / "by_trajectory.csv", trajectory_rows(
        prediction, target, trajectories, experiment=experiment))
    _write_csv(out_dir / "by_trajectory_horizon.csv", trajectory_horizon_rows(
        prediction, target, trajectories, starts, experiment=experiment))
    maps = spatial_maps(prediction, target)
    np.savez_compressed(out_dir / "spatial_maps.npz", **maps)
    summary = {
        "experiment": experiment,
        "windows": int(prediction.shape[0]),
        "horizons": int(prediction.shape[1]),
        "trajectories": len(set(map(str, trajectories))),
        "mean_fluctuation": _mean_fluctuation_metrics(prediction, target),
        "spatial_map_keys": sorted(maps),
    }
    if base_prediction is not None:
        summary["residual_before_after"] = residual_summary(base_prediction, prediction, target)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
