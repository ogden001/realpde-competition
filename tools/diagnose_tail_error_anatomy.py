#!/usr/bin/env python3
"""Zero-training F18/F19/F20 error anatomy for Exp1 Strong Backbone.

The script runs several *diagnostic oracles* on Clean Seen-Dev12 only:
1. non-wrapping integer spatial shifts (dx,dy in [-2,2]);
2. temporal lag checks and one-step linear extrapolation;
3. fluctuation-amplitude scaling around the model's Future20 temporal mean;
4. residual-direction alpha diagnostics already used in the first tail study.

These are upper-bound / mechanism diagnostics, not submission parameters.
No optimizer step is performed and no AoA10/locked-final/Codabench data is read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from colleague_80pt.realpde_h5_feature_adapter_train import (
    H5WindowDataset,
    measured_channels,
    mvpe_rel_l2_per_sample,
    paths_from_split_manifest,
    rel_l2_per_sample,
    tke_rel_l2_per_sample,
)
from colleague_80pt.residual_multi import (
    CorrectorConfig,
    ResidualCorrectionModel,
    ResidualCorrector3D,
    load_frozen_base,
    load_residual_checkpoint,
)
from diagnose_strong_backbone_tail import (
    EXP1_BACKBONE_SHA256,
    EXP1_RESIDUAL_SHA256,
    EXPECTED_DEV,
    EXPECTED_DEV_WINDOWS,
    EXPECTED_TRAIN,
    residual_geometry_by_horizon,
)

SHIFT_VALUES = (-2, -1, 0, 1, 2)
GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0)
TAIL_HORIZONS = (18, 19, 20)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty rows")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def frame_rel(pred: np.ndarray, target: np.ndarray) -> float:
    p = np.asarray(pred[..., :2], dtype=np.float64).reshape(-1)
    y = np.asarray(target[..., :2], dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(p - y) / max(np.linalg.norm(y), 1e-30))


def frame_sse(pred: np.ndarray, target: np.ndarray) -> float:
    d = np.asarray(pred[..., :2], dtype=np.float64) - np.asarray(target[..., :2], dtype=np.float64)
    return float(np.sum(d * d))


def nonwrapping_shift_uv(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift HxWxC or NxHxWxC without circular wrap; replicate border values."""
    x = np.asarray(frame)
    if x.ndim not in (3, 4):
        raise ValueError(f"expected HWC or NHWC, got {x.shape}")
    h_axis = -3
    w_axis = -2
    h, w = x.shape[h_axis], x.shape[w_axis]
    if abs(dx) >= w or abs(dy) >= h:
        raise ValueError("shift exceeds frame size")

    pad = [(0, 0)] * x.ndim
    pad[h_axis] = (max(dy, 0), max(-dy, 0))
    pad[w_axis] = (max(dx, 0), max(-dx, 0))
    padded = np.pad(x, pad, mode="edge")

    y0 = max(-dy, 0)
    x0 = max(-dx, 0)
    sl = [slice(None)] * x.ndim
    sl[h_axis] = slice(y0, y0 + h)
    sl[w_axis] = slice(x0, x0 + w)
    return padded[tuple(sl)]


def apply_shift_to_horizon(pred: np.ndarray, horizon: int, dx: int, dy: int) -> np.ndarray:
    out = pred.copy()
    out[:, horizon - 1] = nonwrapping_shift_uv(out[:, horizon - 1], dx, dy)
    return out


def global_shift_oracle(pred: np.ndarray, target: np.ndarray, horizon: int) -> dict[str, object]:
    p = pred[:, horizon - 1]
    y = target[:, horizon - 1]
    base_sse = frame_sse(p, y)
    rows = []
    for dy in SHIFT_VALUES:
        for dx in SHIFT_VALUES:
            shifted = nonwrapping_shift_uv(p, dx, dy)
            sse = frame_sse(shifted, y)
            rows.append({"dx": dx, "dy": dy, "sse": sse, "rel": frame_rel(shifted, y)})
    best = min(rows, key=lambda row: float(row["sse"]))
    return {
        "horizon": horizon,
        "base_rel": frame_rel(p, y),
        "best_dx": int(best["dx"]),
        "best_dy": int(best["dy"]),
        "best_rel": float(best["rel"]),
        "sse_gain_pct": 100.0 * (base_sse - float(best["sse"])) / max(base_sse, 1e-30),
        "grid": rows,
    }


def per_window_shift_oracle(pred: np.ndarray, target: np.ndarray, horizon: int) -> dict[str, object]:
    p = pred[:, horizon - 1]
    y = target[:, horizon - 1]
    choices: list[tuple[int, int]] = []
    base_sse = 0.0
    best_sse = 0.0
    for i in range(p.shape[0]):
        b = frame_sse(p[i], y[i])
        base_sse += b
        candidates = []
        for dy in SHIFT_VALUES:
            for dx in SHIFT_VALUES:
                s = frame_sse(nonwrapping_shift_uv(p[i], dx, dy), y[i])
                candidates.append((s, dx, dy))
        s, dx, dy = min(candidates)
        best_sse += s
        choices.append((dx, dy))
    counts = Counter(choices)
    mode, mode_n = counts.most_common(1)[0]
    return {
        "horizon": horizon,
        "oracle_sse_gain_pct": 100.0 * (base_sse - best_sse) / max(base_sse, 1e-30),
        "mode_dx": int(mode[0]),
        "mode_dy": int(mode[1]),
        "mode_fraction": float(mode_n / len(choices)),
        "nonzero_shift_fraction": float(np.mean([dx != 0 or dy != 0 for dx, dy in choices])),
        "choice_counts": {f"{dx},{dy}": int(n) for (dx, dy), n in sorted(counts.items())},
    }


def temporal_lag_probe(pred: np.ndarray, target: np.ndarray, horizon: int) -> dict[str, object]:
    """Compare Pred F_h with GT F_{h-2}, F_{h-1}, F_h when available."""
    p = pred[:, horizon - 1]
    offsets = []
    for target_h in range(max(1, horizon - 2), horizon + 1):
        y = target[:, target_h - 1]
        offsets.append({
            "pred_horizon": horizon,
            "target_horizon": target_h,
            "offset": target_h - horizon,
            "rel": frame_rel(p, y),
            "sse": frame_sse(p, y),
        })
    best = min(offsets, key=lambda row: float(row["sse"]))
    current = next(row for row in offsets if int(row["offset"]) == 0)
    return {
        "horizon": horizon,
        "best_target_horizon": int(best["target_horizon"]),
        "best_offset": int(best["offset"]),
        "best_rel": float(best["rel"]),
        "current_rel": float(current["rel"]),
        "best_vs_current_sse_gain_pct": 100.0 * (float(current["sse"]) - float(best["sse"])) / max(float(current["sse"]), 1e-30),
        "comparisons": offsets,
    }


def temporal_extrapolation_oracle(pred: np.ndarray, target: np.ndarray, horizon: int) -> dict[str, object]:
    if horizon < 2:
        raise ValueError("temporal extrapolation requires horizon >=2")
    p_now = np.asarray(pred[:, horizon - 1], dtype=np.float64)
    p_prev = np.asarray(pred[:, horizon - 2], dtype=np.float64)
    y = np.asarray(target[:, horizon - 1], dtype=np.float64)
    base_sse = frame_sse(p_now, y)
    rows = []
    for gamma in GAMMAS:
        candidate = p_now + gamma * (p_now - p_prev)
        sse = frame_sse(candidate, y)
        rows.append({"gamma": gamma, "rel": frame_rel(candidate, y), "sse": sse})
    best = min(rows, key=lambda row: float(row["sse"]))
    return {
        "horizon": horizon,
        "best_gamma": float(best["gamma"]),
        "best_rel": float(best["rel"]),
        "base_rel": frame_rel(p_now, y),
        "sse_gain_pct": 100.0 * (base_sse - float(best["sse"])) / max(base_sse, 1e-30),
        "grid": rows,
    }


def amplitude_oracle(pred: np.ndarray, target: np.ndarray, horizon: int) -> dict[str, object]:
    """Scale one horizon's deviation from the prediction's Future20 temporal mean."""
    p = np.asarray(pred[..., :2], dtype=np.float64)
    y = np.asarray(target[..., :2], dtype=np.float64)
    mean = p.mean(axis=1, keepdims=True)
    ph = p[:, horizon - 1]
    mh = mean[:, 0]
    yh = y[:, horizon - 1]
    fluct = ph - mh
    desired = yh - mh

    f = fluct.reshape(-1)
    d = desired.reshape(-1)
    scale = float(np.dot(f, d) / max(np.dot(f, f), 1e-30))
    candidate = mh + scale * fluct
    base_sse = frame_sse(ph, yh)
    cand_sse = frame_sse(candidate, yh)
    return {
        "horizon": horizon,
        "scale_star": scale,
        "base_rel": frame_rel(ph, yh),
        "best_rel": frame_rel(candidate, yh),
        "sse_gain_pct": 100.0 * (base_sse - cand_sse) / max(base_sse, 1e-30),
    }


def whole_sequence_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    channels = measured_channels(target)
    rel = rel_l2_per_sample(pred, target, channels)
    tke = tke_rel_l2_per_sample(pred, target, channels)
    mvpe = mvpe_rel_l2_per_sample(pred, target)
    return {
        "rel_l2_raw": float(np.mean(rel)),
        "tke_raw": float(np.mean(tke)),
        "mvpe_raw": float(np.mean(mvpe)),
    }


def candidate_global_shift(pred: np.ndarray, target: np.ndarray, shift_rows: list[dict[str, object]]) -> np.ndarray:
    out = pred.copy()
    for row in shift_rows:
        h = int(row["horizon"])
        out[:, h - 1] = nonwrapping_shift_uv(out[:, h - 1], int(row["best_dx"]), int(row["best_dy"]))
    return out


def candidate_temporal_extrap(pred: np.ndarray, rows: list[dict[str, object]]) -> np.ndarray:
    out = pred.copy()
    source = pred.copy()
    for row in rows:
        h = int(row["horizon"])
        gamma = float(row["best_gamma"])
        out[:, h - 1] = source[:, h - 1] + gamma * (source[:, h - 1] - source[:, h - 2])
    return out


def candidate_amplitude(pred: np.ndarray, rows: list[dict[str, object]]) -> np.ndarray:
    out = pred.copy()
    source = pred.copy()
    mean = np.asarray(source[..., :2], dtype=np.float64).mean(axis=1, keepdims=True)
    for row in rows:
        h = int(row["horizon"])
        scale = float(row["scale_star"])
        out[:, h - 1, ..., :2] = (
            mean[:, 0] + scale * (source[:, h - 1, ..., :2] - mean[:, 0])
        ).astype(out.dtype)
    return out


def metric_delta(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {
        key: 100.0 * (candidate[key] - baseline[key]) / max(abs(baseline[key]), 1e-30)
        for key in baseline
    }


def candidate_residual_tail_alpha(
    base: np.ndarray,
    final: np.ndarray,
    geometry: list[dict[str, object]],
) -> np.ndarray:
    """Apply the diagnostic least-squares alpha* only to F18/F19/F20."""
    out = final.copy()
    for row in geometry:
        h = int(row["horizon"])
        if h not in TAIL_HORIZONS:
            continue
        alpha = float(row["alpha_star"])
        delta = final[:, h - 1] - base[:, h - 1]
        out[:, h - 1] = base[:, h - 1] + alpha * delta
        if out.shape[-1] >= 3:
            out[:, h - 1, ..., 2] = 0.0
    return out


def diagnose_prediction(label: str, pred: np.ndarray, target: np.ndarray) -> dict[str, object]:
    shifts = [global_shift_oracle(pred, target, h) for h in TAIL_HORIZONS]
    shifts_pw = [per_window_shift_oracle(pred, target, h) for h in TAIL_HORIZONS]
    lags = [temporal_lag_probe(pred, target, h) for h in TAIL_HORIZONS]
    extrap = [temporal_extrapolation_oracle(pred, target, h) for h in TAIL_HORIZONS]
    amplitude = [amplitude_oracle(pred, target, h) for h in TAIL_HORIZONS]

    baseline = whole_sequence_metrics(pred, target)
    candidates = {
        "spatial_global_shift_f18_f20": whole_sequence_metrics(candidate_global_shift(pred, target, shifts), target),
        "temporal_extrapolation_f18_f20": whole_sequence_metrics(candidate_temporal_extrap(pred, extrap), target),
        "amplitude_scaling_f18_f20": whole_sequence_metrics(candidate_amplitude(pred, amplitude), target),
    }
    return {
        "label": label,
        "baseline_metrics": baseline,
        "spatial_shift_global": shifts,
        "spatial_shift_per_window_oracle": shifts_pw,
        "temporal_lag": lags,
        "temporal_extrapolation": extrap,
        "amplitude_scaling": amplitude,
        "candidate_metrics": {
            name: {"metrics": metrics, "delta_pct_vs_baseline": metric_delta(metrics, baseline)}
            for name, metrics in candidates.items()
        },
    }


def mechanism_summary(base_diag: dict[str, object], final_diag: dict[str, object], residual_geom: list[dict[str, object]]) -> dict[str, object]:
    def hrow(rows: list[dict[str, object]], h: int) -> dict[str, object]:
        return next(row for row in rows if int(row["horizon"]) == h)

    signals = {}
    for h in TAIL_HORIZONS:
        b_shift = hrow(base_diag["spatial_shift_global"], h)
        b_shift_pw = hrow(base_diag["spatial_shift_per_window_oracle"], h)
        b_lag = hrow(base_diag["temporal_lag"], h)
        b_ext = hrow(base_diag["temporal_extrapolation"], h)
        b_amp = hrow(base_diag["amplitude_scaling"], h)
        rg = hrow(residual_geom, h)
        signals[str(h)] = {
            "backbone_spatial_global_gain_pct": float(b_shift["sse_gain_pct"]),
            "backbone_spatial_oracle_gain_pct": float(b_shift_pw["oracle_sse_gain_pct"]),
            "backbone_spatial_mode_fraction": float(b_shift_pw["mode_fraction"]),
            "backbone_temporal_lag_best_offset": int(b_lag["best_offset"]),
            "backbone_temporal_lag_gain_pct": float(b_lag["best_vs_current_sse_gain_pct"]),
            "backbone_temporal_extrap_gain_pct": float(b_ext["sse_gain_pct"]),
            "backbone_temporal_best_gamma": float(b_ext["best_gamma"]),
            "backbone_amplitude_gain_pct": float(b_amp["sse_gain_pct"]),
            "backbone_amplitude_scale_star": float(b_amp["scale_star"]),
            "residual_delta_target_cosine": float(rg["delta_target_cosine"]),
            "residual_alpha_star": float(rg["alpha_star"]),
            "residual_sse_gain_pct": float(rg["sse_gain_pct"]),
        }

    h20 = signals["20"]
    ranked = sorted([
        ("spatial_phase_global", h20["backbone_spatial_global_gain_pct"]),
        ("temporal_extrapolation", h20["backbone_temporal_extrap_gain_pct"]),
        ("amplitude_scaling", h20["backbone_amplitude_gain_pct"]),
    ], key=lambda x: float(x[1]), reverse=True)

    return {
        "tail_signals": signals,
        "h20_ranked_zero_training_mechanisms": [
            {"mechanism": name, "sse_gain_pct": float(gain)} for name, gain in ranked
        ],
        "interpretation_note": (
            "Use these as mechanism diagnostics only. A large global spatial-shift gain suggests coherent phase drift; "
            "a much larger per-window than global shift gain suggests heterogeneous phase error; "
            "negative lag offsets suggest temporal delay; positive extrapolation gain suggests under-advanced dynamics; "
            "amplitude gain isolates fluctuation-scale mismatch. Do not auto-promote any oracle setting to submission."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--real-root", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--backbone-checkpoint", type=Path, required=True)
    p.add_argument("--residual-checkpoint", type=Path, required=True)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--require-cuda", action="store_true")
    args = p.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)

    for path in (args.real_root, args.split_manifest, args.backbone_checkpoint, args.residual_checkpoint, args.model_root):
        if not path.exists():
            raise FileNotFoundError(path)

    train_paths, dev_paths = paths_from_split_manifest(args.real_root, args.split_manifest)
    if (len(train_paths), len(dev_paths)) != (EXPECTED_TRAIN, EXPECTED_DEV):
        raise ValueError(
            f"Seen-Dev12 only: expected {EXPECTED_TRAIN}/{EXPECTED_DEV}, got {len(train_paths)}/{len(dev_paths)}"
        )
    if set(p.name for p in train_paths) & set(p.name for p in dev_paths):
        raise ValueError("train/dev overlap")

    backbone_sha = sha256(args.backbone_checkpoint)
    residual_sha = sha256(args.residual_checkpoint)
    if backbone_sha != EXP1_BACKBONE_SHA256:
        raise RuntimeError(f"wrong backbone SHA: {backbone_sha}")
    if residual_sha != EXP1_RESIDUAL_SHA256:
        raise RuntimeError(f"wrong residual SHA: {residual_sha}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")

    payload = torch.load(args.residual_checkpoint, map_location="cpu", weights_only=False)
    raw_cfg = payload.get("corrector_config")
    if not isinstance(raw_cfg, dict):
        raise ValueError("residual checkpoint missing corrector_config")

    base_model = load_frozen_base("sota_v2_mf", args.backbone_checkpoint, args.model_root, device)
    model = ResidualCorrectionModel(base_model, ResidualCorrector3D(CorrectorConfig(**raw_cfg))).to(device)
    load_residual_checkpoint(model, args.residual_checkpoint, device)
    model.eval()

    ds = H5WindowDataset(
        dev_paths,
        in_steps=20,
        out_steps=20,
        stride=20,
        sub_sample=2,
        include_pressure=False,
        window_mode="fixed",
        preload_to_ram=False,
    )
    if len(ds) != EXPECTED_DEV_WINDOWS:
        raise ValueError(f"expected {EXPECTED_DEV_WINDOWS} dev windows, got {len(ds)}")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    base_parts, final_parts, target_parts = [], [], []
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            base = model.base_predict(x)
            delta = model.predict_delta(x, base)
            final = model.combine(base, delta, 1.0)
            base_parts.append(base.cpu().numpy().astype(np.float32))
            final_parts.append(final.cpu().numpy().astype(np.float32))
            target_parts.append(y.numpy().astype(np.float32))

    base = np.concatenate(base_parts, axis=0)
    final = np.concatenate(final_parts, axis=0)
    target = np.concatenate(target_parts, axis=0)
    if not (np.isfinite(base).all() and np.isfinite(final).all() and np.isfinite(target).all()):
        raise FloatingPointError("non-finite arrays")

    base_diag = diagnose_prediction("strong_backbone", base, target)
    final_diag = diagnose_prediction("strong_backbone_plus_residual", final, target)
    residual_geom = residual_geometry_by_horizon(base, final, target)
    summary = mechanism_summary(base_diag, final_diag, residual_geom)
    final_metrics = whole_sequence_metrics(final, target)
    alpha_candidate = candidate_residual_tail_alpha(base, final, residual_geom)
    alpha_metrics = whole_sequence_metrics(alpha_candidate, target)
    summary["residual_tail_alpha_oracle_candidate"] = {
        "metrics": alpha_metrics,
        "delta_pct_vs_final": metric_delta(alpha_metrics, final_metrics),
        "alphas": {
            str(int(row["horizon"])): float(row["alpha_star"])
            for row in residual_geom if int(row["horizon"]) in TAIL_HORIZONS
        },
        "note": "Seen-Dev diagnostic oracle only; not a submission calibration.",
    }

    dump(args.out_dir / "backbone_error_anatomy.json", base_diag)
    dump(args.out_dir / "final_error_anatomy.json", final_diag)
    dump(args.out_dir / "mechanism_summary.json", summary)
    write_rows(
        args.out_dir / "residual_geometry_f18_f20.csv",
        [row for row in residual_geom if int(row["horizon"]) in TAIL_HORIZONS],
    )
    write_rows(
        args.out_dir / "backbone_spatial_shift.csv",
        [{k: v for k, v in row.items() if k != "grid"} for row in base_diag["spatial_shift_global"]],
    )
    write_rows(
        args.out_dir / "backbone_temporal_extrapolation.csv",
        [{k: v for k, v in row.items() if k != "grid"} for row in base_diag["temporal_extrapolation"]],
    )
    write_rows(args.out_dir / "backbone_amplitude_scaling.csv", base_diag["amplitude_scaling"])

    dump(args.out_dir / "run_manifest.json", {
        "status": "REVIEW_REQUIRED",
        "purpose": "Exp1 F18-F20 zero-training error anatomy",
        "split": "Clean Seen-Dev12 only",
        "train_trajectories_in_manifest": len(train_paths),
        "eval_trajectories": len(dev_paths),
        "eval_windows": len(ds),
        "backbone_sha256": backbone_sha,
        "residual_sha256": residual_sha,
        "shift_values": list(SHIFT_VALUES),
        "temporal_extrapolation_gammas": list(GAMMAS),
        "optimizer_steps": 0,
        "training_performed": False,
        "aoa10_accessed": False,
        "locked_final_accessed": False,
        "codabench_accessed": False,
        "full_data_refit_started": False,
        "submission_packaging_started": False,
    })
    (args.out_dir / "DONE").touch()


if __name__ == "__main__":
    main()
