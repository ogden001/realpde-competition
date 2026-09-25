#!/usr/bin/env python3
"""Zero-training diagnosis for the Exp1 Strong-Backbone F19/F20 tail cliff.

This script is deliberately narrow:
- evaluate ONLY Clean Seen-Dev12 from the frozen Train51/Seen-Dev12 manifest;
- load the exact Exp1 strong-backbone + residual checkpoints;
- compare backbone prediction vs backbone+residual on the same windows;
- quantify whether F19/F20 comes from backbone error growth, loss of residual
  correction effectiveness, or residual direction/magnitude mismatch;
- perform zero optimizer updates and never open AoA10/locked-final/Codabench.

Outputs are diagnostic evidence, not a new model-selection experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from colleague_80pt.realpde_h5_feature_adapter_train import H5WindowDataset, paths_from_split_manifest
from colleague_80pt.residual_multi import (
    CorrectorConfig,
    ResidualCorrectionModel,
    ResidualCorrector3D,
    load_frozen_base,
    load_residual_checkpoint,
)
from dw01_by_horizon import aggregate_by_horizon, compute_window_horizon_metrics, write_csv

EXP1_BACKBONE_SHA256 = "d340effd68031aba1cd2bc610c676878fb9105dd7f81b9ce15767280e665a5c0"
EXP1_RESIDUAL_SHA256 = "d2b4ddd0df6064d053ec918c2671a93b7d9c2a40a15afe854bb3a5badf1438fc"
EXPECTED_TRAIN = 51
EXPECTED_DEV = 12
EXPECTED_DEV_WINDOWS = 491


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _flatten_uv(x: np.ndarray) -> np.ndarray:
    return np.asarray(x[..., :2], dtype=np.float64).reshape(-1)


def residual_geometry_by_horizon(
    base: np.ndarray,
    final: np.ndarray,
    target: np.ndarray,
) -> list[dict[str, float | int]]:
    """Measure residual direction, magnitude and actual error reduction per horizon.

    alpha_star is the least-squares scalar on the *existing residual direction*:
        argmin_a ||base + a * delta - target||^2.
    It diagnoses magnitude mismatch without retraining or changing the submitted
    predictor.  alpha_star ~= 1 means the learned magnitude is locally suitable;
    >1 suggests under-correction; <1 suggests over-correction; negative values
    mean the correction direction is harmful.
    """
    if base.shape != final.shape or base.shape != target.shape:
        raise ValueError("base/final/target shapes must match")
    if base.ndim != 5 or base.shape[1] != 20 or base.shape[-1] < 2:
        raise ValueError(f"expected [N,20,H,W,C>=2], got {base.shape}")

    rows: list[dict[str, float | int]] = []
    for h in range(20):
        b = np.asarray(base[:, h, ..., :2], dtype=np.float64)
        f = np.asarray(final[:, h, ..., :2], dtype=np.float64)
        y = np.asarray(target[:, h, ..., :2], dtype=np.float64)
        delta = f - b
        residual_target = y - b

        d = delta.reshape(delta.shape[0], -1)
        r = residual_target.reshape(residual_target.shape[0], -1)
        base_err = b - y
        final_err = f - y
        base_sse_per_window = np.sum(base_err * base_err, axis=(1, 2, 3))
        final_sse_per_window = np.sum(final_err * final_err, axis=(1, 2, 3))

        dflat, rflat = d.reshape(-1), r.reshape(-1)
        dot = float(np.dot(dflat, rflat))
        d2 = float(np.dot(dflat, dflat))
        r2 = float(np.dot(rflat, rflat))
        cosine = dot / max(math.sqrt(d2 * r2), 1e-30)
        alpha_star = dot / max(d2, 1e-30)

        oracle_err = residual_target - alpha_star * delta
        oracle_sse = float(np.sum(oracle_err * oracle_err))
        base_sse = float(np.sum(base_err * base_err))
        final_sse = float(np.sum(final_err * final_err))

        rows.append({
            "horizon": h + 1,
            "delta_rms": float(np.sqrt(np.mean(delta * delta))),
            "residual_target_rms": float(np.sqrt(np.mean(residual_target * residual_target))),
            "delta_to_target_rms_ratio": float(np.sqrt(d2 / max(r2, 1e-30))),
            "delta_target_cosine": float(cosine),
            "alpha_star": float(alpha_star),
            "help_fraction": float(np.mean(final_sse_per_window < base_sse_per_window)),
            "base_sse": base_sse,
            "final_sse": final_sse,
            "sse_gain_pct": 100.0 * (base_sse - final_sse) / max(base_sse, 1e-30),
            "oracle_alpha_sse_gain_pct": 100.0 * (base_sse - oracle_sse) / max(base_sse, 1e-30),
        })
    return rows


def build_tail_rows(
    base_horizon: list[dict[str, object]],
    final_horizon: list[dict[str, object]],
    geometry: list[dict[str, float | int]],
) -> list[dict[str, object]]:
    if len(base_horizon) != 20 or len(final_horizon) != 20 or len(geometry) != 20:
        raise ValueError("expected exactly 20 horizon rows")
    result: list[dict[str, object]] = []
    for b, f, g in zip(base_horizon, final_horizon, geometry, strict=True):
        if int(b["horizon"]) != int(f["horizon"]) or int(b["horizon"]) != int(g["horizon"]):
            raise ValueError("horizon alignment mismatch")
        base_rel = float(b["frame_rel_l2"])
        final_rel = float(f["frame_rel_l2"])
        base_rmse = float(b["frame_rmse"])
        final_rmse = float(f["frame_rmse"])
        result.append({
            "horizon": int(b["horizon"]),
            "base_frame_rel_l2": base_rel,
            "final_frame_rel_l2": final_rel,
            "correction_gain_rel_pct": 100.0 * (base_rel - final_rel) / max(base_rel, 1e-30),
            "base_frame_rmse": base_rmse,
            "final_frame_rmse": final_rmse,
            "correction_gain_rmse_pct": 100.0 * (base_rmse - final_rmse) / max(base_rmse, 1e-30),
            "base_tke_contrib_rel_l2": float(b["tke_contrib_rel_l2"]),
            "final_tke_contrib_rel_l2": float(f["tke_contrib_rel_l2"]),
            "base_mvpe_probe_rel_l2": float(b["mvpe_probe_rel_l2"]),
            "final_mvpe_probe_rel_l2": float(f["mvpe_probe_rel_l2"]),
            **{k: v for k, v in g.items() if k != "horizon"},
        })
    return result


def summarize_tail(rows: list[dict[str, object]]) -> dict[str, object]:
    if len(rows) != 20:
        raise ValueError("tail summary requires 20 horizons")

    def row(h: int) -> dict[str, object]:
        return rows[h - 1]

    def rel_growth(a: int, b: int, key: str) -> float:
        x, y = float(row(a)[key]), float(row(b)[key])
        return 100.0 * (y - x) / max(abs(x), 1e-30)

    reference_gain = float(np.mean([float(row(h)["correction_gain_rel_pct"]) for h in range(15, 19)]))
    late_gain = float(np.mean([float(row(h)["correction_gain_rel_pct"]) for h in (19, 20)]))
    h18_gain = float(row(18)["correction_gain_rel_pct"])
    h20_gain = float(row(20)["correction_gain_rel_pct"])
    h18_cos = float(row(18)["delta_target_cosine"])
    h20_cos = float(row(20)["delta_target_cosine"])

    return {
        "h18": row(18),
        "h19": row(19),
        "h20": row(20),
        "base_rel_growth_h18_to_h19_pct": rel_growth(18, 19, "base_frame_rel_l2"),
        "base_rel_growth_h19_to_h20_pct": rel_growth(19, 20, "base_frame_rel_l2"),
        "final_rel_growth_h18_to_h19_pct": rel_growth(18, 19, "final_frame_rel_l2"),
        "final_rel_growth_h19_to_h20_pct": rel_growth(19, 20, "final_frame_rel_l2"),
        "reference_correction_gain_h15_h18_pct": reference_gain,
        "late_correction_gain_h19_h20_pct": late_gain,
        "correction_gain_drop_h18_to_h20_pp": h18_gain - h20_gain,
        "correction_cosine_drop_h18_to_h20": h18_cos - h20_cos,
        "diagnostic_flags": {
            "backbone_tail_growth_present": rel_growth(18, 20, "base_frame_rel_l2") >= 10.0,
            "residual_effectiveness_drop_present": h20_gain <= h18_gain - 10.0,
            "residual_alignment_drop_present": h20_cos <= h18_cos - 0.10,
        },
        "interpretation_rule": (
            "Backbone growth with stable correction => backbone/temporal-boundary issue; "
            "stable backbone with falling correction gain/cosine => residual tail failure; "
            "both flags => mixed mechanism."
        ),
    }


def trajectory_tail_rows(
    base_window_rows: list[dict[str, object]],
    final_window_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    base_map = {
        (str(r["trajectory"]), int(r["window_start"]), int(r["horizon"])): float(r["frame_rel_l2"])
        for r in base_window_rows if int(r["horizon"]) >= 18
    }
    final_map = {
        (str(r["trajectory"]), int(r["window_start"]), int(r["horizon"])): float(r["frame_rel_l2"])
        for r in final_window_rows if int(r["horizon"]) >= 18
    }
    if set(base_map) != set(final_map):
        raise ValueError("base/final window-horizon keys differ")

    grouped: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for (trajectory, _start, horizon), b in base_map.items():
        grouped[(trajectory, horizon)].append((b, final_map[(trajectory, _start, horizon)]))

    rows: list[dict[str, object]] = []
    for (trajectory, horizon), pairs in sorted(grouped.items()):
        b = float(np.mean([p[0] for p in pairs]))
        f = float(np.mean([p[1] for p in pairs]))
        rows.append({
            "trajectory": trajectory,
            "horizon": horizon,
            "windows": len(pairs),
            "base_frame_rel_l2": b,
            "final_frame_rel_l2": f,
            "correction_gain_rel_pct": 100.0 * (b - f) / max(b, 1e-30),
        })
    return rows


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

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    train_paths, dev_paths = paths_from_split_manifest(args.real_root, args.split_manifest)
    if len(train_paths) != EXPECTED_TRAIN or len(dev_paths) != EXPECTED_DEV:
        raise ValueError(
            f"this diagnostic is Seen-Dev12 only; expected {EXPECTED_TRAIN}/{EXPECTED_DEV}, "
            f"got {len(train_paths)}/{len(dev_paths)}"
        )
    if set(p.name for p in train_paths) & set(p.name for p in dev_paths):
        raise ValueError("train/dev overlap")
    if manifest.get("holdout_exposed_to_training") not in (False, None):
        raise ValueError("unexpected manifest holdout exposure")

    backbone_sha = sha256(args.backbone_checkpoint)
    residual_sha = sha256(args.residual_checkpoint)
    if backbone_sha != EXP1_BACKBONE_SHA256:
        raise RuntimeError(f"wrong Exp1 backbone checkpoint SHA: {backbone_sha}")
    if residual_sha != EXP1_RESIDUAL_SHA256:
        raise RuntimeError(f"wrong Exp1 residual checkpoint SHA: {residual_sha}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")

    residual_payload = torch.load(args.residual_checkpoint, map_location="cpu", weights_only=False)
    cfg_raw = residual_payload.get("corrector_config")
    if not isinstance(cfg_raw, dict):
        raise ValueError("residual checkpoint missing corrector_config")

    base_model = load_frozen_base("sota_v2_mf", args.backbone_checkpoint, args.model_root, device)
    model = ResidualCorrectionModel(base_model, ResidualCorrector3D(CorrectorConfig(**cfg_raw))).to(device)
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
        raise ValueError(f"expected {EXPECTED_DEV_WINDOWS} Seen-Dev windows, got {len(ds)}")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    base_predictions: list[np.ndarray] = []
    final_predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            base = model.base_predict(x)
            delta = model.predict_delta(x, base)
            final = model.combine(base, delta, 1.0)
            base_predictions.append(base.cpu().numpy().astype(np.float32))
            final_predictions.append(final.cpu().numpy().astype(np.float32))
            targets.append(y.numpy().astype(np.float32))

    base = np.concatenate(base_predictions, axis=0)
    final = np.concatenate(final_predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    if not (np.isfinite(base).all() and np.isfinite(final).all() and np.isfinite(target).all()):
        raise FloatingPointError("non-finite prediction or target")

    names = [ref.path.name for ref in ds.refs]
    starts = [ref.start for ref in ds.refs]
    base_window = compute_window_horizon_metrics(base, target, names, starts)
    final_window = compute_window_horizon_metrics(final, target, names, starts)
    base_horizon = aggregate_by_horizon(base_window, experiment="strong_backbone", trajectories=EXPECTED_DEV)
    final_horizon = aggregate_by_horizon(final_window, experiment="strong_backbone_plus_residual", trajectories=EXPECTED_DEV)
    geometry = residual_geometry_by_horizon(base, final, target)
    tail_rows = build_tail_rows(base_horizon, final_horizon, geometry)
    summary = summarize_tail(tail_rows)

    write_csv(args.out_dir / "backbone_by_horizon.csv", base_horizon, list(base_horizon[0]))
    write_csv(args.out_dir / "final_by_horizon.csv", final_horizon, list(final_horizon[0]))
    write_csv(args.out_dir / "tail_diagnostic_by_horizon.csv", tail_rows, list(tail_rows[0]))
    tr_rows = trajectory_tail_rows(base_window, final_window)
    write_csv(args.out_dir / "tail_by_trajectory.csv", tr_rows, list(tr_rows[0]))
    dump(args.out_dir / "tail_summary.json", summary)
    dump(args.out_dir / "run_manifest.json", {
        "status": "REVIEW_REQUIRED",
        "purpose": "Exp1 F19/F20 zero-training diagnosis",
        "split": "Clean Seen-Dev12 only",
        "train_trajectories_in_manifest": len(train_paths),
        "eval_trajectories": len(dev_paths),
        "eval_windows": len(ds),
        "backbone_sha256": backbone_sha,
        "residual_sha256": residual_sha,
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
