#!/usr/bin/env python3
"""Zero-training fluctuation-coherence diagnosis for Exp1 Strong Backbone tail.

This is the final mechanism diagnostic for the F19/F20 cliff. It decomposes each
Future20 prediction and target into its own temporal mean field plus a zero-mean
fluctuation, then measures:

- temporal-mean field error;
- fluctuation amplitude ratio by horizon;
- fluctuation cosine/coherence by horizon;
- least-squares fluctuation scale after the means are separated correctly;
- a 20x20 Pred-horizon vs GT-horizon fluctuation cosine matrix;
- best GT phase match / offset for every predicted horizon;
- per-trajectory tail coherence for F18/F19/F20.

It performs zero optimizer steps and accepts Clean Seen-Dev12 only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from colleague_80pt.realpde_h5_feature_adapter_train import (
    H5WindowDataset,
    paths_from_split_manifest,
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
)

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
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _uv(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 5 or a.shape[1] != 20 or a.shape[-1] < 2:
        raise ValueError(f"expected [N,20,H,W,C>=2], got {a.shape}")
    return a[..., :2]


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(av - bv) / max(np.linalg.norm(bv), 1e-30))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(av, bv) / max(np.linalg.norm(av) * np.linalg.norm(bv), 1e-30))


def _window_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape or a.ndim != 4:
        raise ValueError(f"expected matched [N,H,W,C], got {a.shape}/{b.shape}")
    av = np.asarray(a, dtype=np.float64).reshape(a.shape[0], -1)
    bv = np.asarray(b, dtype=np.float64).reshape(b.shape[0], -1)
    num = np.sum(av * bv, axis=1)
    den = np.linalg.norm(av, axis=1) * np.linalg.norm(bv, axis=1)
    return num / np.maximum(den, 1e-30)


def decompose(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uv = _uv(x)
    mean = uv.mean(axis=1, keepdims=True)
    fluct = uv - mean
    # The numerical check matters because the Strong Backbone architecture
    # explicitly imposes zero-mean Future20 fluctuations.
    if float(np.abs(fluct.mean(axis=1)).max()) > 1e-10:
        raise RuntimeError("fluctuation decomposition is not temporally zero-mean")
    return mean, fluct


def mean_field_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pm, _ = decompose(pred)
    tm, _ = decompose(target)
    p = pm[:, 0]
    t = tm[:, 0]
    per_window = np.asarray([_rel_l2(p[i], t[i]) for i in range(p.shape[0])], dtype=np.float64)
    pnorm = np.linalg.norm(p.reshape(p.shape[0], -1), axis=1)
    tnorm = np.linalg.norm(t.reshape(t.shape[0], -1), axis=1)
    return {
        "mean_field_rel_l2_global": _rel_l2(p, t),
        "mean_field_rel_l2_window_mean": float(np.mean(per_window)),
        "mean_field_rel_l2_window_median": float(np.median(per_window)),
        "mean_field_norm_ratio_global": float(np.linalg.norm(p.reshape(-1)) / max(np.linalg.norm(t.reshape(-1)), 1e-30)),
        "mean_field_norm_ratio_window_mean": float(np.mean(pnorm / np.maximum(tnorm, 1e-30))),
    }


def fluctuation_horizon_rows(pred: np.ndarray, target: np.ndarray) -> list[dict[str, object]]:
    _pm, pf = decompose(pred)
    _tm, tf = decompose(target)
    rows: list[dict[str, object]] = []
    for h in range(20):
        p = pf[:, h]
        t = tf[:, h]
        pflat = p.reshape(-1)
        tflat = t.reshape(-1)
        pnorm = float(np.linalg.norm(pflat))
        tnorm = float(np.linalg.norm(tflat))
        dot = float(np.dot(pflat, tflat))
        scale_star = dot / max(pnorm * pnorm, 1e-30)
        wc = _window_cosines(p, t)
        pwin = np.linalg.norm(p.reshape(p.shape[0], -1), axis=1)
        twin = np.linalg.norm(t.reshape(t.shape[0], -1), axis=1)
        amp_win = pwin / np.maximum(twin, 1e-30)
        rows.append({
            "horizon": h + 1,
            "pred_fluct_rms": float(np.sqrt(np.mean(p * p))),
            "target_fluct_rms": float(np.sqrt(np.mean(t * t))),
            "amplitude_ratio_global": pnorm / max(tnorm, 1e-30),
            "amplitude_ratio_window_mean": float(np.mean(amp_win)),
            "amplitude_ratio_window_median": float(np.median(amp_win)),
            "fluctuation_cosine_global": dot / max(pnorm * tnorm, 1e-30),
            "fluctuation_cosine_window_mean": float(np.mean(wc)),
            "fluctuation_cosine_window_median": float(np.median(wc)),
            "fluctuation_scale_star": scale_star,
            "fluctuation_rel_l2": _rel_l2(p, t),
        })
    return rows


def phase_similarity_matrix(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return global and mean-per-window 20x20 fluctuation-cosine matrices."""
    _pm, pf = decompose(pred)
    _tm, tf = decompose(target)
    global_matrix = np.empty((20, 20), dtype=np.float64)
    window_mean_matrix = np.empty((20, 20), dtype=np.float64)
    for i in range(20):
        for j in range(20):
            global_matrix[i, j] = _cosine(pf[:, i], tf[:, j])
            window_mean_matrix[i, j] = float(np.mean(_window_cosines(pf[:, i], tf[:, j])))
    return global_matrix, window_mean_matrix


def matrix_rows(matrix: np.ndarray, value_name: str) -> list[dict[str, object]]:
    if matrix.shape != (20, 20):
        raise ValueError(f"expected 20x20 matrix, got {matrix.shape}")
    rows = []
    for i in range(20):
        row: dict[str, object] = {"pred_horizon": i + 1}
        for j in range(20):
            row[f"gt_f{j+1:02d}"] = float(matrix[i, j])
        row["value"] = value_name
        rows.append(row)
    return rows


def best_phase_rows(global_matrix: np.ndarray, window_mean_matrix: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i in range(20):
        gbest = int(np.argmax(global_matrix[i]))
        wbest = int(np.argmax(window_mean_matrix[i]))
        rows.append({
            "pred_horizon": i + 1,
            "diagonal_global_cosine": float(global_matrix[i, i]),
            "best_gt_horizon_global": gbest + 1,
            "best_offset_global": gbest - i,
            "best_global_cosine": float(global_matrix[i, gbest]),
            "best_vs_diagonal_global_gain": float(global_matrix[i, gbest] - global_matrix[i, i]),
            "diagonal_window_mean_cosine": float(window_mean_matrix[i, i]),
            "best_gt_horizon_window_mean": wbest + 1,
            "best_offset_window_mean": wbest - i,
            "best_window_mean_cosine": float(window_mean_matrix[i, wbest]),
            "best_vs_diagonal_window_mean_gain": float(window_mean_matrix[i, wbest] - window_mean_matrix[i, i]),
        })
    return rows


def trajectory_tail_rows(
    pred: np.ndarray,
    target: np.ndarray,
    names: list[str],
) -> list[dict[str, object]]:
    if len(names) != pred.shape[0]:
        raise ValueError("trajectory names do not align with predictions")
    _pm, pf = decompose(pred)
    _tm, tf = decompose(target)
    rows: list[dict[str, object]] = []
    for name in sorted(set(names)):
        idx = np.asarray([i for i, value in enumerate(names) if value == name], dtype=np.int64)
        for h in TAIL_HORIZONS:
            p = pf[idx, h - 1]
            t = tf[idx, h - 1]
            wc = _window_cosines(p, t)
            pnorm = np.linalg.norm(p.reshape(p.shape[0], -1), axis=1)
            tnorm = np.linalg.norm(t.reshape(t.shape[0], -1), axis=1)
            rows.append({
                "trajectory": name,
                "horizon": h,
                "windows": int(idx.size),
                "fluctuation_cosine_global": _cosine(p, t),
                "fluctuation_cosine_window_mean": float(np.mean(wc)),
                "amplitude_ratio_global": float(np.linalg.norm(p.reshape(-1)) / max(np.linalg.norm(t.reshape(-1)), 1e-30)),
                "amplitude_ratio_window_mean": float(np.mean(pnorm / np.maximum(tnorm, 1e-30))),
            })
    return rows


def diagnose(label: str, pred: np.ndarray, target: np.ndarray, names: list[str]) -> dict[str, object]:
    mean = mean_field_metrics(pred, target)
    horizon = fluctuation_horizon_rows(pred, target)
    gm, wm = phase_similarity_matrix(pred, target)
    phase = best_phase_rows(gm, wm)
    trajectory = trajectory_tail_rows(pred, target, names)

    h18 = horizon[17]
    h19 = horizon[18]
    h20 = horizon[19]
    p18 = phase[17]
    p19 = phase[18]
    p20 = phase[19]

    summary = {
        "label": label,
        "mean_field": mean,
        "tail": {
            "f18": h18,
            "f19": h19,
            "f20": h20,
        },
        "tail_phase": {
            "f18": p18,
            "f19": p19,
            "f20": p20,
        },
        "diagnostic_flags": {
            "f20_amplitude_ratio_near_one": 0.8 <= float(h20["amplitude_ratio_global"]) <= 1.2,
            "f20_fluctuation_coherence_low": float(h20["fluctuation_cosine_global"]) < 0.5,
            "f20_best_phase_not_diagonal": int(p20["best_offset_global"]) != 0,
            "mean_field_rel_below_5pct": float(mean["mean_field_rel_l2_global"]) < 0.05,
        },
        "interpretation_rule": (
            "Amplitude ratio near 1 + low diagonal fluctuation cosine => coherence/phase problem, not amplitude. "
            "High cosine + amplitude ratio far from 1 => amplitude problem. "
            "Large mean-field error => broader tail representation/domain issue. "
            "A nonzero best phase offset in the 20x20 fluctuation matrix indicates temporal phase drift, "
            "but the offset must be interpreted together with the diagonal cosine and trajectory consistency."
        ),
    }
    return {
        "summary": summary,
        "horizon_rows": horizon,
        "global_matrix": gm,
        "window_mean_matrix": wm,
        "phase_rows": phase,
        "trajectory_tail_rows": trajectory,
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

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    train_paths, dev_paths = paths_from_split_manifest(args.real_root, args.split_manifest)
    if (len(train_paths), len(dev_paths)) != (EXPECTED_TRAIN, EXPECTED_DEV):
        raise ValueError(
            f"Clean Seen-Dev12 only: expected {EXPECTED_TRAIN}/{EXPECTED_DEV}, "
            f"got {len(train_paths)}/{len(dev_paths)}"
        )
    if set(p.name for p in train_paths) & set(p.name for p in dev_paths):
        raise ValueError("train/dev overlap")
    if manifest.get("holdout_exposed_to_training") not in (False, None):
        raise ValueError("unexpected holdout exposure in manifest")

    backbone_sha = sha256(args.backbone_checkpoint)
    residual_sha = sha256(args.residual_checkpoint)
    if backbone_sha != EXP1_BACKBONE_SHA256:
        raise RuntimeError(f"wrong Exp1 backbone SHA: {backbone_sha}")
    if residual_sha != EXP1_RESIDUAL_SHA256:
        raise RuntimeError(f"wrong Exp1 residual SHA: {residual_sha}")

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
        raise ValueError(f"expected {EXPECTED_DEV_WINDOWS} Seen-Dev windows, got {len(ds)}")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    base_parts: list[np.ndarray] = []
    final_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
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

    names = [ref.path.name for ref in ds.refs]
    backbone = diagnose("strong_backbone", base, target, names)
    final_diag = diagnose("strong_backbone_plus_residual", final, target, names)

    dump(args.out_dir / "backbone_summary.json", backbone["summary"])
    dump(args.out_dir / "final_summary.json", final_diag["summary"])
    write_rows(args.out_dir / "backbone_fluctuation_by_horizon.csv", backbone["horizon_rows"])
    write_rows(args.out_dir / "final_fluctuation_by_horizon.csv", final_diag["horizon_rows"])
    write_rows(args.out_dir / "backbone_phase_alignment.csv", backbone["phase_rows"])
    write_rows(args.out_dir / "final_phase_alignment.csv", final_diag["phase_rows"])
    write_rows(args.out_dir / "backbone_tail_by_trajectory.csv", backbone["trajectory_tail_rows"])
    write_rows(args.out_dir / "final_tail_by_trajectory.csv", final_diag["trajectory_tail_rows"])
    write_rows(
        args.out_dir / "backbone_phase_similarity_global.csv",
        matrix_rows(backbone["global_matrix"], "global_cosine"),
    )
    write_rows(
        args.out_dir / "backbone_phase_similarity_window_mean.csv",
        matrix_rows(backbone["window_mean_matrix"], "window_mean_cosine"),
    )
    write_rows(
        args.out_dir / "final_phase_similarity_global.csv",
        matrix_rows(final_diag["global_matrix"], "global_cosine"),
    )
    write_rows(
        args.out_dir / "final_phase_similarity_window_mean.csv",
        matrix_rows(final_diag["window_mean_matrix"], "window_mean_cosine"),
    )

    dump(args.out_dir / "comparison_summary.json", {
        "status": "REVIEW_REQUIRED",
        "backbone": backbone["summary"],
        "final": final_diag["summary"],
        "purpose": "Exp1 tail fluctuation coherence diagnosis",
        "optimizer_steps": 0,
        "training_performed": False,
        "selection_or_tuning_performed": False,
        "aoa10_accessed": False,
        "locked_final_accessed": False,
        "codabench_accessed": False,
        "full_data_refit_started": False,
        "submission_packaging_started": False,
    })
    dump(args.out_dir / "run_manifest.json", {
        "status": "REVIEW_REQUIRED",
        "split": "Clean Train51 / Seen-Dev12 only",
        "train_trajectories_in_manifest": len(train_paths),
        "eval_trajectories": len(dev_paths),
        "eval_windows": len(ds),
        "backbone_sha256": backbone_sha,
        "residual_sha256": residual_sha,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "optimizer_steps": 0,
        "training_performed": False,
        "aoa10_accessed": False,
        "locked_final_accessed": False,
        "codabench_accessed": False,
    })
    (args.out_dir / "DONE").touch()


if __name__ == "__main__":
    main()
