#!/usr/bin/env python3
"""Coarse mean-preserving fluctuation-amplitude calibration screen.

This experiment starts from the frozen colleague 80-point residual checkpoint.
It does not train any network.  It keeps the predicted Future20 temporal mean
exactly fixed and only changes the amplitude of the zero-mean dynamic component:

    pred_t = mean(pred) + fluct_t
    corrected_t = mean(pred) + recenter(scale_t * fluct_t)

where scale_t grows linearly from 1.0 at Future1 to a small candidate end scale
at Future20.  The re-centering step guarantees that the Future20 temporal mean
is unchanged, so this probe isolates dynamic-amplitude calibration rather than
mean-flow correction.

The script writes the full standard post-train diagnostic bundle for every
candidate.  It never accesses locked-final data and never trains a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from post_train_diagnostics import (  # noqa: E402
    horizon_rows,
    spatial_maps,
    trajectory_rows,
    write_post_train_diagnostics,
)
from realpde_h5_feature_adapter_train import (  # noqa: E402
    H5WindowDataset,
    measured_channels,
    mvpe_rel_l2_per_sample,
    paths_from_split_manifest,
    rel_l2_per_sample,
    tke_rel_l2_per_sample,
)
from residual_multi import load_full_residual_model  # noqa: E402


EXPECTED_START_SHA256 = "909fdc7f8a6a42335e4ea4ce7471fc9a507135e9a29a4927bc7b7318be9c85b2"
EXPECTED_BASELINE = {
    "rel_l2_raw": 0.0804204195737838,
    "tke_raw": 0.4491582512855530,
    "mvpe_raw": 0.0711169168353080,
}
BASELINE_TOLERANCE = 2e-4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_scales(text: str) -> list[float]:
    values = [float(item) for item in text.replace(";", ",").split(",") if item.strip()]
    if not values:
        raise ValueError("at least one end scale is required")
    if any(value < 1.0 for value in values):
        raise ValueError("this bounded screen only permits end scales >= 1.0")
    if 1.0 not in values:
        values = [1.0, *values]
    return sorted(set(values))


def mean_preserving_ramp_scale(
    prediction: np.ndarray,
    end_scale: float,
    *,
    start_scale: float = 1.0,
) -> np.ndarray:
    """Increase only temporal fluctuation amplitude while preserving Future mean."""
    pred = np.asarray(prediction, dtype=np.float32)
    if pred.ndim != 5 or pred.shape[-1] < 2:
        raise ValueError(f"expected [N,T,H,W,C>=2], got {pred.shape}")
    if start_scale <= 0 or end_scale <= 0:
        raise ValueError("scales must be positive")
    out = pred.copy()
    velocity = pred[..., :2].astype(np.float64, copy=False)
    temporal_mean = velocity.mean(axis=1, keepdims=True)
    fluctuation = velocity - temporal_mean
    ramp = np.linspace(
        float(start_scale),
        float(end_scale),
        pred.shape[1],
        dtype=np.float64,
    ).reshape(1, pred.shape[1], 1, 1, 1)
    scaled_fluctuation = fluctuation * ramp
    # A horizon-varying scale can introduce a non-zero temporal mean. Remove it
    # so the output Future20 mean remains exactly the original predicted mean.
    scaled_fluctuation -= scaled_fluctuation.mean(axis=1, keepdims=True)
    corrected = temporal_mean + scaled_fluctuation
    out[..., :2] = corrected.astype(np.float32)
    if out.shape[-1] >= 3:
        out[..., 2] = 0.0
    return out


def raw_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    channels = measured_channels(target)
    return {
        "rel_l2_raw": float(np.mean(rel_l2_per_sample(prediction, target, channels))),
        "tke_raw": float(np.mean(tke_rel_l2_per_sample(prediction, target, channels))),
        "mvpe_raw": float(np.mean(mvpe_rel_l2_per_sample(prediction, target))),
    }


def delta_pct(value: float, baseline: float) -> float:
    return 100.0 * (float(value) - float(baseline)) / max(abs(float(baseline)), 1e-12)


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    # Some diagnostic rows intentionally have candidate-only columns, e.g.
    # residual-before/after fields are absent for the scale=1.0 baseline but
    # present for calibrated candidates. Build a stable union schema instead
    # of assuming every row has exactly the first row's keys.
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_spatial_summary(
    rows: list[dict[str, object]],
    *,
    label: str,
    maps: dict[str, np.ndarray],
) -> None:
    for metric, value in maps.items():
        arr = np.asarray(value, dtype=np.float64)
        flat_index = int(np.argmax(arr))
        max_row, max_col = np.unravel_index(flat_index, arr.shape)
        rows.append({
            "model": label,
            "metric": metric,
            "mean": float(arr.mean()),
            "p95": float(np.percentile(arr, 95.0)),
            "max": float(arr.max()),
            "max_row": int(max_row),
            "max_col": int(max_col),
        })


@torch.no_grad()
def collect_predictions(
    checkpoint: Path,
    real_root: Path,
    split_manifest: Path,
    realpdebench_root: Path,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    _, val_paths = paths_from_split_manifest(
        real_root,
        split_manifest,
        allow_train_dev_overlap=True,
    )
    dataset = H5WindowDataset(
        val_paths,
        in_steps=20,
        out_steps=20,
        stride=20,
        sub_sample=2,
        include_pressure=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    model, _ = load_full_residual_model(checkpoint, realpdebench_root, device)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        pred = model(x, alpha=1.0)
        predictions.append(pred.cpu().numpy().astype(np.float32))
        targets.append(y.numpy().astype(np.float32))
    names = [ref.path.name for ref in dataset.refs]
    starts = [int(ref.start) for ref in dataset.refs]
    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        names,
        starts,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--realpdebench-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--end-scales",
        default="1.0,1.2,1.4,1.6",
        help="Coarse Future20 end-scale scan. Future1 is always scale 1.0.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing out dir: {args.out_dir}")
    for path in (args.real_root, args.split_manifest, args.checkpoint, args.realpdebench_root):
        if not path.exists():
            raise FileNotFoundError(path)
    if sha256(args.checkpoint) != EXPECTED_START_SHA256:
        raise RuntimeError("checkpoint is not the frozen colleague 80-point residual baseline")

    args.out_dir.mkdir(parents=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prediction, target, names, starts = collect_predictions(
        args.checkpoint,
        args.real_root,
        args.split_manifest,
        args.realpdebench_root,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    baseline_metrics = raw_metrics(prediction, target)
    for key, expected in EXPECTED_BASELINE.items():
        if abs(baseline_metrics[key] - expected) > BASELINE_TOLERANCE:
            raise RuntimeError(
                f"baseline parity failed for {key}: {baseline_metrics[key]} vs {expected}"
            )

    scales = parse_scales(args.end_scales)
    scan_rows: list[dict[str, object]] = []
    horizon_compare: list[dict[str, object]] = []
    trajectory_compare: list[dict[str, object]] = []
    spatial_summary_rows: list[dict[str, object]] = []
    baseline_horizon: dict[int, dict[str, object]] | None = None
    baseline_trajectory: dict[str, dict[str, object]] | None = None

    for scale in scales:
        label = f"fluct_ramp_end_{scale:.2f}".replace(".", "p")
        candidate = mean_preserving_ramp_scale(prediction, scale)
        metrics = raw_metrics(candidate, target)
        candidate_dir = args.out_dir / label
        candidate_dir.mkdir()
        diagnostics = write_post_train_diagnostics(
            out_dir=candidate_dir / "diagnostics",
            experiment=label,
            prediction=candidate,
            target=target,
            trajectories=names,
            starts=starts,
            base_prediction=prediction if scale != 1.0 else None,
        )
        (candidate_dir / "final_primary_metrics.json").write_text(
            json.dumps(
                {
                    "experiment": label,
                    "end_scale": scale,
                    "start_scale": 1.0,
                    "mean_preserved": True,
                    **metrics,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )

        hrows = horizon_rows(
            candidate,
            target,
            experiment=label,
            base_prediction=prediction if scale != 1.0 else None,
        )
        trows = trajectory_rows(candidate, target, names, experiment=label)
        if scale == 1.0:
            baseline_horizon = {int(row["horizon"]): row for row in hrows}
            baseline_trajectory = {str(row["trajectory"]): row for row in trows}
        assert baseline_horizon is not None
        assert baseline_trajectory is not None

        for row in hrows:
            base = baseline_horizon[int(row["horizon"])]
            horizon_compare.append({
                **row,
                "delta_rel_pct_vs_current80": delta_pct(
                    float(row["frame_rel_l2"]), float(base["frame_rel_l2"])
                ),
                "delta_tke_contrib_pct_vs_current80": delta_pct(
                    float(row["tke_contrib_rel_l2"]),
                    float(base["tke_contrib_rel_l2"]),
                ),
            })
        for row in trows:
            base = baseline_trajectory[str(row["trajectory"])]
            trajectory_compare.append({
                **row,
                "delta_rel_pct_vs_current80": delta_pct(
                    float(row["rel_l2"]), float(base["rel_l2"])
                ),
                "delta_tke_pct_vs_current80": delta_pct(
                    float(row["tke_rel_l2"]), float(base["tke_rel_l2"])
                ),
                "delta_fluctuation_pct_vs_current80": delta_pct(
                    float(row["fluctuation_rel_l2"]),
                    float(base["fluctuation_rel_l2"]),
                ),
            })

        maps = spatial_maps(candidate, target)
        write_spatial_summary(spatial_summary_rows, label=label, maps=maps)
        mean_fluct = diagnostics["mean_fluctuation"]
        rel_delta = delta_pct(metrics["rel_l2_raw"], baseline_metrics["rel_l2_raw"])
        tke_delta = delta_pct(metrics["tke_raw"], baseline_metrics["tke_raw"])
        mvpe_delta = delta_pct(metrics["mvpe_raw"], baseline_metrics["mvpe_raw"])
        scan_rows.append({
            "model": label,
            "end_scale": scale,
            **metrics,
            "tke_energy_ratio": float(mean_fluct["tke_energy_ratio"]),
            "delta_rel_pct_vs_current80": rel_delta,
            "delta_tke_pct_vs_current80": tke_delta,
            "delta_mvpe_pct_vs_current80": mvpe_delta,
            "mechanical_tke_gain_ge_2pct": bool(tke_delta <= -2.0),
            "mechanical_rel_guard_le_0p5pct": bool(rel_delta <= 0.5),
            "mechanical_mvpe_guard_le_0p2pct": bool(mvpe_delta <= 0.2),
        })

    write_csv(args.out_dir / "scan.csv", scan_rows)
    write_csv(args.out_dir / "comparison_by_horizon.csv", horizon_compare)
    write_csv(args.out_dir / "trajectory_comparison_summary.csv", trajectory_compare)
    write_csv(args.out_dir / "spatial_summary.csv", spatial_summary_rows)
    manifest = {
        "experiment": "mean_preserving_fluctuation_amplitude_calibration",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": sha256(args.split_manifest),
        "execution_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=SCRIPT_DIR.parents[1],
            text=True,
        ).strip(),
        "dev_windows": int(prediction.shape[0]),
        "dev_trajectories": len(set(names)),
        "end_scales": scales,
        "future1_scale": 1.0,
        "temporal_mean_preserved": True,
        "training_started": False,
        "locked_final_accessed": False,
        "codabench_accessed": False,
    }
    (args.out_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "DONE").touch()
    print(json.dumps({"status": "DONE", "scan": scan_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
