#!/usr/bin/env python3
"""One-shot merge calibration for the fully trained residual corrector.

This is analysis-only. It compares the frozen @32500 backbone, the frozen
30k residual corrector, and two deterministic energy-preserving projections:

1. window_energy: preserve the corrected temporal mean while restoring the
   backbone's total Future20 fluctuation energy for each window.
2. spatial_tke_map: preserve the corrected temporal mean while restoring the
   backbone's Future20 TKE magnitude at every spatial location.

No training, alpha sweep, loss tuning, architecture changes, SPS, uncertainty,
full-data, locked-final/private data, packaging, or Codabench access occurs.
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

import realpde_loss_official_v9 as core
import realpde_residual_corrector_longtrain as longtrain
import realpde_sota_v2_integrated as sota
import realpde_teammate_residual_transfer as transfer
import realpde_tmr01a_trajectory_horizon_audit as audit

BACKBONE_UPDATE = 32_500
CORRECTOR_UPDATE = 30_000
BACKBONE_SHA256 = longtrain.BACKBONE_SHA256
EXPECTED_DEV_TRAJECTORIES = 16
EXPECTED_DEV_WINDOWS = 659
VARIANTS = ("base", "corrected", "window_energy", "spatial_tke_map")
EPS = 1e-12


def _validate_pair(base: torch.Tensor, corrected: torch.Tensor) -> None:
    if base.shape != corrected.shape or base.ndim != 5:
        raise ValueError("base/corrected must match [B,T,H,W,C]")
    if base.shape[1] != 20 or base.shape[-1] < 3:
        raise ValueError("projection requires Future20 with u/v/p channels")
    if not torch.isfinite(base).all() or not torch.isfinite(corrected).all():
        raise FloatingPointError("non-finite base/corrected prediction")


def window_energy_projection(base: torch.Tensor, corrected: torch.Tensor) -> torch.Tensor:
    """Restore per-window total fluctuation energy to the frozen backbone.

    The temporal mean remains the corrected prediction's mean. One positive
    scalar is applied to the corrected u/v fluctuation field per window.
    """
    _validate_pair(base, corrected)
    b_uv = base[..., :2]
    c_uv = corrected[..., :2]
    b_fluct = b_uv - b_uv.mean(dim=1, keepdim=True)
    c_mean = c_uv.mean(dim=1, keepdim=True)
    c_fluct = c_uv - c_mean

    numerator = b_fluct.double().square().sum(dim=(1, 2, 3, 4), keepdim=True)
    denominator = c_fluct.double().square().sum(dim=(1, 2, 3, 4), keepdim=True)
    scale = torch.where(
        denominator > EPS,
        torch.sqrt(numerator.clamp_min(0.0) / denominator.clamp_min(EPS)),
        torch.ones_like(denominator),
    ).to(dtype=corrected.dtype)

    out = corrected.clone()
    out[..., :2] = c_mean + scale * c_fluct
    out[..., 2] = 0.0
    return out


def spatial_tke_map_projection(base: torch.Tensor, corrected: torch.Tensor) -> torch.Tensor:
    """Restore the backbone Future20 TKE magnitude independently per pixel.

    A single positive scale per window/spatial point is shared by u/v and all
    Future20 frames, so corrected temporal phase/direction and mean are kept.
    """
    _validate_pair(base, corrected)
    b_uv = base[..., :2]
    c_uv = corrected[..., :2]
    b_fluct = b_uv - b_uv.mean(dim=1, keepdim=True)
    c_mean = c_uv.mean(dim=1, keepdim=True)
    c_fluct = c_uv - c_mean

    numerator = b_fluct.double().square().mean(dim=1).sum(dim=-1)
    denominator = c_fluct.double().square().mean(dim=1).sum(dim=-1)
    scale = torch.where(
        denominator > EPS,
        torch.sqrt(numerator.clamp_min(0.0) / denominator.clamp_min(EPS)),
        torch.ones_like(denominator),
    ).to(dtype=corrected.dtype)
    scale = scale[:, None, :, :, None]

    out = corrected.clone()
    out[..., :2] = c_mean + scale * c_fluct
    out[..., 2] = 0.0
    return out


def _window_energy_np(array: np.ndarray) -> np.ndarray:
    uv = array[..., :2].astype(np.float64)
    fluct = uv - uv.mean(axis=1, keepdims=True)
    return np.square(fluct).sum(axis=(1, 2, 3, 4))


def _spatial_tke_np(array: np.ndarray) -> np.ndarray:
    uv = array[..., :2].astype(np.float64)
    fluct = uv - uv.mean(axis=1, keepdims=True)
    return 0.5 * np.square(fluct).mean(axis=1).sum(axis=-1)


def _max_relative_delta(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right) / np.maximum(np.abs(right), EPS)))


def projection_audit(predictions: dict[str, np.ndarray]) -> dict[str, float]:
    base = predictions["base"]
    corrected = predictions["corrected"]
    return {
        "window_energy_max_relative_delta_vs_base": _max_relative_delta(
            _window_energy_np(predictions["window_energy"]),
            _window_energy_np(base),
        ),
        "spatial_tke_map_max_relative_delta_vs_base": _max_relative_delta(
            _spatial_tke_np(predictions["spatial_tke_map"]),
            _spatial_tke_np(base),
        ),
        "window_energy_mean_max_abs_delta_vs_corrected": float(
            np.max(
                np.abs(
                    predictions["window_energy"][..., :2].mean(axis=1)
                    - corrected[..., :2].mean(axis=1)
                )
            )
        ),
        "spatial_tke_map_mean_max_abs_delta_vs_corrected": float(
            np.max(
                np.abs(
                    predictions["spatial_tke_map"][..., :2].mean(axis=1)
                    - corrected[..., :2].mean(axis=1)
                )
            )
        ),
    }


def _load_corrector(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("updates", -1)) != CORRECTOR_UPDATE:
        raise ValueError(f"corrector checkpoint must be update {CORRECTOR_UPDATE}")
    if payload.get("backbone_sha256") != BACKBONE_SHA256:
        raise ValueError("corrector checkpoint backbone hash mismatch")
    return longtrain._load_frozen_corrector(path, CORRECTOR_UPDATE, device), payload


@torch.no_grad()
def replay(args: argparse.Namespace) -> dict:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if sota.sha256(args.manifest) != sota.MANIFEST_SHA:
        raise ValueError("frozen 50/16 manifest SHA mismatch")
    if sota.sha256(args.checkpoint_32500) != BACKBONE_SHA256:
        raise ValueError("frozen @32500 backbone SHA mismatch")
    if not (args.kit_root / "scoring.py").is_file():
        raise FileNotFoundError(args.kit_root / "scoring.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")

    train_paths = sota.split_paths(args.manifest, "train", args.data_root)
    dev_paths = sota.split_paths(args.manifest, "dev", args.data_root)
    if (len(train_paths), len(dev_paths)) != (50, EXPECTED_DEV_TRAJECTORIES):
        raise ValueError("expected frozen Train50/Dev16")

    builder, feature_config = sota.build_features(train_paths, device)
    backbone, _ = transfer.load_backbone(
        args.checkpoint_32500, BACKBONE_UPDATE, args.kit_root, builder, device
    )
    corrector, corrector_payload = _load_corrector(args.corrector_checkpoint, device)

    ds, loader = sota.dev_loader(
        dev_paths, Namespace(eval_batch_size=args.eval_batch_size, workers=args.workers)
    )
    if len(ds) != EXPECTED_DEV_WINDOWS:
        raise ValueError(f"Dev windows {len(ds)} != {EXPECTED_DEV_WINDOWS}")

    chunks: dict[str, list[np.ndarray]] = {name: [] for name in VARIANTS}
    targets: list[np.ndarray] = []
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        base = sota.forward_mf(backbone, builder, x)
        delta = transfer._delta_from_corrector(corrector, x, base)
        corrected = transfer.apply_scaled_correction(base, delta, 1.0)
        window_energy = window_energy_projection(base, corrected)
        spatial_map = spatial_tke_map_projection(base, corrected)

        batch_predictions = {
            "base": base,
            "corrected": corrected,
            "window_energy": window_energy,
            "spatial_tke_map": spatial_map,
        }
        for name, pred in batch_predictions.items():
            if not torch.isfinite(pred).all() or torch.count_nonzero(pred[..., 2]) != 0:
                raise FloatingPointError(f"invalid prediction for {name}")
            chunks[name].append(pred.cpu().numpy().astype(np.float32))
        targets.append(y.numpy().astype(np.float32))

    predictions = {
        name: np.concatenate(values, axis=0)
        for name, values in chunks.items()
    }
    target = np.concatenate(targets, axis=0)
    trajectory_names = np.asarray([ref.path.name for ref in ds.refs])
    if len(set(trajectory_names.tolist())) != EXPECTED_DEV_TRAJECTORIES:
        raise RuntimeError("trajectory mapping does not contain exactly 16 Dev trajectories")

    base_raw = transfer.raw_physical_errors(args.kit_root, predictions["base"], target)
    for metric, expected in longtrain.REGISTERED_BASELINE.items():
        if abs(base_raw[metric] - expected) > 5e-6:
            raise RuntimeError(
                f"registered @32500 baseline mismatch {metric}: "
                f"observed={base_raw[metric]} expected={expected}"
            )

    physical_rows: list[dict] = []
    trajectory_long: list[dict] = []
    anatomy: dict[str, object] = {}
    target_energy = transfer._fluctuation_energy(target)
    for variant in VARIANTS:
        pred = predictions[variant]
        raw = transfer.raw_physical_errors(args.kit_root, pred, target)
        trajectory, variant_anatomy = core.trajectory_rows(ds, pred, target, args.kit_root)
        anatomy[variant] = variant_anatomy
        for row in trajectory:
            trajectory_long.append({"variant": variant, **row})
        horizon = sota.horizon_error_summary(pred, target)
        energy = transfer._fluctuation_energy(pred)
        physical_rows.append({
            "variant": variant,
            "rel_l2": raw["rel_l2"],
            "tke": raw["tke"],
            "mvpe": raw["mvpe"],
            "rel_l2_improvement_vs_base": audit._relative_improvement(base_raw["rel_l2"], raw["rel_l2"]),
            "tke_improvement_vs_base": audit._relative_improvement(base_raw["tke"], raw["tke"]),
            "mvpe_improvement_vs_base": audit._relative_improvement(base_raw["mvpe"], raw["mvpe"]),
            "fluctuation_energy": energy,
            "target_fluctuation_energy": target_energy,
            "fluctuation_energy_ratio": energy / max(target_energy, 1e-30),
            "h19_fraction": horizon["h19"]["fraction"],
            "h20_fraction": horizon["h20"]["fraction"],
            "h19_h20_fraction": horizon["h19_h20"]["fraction"],
        })

    tables = longtrain.build_analysis_tables(
        predictions=predictions,
        target=target,
        trajectory_names=trajectory_names,
    )
    if len(tables["by_horizon"]) != 20 * len(VARIANTS):
        raise RuntimeError("unexpected by_horizon row count")
    if len(tables["by_trajectory_horizon"]) != EXPECTED_DEV_TRAJECTORIES * 20 * len(VARIANTS):
        raise RuntimeError("unexpected by_trajectory_horizon row count")
    if len(tables["horizon_trajectory_stability"]) != 20 * (len(VARIANTS) - 1):
        raise RuntimeError("unexpected stability row count")

    audit_result = projection_audit(predictions)
    transfer.write_rows(args.out_dir / "physical_metrics.csv", physical_rows)
    transfer.write_rows(args.out_dir / "trajectory_metrics_long.csv", trajectory_long)
    transfer.write_rows(args.out_dir / "by_horizon.csv", tables["by_horizon"])
    transfer.write_rows(args.out_dir / "by_trajectory_horizon.csv", tables["by_trajectory_horizon"])
    transfer.write_rows(
        args.out_dir / "horizon_trajectory_stability.csv",
        tables["horizon_trajectory_stability"],
    )
    transfer.dump(args.out_dir / "trajectory_anatomy.json", anatomy)
    transfer.dump(args.out_dir / "projection_audit.json", audit_result)

    metadata = {
        "experiment": "TMR-03_residual_corrector_merge_calibration",
        "status": "REVIEW_REQUIRED",
        "analysis_only": True,
        "manifest_sha256": sota.MANIFEST_SHA,
        "backbone_update": BACKBONE_UPDATE,
        "backbone_sha256": BACKBONE_SHA256,
        "corrector_update": CORRECTOR_UPDATE,
        "corrector_sha256": sota.sha256(args.corrector_checkpoint),
        "corrector_metadata": {
            "updates": int(corrector_payload["updates"]),
            "backbone_sha256": corrector_payload["backbone_sha256"],
        },
        "feature_config": vars(feature_config),
        "variants": list(VARIANTS),
        "projection_semantics": {
            "window_energy": "corrected mean + one per-window scalar restoring backbone total Future20 u/v fluctuation energy",
            "spatial_tke_map": "corrected mean + one per-pixel scalar restoring backbone Future20 TKE magnitude",
        },
        "dev_trajectories": EXPECTED_DEV_TRAJECTORIES,
        "dev_windows": EXPECTED_DEV_WINDOWS,
        "scope": {
            "training": "NOT_PERFORMED",
            "alpha_sweep": "NOT_PERFORMED",
            "loss_sweep": "NOT_PERFORMED",
            "architecture_change": "NOT_PERFORMED",
            "sps": "NOT_ACCESSED",
            "uncertainty": "NOT_ACCESSED",
            "full_data": "NOT_ACCESSED",
            "locked_final_private": "NOT_ACCESSED",
            "package": "NOT_BUILT",
            "codabench": "NOT_ACCESSED",
        },
    }
    transfer.dump(args.out_dir / "run_metadata.json", metadata)
    summary = {
        "status": "REVIEW_REQUIRED",
        "physical_metrics": physical_rows,
        "projection_audit": audit_result,
        "by_horizon_rows": len(tables["by_horizon"]),
        "by_trajectory_horizon_rows": len(tables["by_trajectory_horizon"]),
        "stability_rows": len(tables["horizon_trajectory_stability"]),
        "automatic_go_no_go": False,
        "interpretation_owner": "ChatGPT/Sol",
    }
    transfer.dump(args.out_dir / "summary.json", summary)
    transfer.dump(args.out_dir / "status.json", {"state": "DONE", "status": "REVIEW_REQUIRED"})

    del backbone, corrector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--checkpoint-32500", type=Path, required=True)
    parser.add_argument("--corrector-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    result = replay(args)
    print(json.dumps({"status": result["status"], "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
