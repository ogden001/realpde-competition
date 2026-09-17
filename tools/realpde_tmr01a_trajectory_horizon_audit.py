#!/usr/bin/env python3
"""TMR-01A: trajectory x horizon audit for the frozen TMR-01 @32500 replay.

This is analysis-only. It does not train, tune, package, access SPS/uncertainty,
locked-final/private data, or Codabench. It replays the frozen SOTA-V2 @32500
backbone and its already-trained teammate residual corrector on Dev16, then
emits the missing trajectory x Future20 evidence required by
EXPERIMENT_BY_HORIZON_PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ALPHAS = (0.0, 0.5, 1.0)
BACKBONE_UPDATE = 32_500
MANIFEST_SHA256 = "42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347"
BACKBONE_SHA256 = "6926722895611c38dee79cf16ce43b87c7de3d4a1305e9db431dadd4c1a7bf47"
CORRECTOR_SHA256 = "fa1c1b46f71a16da7b08c2f36a4de0f735dd9880b007c636d1b774ccbe084dc0"
EXPECTED_DEV_TRAJECTORIES = 16
EXPECTED_DEV_WINDOWS = 659
EXPECTED_TMR01 = {
    0.0: {"rel_l2": 0.09993461519479752, "tke": 0.4692927300930023, "mvpe": 0.07577798515558243},
    0.5: {"rel_l2": 0.09295934438705444, "tke": 0.4856766164302826, "mvpe": 0.07391254603862762},
    1.0: {"rel_l2": 0.08972189575433731, "tke": 0.49859240651130676, "mvpe": 0.0733211562037468},
}
EPS = 1e-12

ERROR_METRICS = (
    "frame_rel_l2",
    "frame_rmse",
    "tke_contrib_rel_l2",
    "mvpe_probe_rel_l2",
)


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _validate_arrays(pred: np.ndarray, target: np.ndarray) -> None:
    if pred.shape != target.shape or pred.ndim != 5:
        raise ValueError("pred/target must match [N,T,H,W,C]")
    if pred.shape[1] != 20 or pred.shape[-1] < 2:
        raise ValueError("TMR-01A requires Future20 and at least u/v channels")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise FloatingPointError("non-finite prediction/target")


def mvpe_probe_geometry(height: int, width: int) -> tuple[list[int], list[int]]:
    """Frozen v9 probe geometry used by the repo's differentiable MVPE counterpart."""
    center_y = 16
    interval_y = min(2, max(1, height // 10))
    d = 16
    center_x = 10
    ys = [y for y in range(center_y - interval_y * 4, center_y + interval_y * 5, interval_y)
          if 0 <= y < height]
    if 21 < width:
        xs = [int(((i + 1) * d + center_x) / 2) for i in range(4)]
    else:
        xs = [int(0.5 * (i + 2) * d + center_x) for i in range(4)]
    xs = [x for x in xs if 0 <= x < width]
    if not ys or not xs:
        raise ValueError(f"empty MVPE probe geometry for {height}x{width}")
    return ys, xs


def _relative_l2_last_axes(pred: np.ndarray, target: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    diff_norm = np.sqrt(np.square(pred - target, dtype=np.float64).sum(axis=axes))
    target_norm = np.sqrt(np.square(target, dtype=np.float64).sum(axis=axes))
    return diff_norm / np.maximum(target_norm, EPS)


def window_horizon_diagnostics(pred: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    """Return window-level Future20 diagnostics, each shaped [N,20].

    TKE contribution follows the project protocol: each window first gets its
    own full-Future20 temporal mean, then each horizon contributes a fluctuation
    energy map relative to that mean. These are diagnostic quantities, not
    official per-frame TKE scores.
    """
    _validate_arrays(pred, target)
    p = pred[..., :2].astype(np.float64, copy=False)
    y = target[..., :2].astype(np.float64, copy=False)

    frame_rel = _relative_l2_last_axes(p, y, axes=(2, 3, 4))
    frame_rmse = np.sqrt(np.square(p - y).mean(axis=(2, 3, 4)))

    p_mean = p.mean(axis=1, keepdims=True)
    y_mean = y.mean(axis=1, keepdims=True)
    p_energy = 0.5 * np.square(p - p_mean).sum(axis=-1)
    y_energy = 0.5 * np.square(y - y_mean).sum(axis=-1)
    tke_rel = _relative_l2_last_axes(p_energy, y_energy, axes=(2, 3))
    p_energy_sum = p_energy.sum(axis=(2, 3))
    y_energy_sum = y_energy.sum(axis=(2, 3))
    tke_ratio = p_energy_sum / np.maximum(y_energy_sum, EPS)

    ys, xs = mvpe_probe_geometry(p.shape[2], p.shape[3])
    probe_terms = []
    for x in xs:
        # [N,T,len(ys),2] -> [N,T,2], matching v9's spatial probe average
        pp = p[:, :, ys, x, :].mean(axis=2)
        yy = y[:, :, ys, x, :].mean(axis=2)
        probe_terms.append(_relative_l2_last_axes(pp, yy, axes=(2,)))
    mvpe_probe = np.stack(probe_terms, axis=0).mean(axis=0)

    return {
        "frame_rel_l2": frame_rel,
        "frame_rmse": frame_rmse,
        "tke_contrib_rel_l2": tke_rel,
        "tke_contrib_ratio": tke_ratio,
        "mvpe_probe_rel_l2": mvpe_probe,
    }


def _relative_improvement(base: float, candidate: float) -> float:
    if abs(base) <= EPS:
        return 0.0 if abs(candidate) <= EPS else float("nan")
    return 1.0 - candidate / base


def build_trajectory_horizon_rows(*, predictions: dict[float, np.ndarray], target: np.ndarray,
                                  trajectory_names: np.ndarray) -> list[dict]:
    if tuple(sorted(predictions)) != ALPHAS:
        raise ValueError(f"predictions must contain exactly alphas={ALPHAS}")
    if len(trajectory_names) != len(target):
        raise ValueError("trajectory_names length must equal number of windows")

    diagnostics = {alpha: window_horizon_diagnostics(predictions[alpha], target) for alpha in ALPHAS}
    groups: dict[str, np.ndarray] = {}
    names = np.asarray(trajectory_names).astype(str)
    for name in sorted(set(names.tolist())):
        groups[name] = np.flatnonzero(names == name)

    rows: list[dict] = []
    for trajectory_id, indices in groups.items():
        for h in range(20):
            base_values = {metric: float(diagnostics[0.0][metric][indices, h].mean())
                           for metric in diagnostics[0.0]}
            for alpha in ALPHAS:
                values = {metric: float(diagnostics[alpha][metric][indices, h].mean())
                          for metric in diagnostics[alpha]}
                row = {
                    "trajectory_id": trajectory_id,
                    "horizon": h + 1,
                    "alpha": alpha,
                    "windows": int(len(indices)),
                    **values,
                }
                for metric in ERROR_METRICS:
                    row[f"{metric}_improvement_vs_base"] = _relative_improvement(
                        base_values[metric], values[metric]
                    )
                row["tke_contrib_ratio_delta_vs_base"] = values["tke_contrib_ratio"] - base_values["tke_contrib_ratio"]
                rows.append(row)
    return rows


def build_horizon_rows(*, predictions: dict[float, np.ndarray], target: np.ndarray) -> list[dict]:
    """Build the protocol's 20-row all-Dev by_horizon.csv in wide form."""
    diagnostics = {alpha: window_horizon_diagnostics(predictions[alpha], target) for alpha in ALPHAS}
    rows: list[dict] = []
    suffix = {0.0: "a0", 0.5: "a0p5", 1.0: "a1"}
    for h in range(20):
        row: dict[str, float | int] = {"horizon": h + 1}
        base_values = {metric: float(diagnostics[0.0][metric][:, h].mean()) for metric in diagnostics[0.0]}
        for alpha in ALPHAS:
            values = {metric: float(diagnostics[alpha][metric][:, h].mean()) for metric in diagnostics[alpha]}
            tag = suffix[alpha]
            for metric, value in values.items():
                row[f"{metric}_{tag}"] = value
            if alpha != 0.0:
                for metric in ERROR_METRICS:
                    row[f"{metric}_{tag}_improvement_vs_base"] = _relative_improvement(base_values[metric], values[metric])
                row[f"tke_contrib_ratio_{tag}_delta_vs_base"] = values["tke_contrib_ratio"] - base_values["tke_contrib_ratio"]
        rows.append(row)
    return rows


def horizon_trajectory_stability(trajectory_horizon_rows: list[dict]) -> list[dict]:
    """For each candidate/horizon, summarize paired improvements over Dev trajectories."""
    groups: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for row in trajectory_horizon_rows:
        alpha = float(row["alpha"])
        if alpha == 0.0:
            continue
        groups[(alpha, int(row["horizon"]))].append(row)

    output: list[dict] = []
    for (alpha, horizon), rows in sorted(groups.items()):
        result: dict[str, float | int] = {
            "alpha": alpha,
            "horizon": horizon,
            "trajectory_count": len(rows),
        }
        for metric in ERROR_METRICS:
            key = f"{metric}_improvement_vs_base"
            values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            result[f"{metric}_improved_count"] = int((finite > 0.0).sum())
            result[f"{metric}_improved_fraction"] = float((finite > 0.0).mean()) if len(finite) else float("nan")
            result[f"{metric}_mean_improvement"] = float(finite.mean()) if len(finite) else float("nan")
            result[f"{metric}_median_improvement"] = float(np.median(finite)) if len(finite) else float("nan")
            result[f"{metric}_worst_improvement"] = float(finite.min()) if len(finite) else float("nan")
            result[f"{metric}_best_improvement"] = float(finite.max()) if len(finite) else float("nan")
        output.append(result)
    return output


def _aggregate_parity(kit_root: Path, predictions: dict[float, np.ndarray], target: np.ndarray) -> dict:
    import realpde_teammate_residual_transfer as transfer

    result = {}
    for alpha in ALPHAS:
        raw = transfer.raw_physical_errors(kit_root, predictions[alpha], target)
        expected = EXPECTED_TMR01[alpha]
        delta = {metric: raw[metric] - expected[metric] for metric in expected}
        if any(abs(value) > 5e-6 for value in delta.values()):
            raise RuntimeError(f"TMR-01 aggregate parity mismatch alpha={alpha}: raw={raw}, expected={expected}")
        result[str(alpha)] = {"observed": raw, "expected": expected, "delta": delta}
    return result


def replay(args: argparse.Namespace) -> dict:
    import torch
    import realpde_sota_v2_integrated as sota
    import realpde_teammate_residual_transfer as transfer
    from realpde_adaptive_probe import ResidualCorrector3D

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if sota.sha256(args.manifest) != MANIFEST_SHA256:
        raise ValueError("frozen manifest SHA mismatch")
    if sota.sha256(args.checkpoint_32500) != BACKBONE_SHA256:
        raise ValueError("frozen @32500 backbone SHA mismatch")
    if sota.sha256(args.corrector_checkpoint) != CORRECTOR_SHA256:
        raise ValueError("frozen @32500 corrector SHA mismatch")
    if not (args.kit_root / "scoring.py").is_file():
        raise FileNotFoundError(args.kit_root / "scoring.py")

    train_paths = sota.split_paths(args.manifest, "train", args.data_root)
    dev_paths = sota.split_paths(args.manifest, "dev", args.data_root)
    if len(dev_paths) != EXPECTED_DEV_TRAJECTORIES:
        raise ValueError(f"Dev trajectories {len(dev_paths)} != {EXPECTED_DEV_TRAJECTORIES}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")

    builder, _ = sota.build_features(train_paths, device)
    backbone, _ = transfer.load_backbone(args.checkpoint_32500, BACKBONE_UPDATE, args.kit_root, builder, device)
    payload = torch.load(args.corrector_checkpoint, map_location="cpu", weights_only=False)
    corrector = ResidualCorrector3D(in_channels=42, hidden=64, blocks=2, max_delta=0.04).to(device)
    corrector.load_state_dict(payload["corrector_state_dict"], strict=True)
    transfer.freeze_module(corrector)

    ds, loader = sota.dev_loader(dev_paths, argparse.Namespace(eval_batch_size=args.eval_batch_size, workers=args.workers))
    if len(ds) != EXPECTED_DEV_WINDOWS:
        raise ValueError(f"Dev windows {len(ds)} != {EXPECTED_DEV_WINDOWS}")

    chunks: dict[float, list[np.ndarray]] = {alpha: [] for alpha in ALPHAS}
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for x, y, _, _ in loader:
            x = x.to(device, non_blocking=True)
            base = sota.forward_mf(backbone, builder, x)
            delta = transfer._delta_from_corrector(corrector, x, base)
            for alpha in ALPHAS:
                chunks[alpha].append(transfer.apply_scaled_correction(base, delta, alpha).cpu().numpy().astype(np.float32))
            targets.append(y.numpy().astype(np.float32))

    predictions = {alpha: np.concatenate(chunks[alpha], axis=0) for alpha in ALPHAS}
    target = np.concatenate(targets, axis=0)
    trajectory_names = np.asarray([ref.path.name for ref in ds.refs])
    if len(set(trajectory_names.tolist())) != EXPECTED_DEV_TRAJECTORIES:
        raise RuntimeError("trajectory mapping does not contain exactly 16 Dev trajectories")

    parity = _aggregate_parity(args.kit_root, predictions, target)
    by_horizon = build_horizon_rows(predictions=predictions, target=target)
    by_traj_horizon = build_trajectory_horizon_rows(
        predictions=predictions,
        target=target,
        trajectory_names=trajectory_names,
    )
    stability = horizon_trajectory_stability(by_traj_horizon)

    if len(by_horizon) != 20:
        raise RuntimeError(f"by_horizon rows {len(by_horizon)} != 20")
    if len(by_traj_horizon) != EXPECTED_DEV_TRAJECTORIES * 20 * len(ALPHAS):
        raise RuntimeError(f"by_trajectory_horizon rows {len(by_traj_horizon)} != 960")
    if len(stability) != 20 * 2:
        raise RuntimeError(f"stability rows {len(stability)} != 40")

    write_rows(args.out_dir / "by_horizon.csv", by_horizon)
    write_rows(args.out_dir / "by_trajectory_horizon.csv", by_traj_horizon)
    write_rows(args.out_dir / "horizon_trajectory_stability.csv", stability)
    dump(args.out_dir / "aggregate_parity.json", parity)

    metadata = {
        "experiment": "TMR-01A_trajectory_x_horizon_audit",
        "status": "REVIEW_REQUIRED",
        "analysis_only": True,
        "backbone_update": BACKBONE_UPDATE,
        "manifest_sha256": MANIFEST_SHA256,
        "backbone_sha256": BACKBONE_SHA256,
        "corrector_sha256": CORRECTOR_SHA256,
        "dev_trajectories": EXPECTED_DEV_TRAJECTORIES,
        "dev_windows": EXPECTED_DEV_WINDOWS,
        "alphas": list(ALPHAS),
        "by_horizon_rows": len(by_horizon),
        "by_trajectory_horizon_rows": len(by_traj_horizon),
        "stability_rows": len(stability),
        "metric_notes": {
            "frame_rel_l2": "window-level per-frame u/v relative L2, then aggregated",
            "frame_rmse": "window-level per-frame u/v RMSE, then aggregated",
            "tke_contrib_rel_l2": "diagnostic Future20 temporal-mean fluctuation-energy contribution; not official per-frame TKE",
            "tke_contrib_ratio": "predicted/target fluctuation-energy contribution ratio; not official per-frame TKE",
            "mvpe_probe_rel_l2": "v9 probe geometry without Future20 temporal averaging; diagnostic, not official per-frame MVPE",
        },
        "scope": {
            "training": "NOT_PERFORMED",
            "sps": "NOT_ACCESSED",
            "uncertainty": "NOT_ACCESSED",
            "full_data": "NOT_ACCESSED",
            "locked_final_private": "NOT_ACCESSED",
            "package": "NOT_BUILT",
            "codabench": "NOT_ACCESSED",
        },
    }
    dump(args.out_dir / "run_metadata.json", metadata)
    dump(args.out_dir / "status.json", {"state": "DONE", "status": "REVIEW_REQUIRED"})
    return metadata


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
