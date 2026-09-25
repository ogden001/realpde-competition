#!/usr/bin/env python3
"""Train a conservative feature residual corrector for RealPDE Track 1.

This route keeps the current best CNO as the frozen backbone and learns only a
small correction on top of its 20-step forecast:

    final_prediction = cno_prediction + alpha * residual_corrector(features)

The corrector sees deterministic, submission-safe features built from the input
frames and from the CNO forecast.  Its last layer is initialized to zero, so the
iteration-0 model is exactly the original CNO.  During validation we scan
``alpha``; if the best alpha is 0, the run is not worth submitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, "/repo/tools")
sys.path.insert(0, "/runs/strict")

from realpde_feature_engineering import (  # noqa: E402
    augment_torch,
    feature_names,
    future_context_feature_count,
    future_context_torch,
)
from realpde_h5_feature_adapter_train import (  # noqa: E402
    BAD_TRAIN_FILES,
    H5WindowDataset,
    RandomPhaseWindowSampler,
    TrajectoryStratifiedRandomStartBatchSampler,
    finalize_scores,
    init_sps_candidates,
    list_h5,
    load_cno_checkpoint,
    load_cno_class,
    measured_channels,
    mvpe_rel_l2_per_sample,
    physics_loss,
    paths_from_split_manifest,
    rel_l2_per_sample,
    split_paths,
    tke_rel_l2_per_sample,
    update_sps_candidates,
)
from dw01_by_horizon import aggregate_by_horizon, compute_window_horizon_metrics, write_csv  # noqa: E402
from post_train_diagnostics import write_post_train_diagnostics  # noqa: E402
from aoa_meanfield_augmentation import AoAMeanFieldShiftDataset  # noqa: E402
from pareto_tke import project_tke_backward, split_residual_objective  # noqa: E402


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.replace(";", ",").split(",") if item.strip()]


def load_residual_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, object]:
    """Restore a complete base-plus-corrector checkpoint with strict parity."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError("residual checkpoint must be a mapping")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("residual checkpoint model_state_dict must be a mapping")
    fixed = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model.", "rcm."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        fixed[new_key] = value
    model.load_state_dict(fixed, strict=True)
    return checkpoint


def ensure_three_channels(x: Tensor) -> Tensor:
    if x.shape[-1] >= 3:
        return x[..., :3]
    return torch.cat([x[..., :2], torch.zeros_like(x[..., :1])], dim=-1)


def zero_pressure(x: Tensor) -> Tensor:
    if x.shape[-1] < 3:
        return x
    y = x.clone()
    y[..., 2] = 0.0
    return y


def rotate_velocity_uv(x: Tensor, angle_degrees: Tensor | float) -> Tensor:
    """Rotate u/v vector components by a per-sample angle; keep grid and p unchanged.

    This is a conservative flow-direction perturbation for training augmentation.
    It is not claimed to be an exact CFD solution at a new physical AoA.
    """
    if x.ndim != 5 or x.shape[-1] < 2:
        raise ValueError(f"expected [B,T,H,W,C>=2], got {tuple(x.shape)}")
    angle = torch.as_tensor(angle_degrees, dtype=x.dtype, device=x.device)
    if angle.ndim == 0:
        angle = angle.expand(x.shape[0])
    if angle.ndim != 1 or angle.numel() != x.shape[0]:
        raise ValueError("angle_degrees must be scalar or one value per batch sample")
    radians = torch.deg2rad(angle).view(-1, 1, 1, 1)
    cosine = torch.cos(radians)
    sine = torch.sin(radians)
    y = x.clone()
    u = x[..., 0]
    v = x[..., 1]
    y[..., 0] = cosine * u - sine * v
    y[..., 1] = sine * u + cosine * v
    return y


def future_linear_extrapolation(x: Tensor, out_steps: int) -> Tensor:
    raw = ensure_three_channels(x)
    last = raw[:, -1:]
    if raw.shape[1] > 1:
        trend = raw[:, -1:] - raw[:, -2:-1]
    else:
        trend = torch.zeros_like(last)
    steps = torch.linspace(
        1.0 / float(out_steps),
        1.0,
        out_steps,
        device=x.device,
        dtype=x.dtype,
    ).view(1, out_steps, 1, 1, 1)
    return zero_pressure(last + steps * trend)


def future_feature_count(include_pressure: bool, history_context: bool = False) -> int:
    # CNO forecast features + last-observed features + three 3-channel relation
    # blocks: linear extrapolation, CNO-last, CNO-linear.  The final context
    # block repeats history statistics and boundary-distance hints along the
    # future axis.
    count = 2 * len(feature_names(include_pressure=include_pressure)) + 9
    if history_context:
        count += future_context_feature_count()
    return count


def build_future_features(
    x: Tensor,
    base_pred: Tensor,
    *,
    include_pressure: bool,
    history_context: bool = False,
) -> Tensor:
    base = zero_pressure(ensure_three_channels(base_pred))
    out_steps = int(base.shape[1])
    last_raw = ensure_three_channels(x[:, -1:]).expand(-1, out_steps, -1, -1, -1)
    last_raw = zero_pressure(last_raw)
    linear = future_linear_extrapolation(x, out_steps)

    base_features = augment_torch(base, include_pressure=include_pressure)
    past_features = augment_torch(ensure_three_channels(x), include_pressure=include_pressure)
    last_features = past_features[:, -1:].expand(-1, out_steps, -1, -1, -1)
    pieces = [
        base_features,
        last_features,
        linear,
        base - last_raw,
        base - linear,
    ]
    if history_context:
        pieces.append(future_context_torch(x, out_steps))

    return torch.cat(pieces, dim=-1)


def norm_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups(channels), channels),
            nn.SiLU(),
            nn.Dropout3d(float(dropout)),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups(channels), channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.net(x))


@dataclass(frozen=True)
class CorrectorConfig:
    hidden: int = 32
    blocks: int = 2
    dropout: float = 0.0
    include_pressure: bool = True
    max_delta: float = 0.05
    history_context: bool = False


class ResidualCorrector3D(nn.Module):
    def __init__(self, config: CorrectorConfig) -> None:
        super().__init__()
        self.config = config
        in_channels = future_feature_count(
            include_pressure=config.include_pressure,
            history_context=config.history_context,
        )
        hidden = int(config.hidden)
        self.input_norm = nn.LayerNorm(in_channels)
        layers: list[nn.Module] = [
            nn.Conv3d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups(hidden), hidden),
            nn.SiLU(),
        ]
        for _ in range(int(config.blocks)):
            layers.append(ResidualBlock3D(hidden, dropout=config.dropout))
        layers.append(nn.Conv3d(hidden, 3, kernel_size=1))
        self.net = nn.Sequential(*layers)
        final = self.net[-1]
        if isinstance(final, nn.Conv3d):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, x: Tensor, base_pred: Tensor) -> Tensor:
        features = build_future_features(
            x,
            base_pred,
            include_pressure=self.config.include_pressure,
            history_context=self.config.history_context,
        )
        features = self.input_norm(features)
        z = features.permute(0, 4, 1, 2, 3).contiguous()
        raw_delta = self.net(z).permute(0, 2, 3, 4, 1).contiguous()
        if self.config.max_delta > 0:
            delta = self.config.max_delta * torch.tanh(raw_delta / self.config.max_delta)
        else:
            delta = raw_delta
        delta = delta.clone()
        delta[..., 2] = 0.0
        return delta


class ResidualCorrectionModel(nn.Module):
    def __init__(self, base_model: nn.Module, corrector: ResidualCorrector3D) -> None:
        super().__init__()
        self.base_model = base_model
        self.corrector = corrector

    @torch.no_grad()
    def base_predict(self, x: Tensor) -> Tensor:
        base = self.base_model(ensure_three_channels(x))
        return zero_pressure(ensure_three_channels(base))

    def predict_delta(self, x: Tensor, base_pred: Tensor) -> Tensor:
        return self.corrector(ensure_three_channels(x), base_pred)

    def combine(self, base_pred: Tensor, delta: Tensor, alpha: float) -> Tensor:
        pred = base_pred + float(alpha) * delta
        return zero_pressure(pred)

    def forward(self, x: Tensor, alpha: float = 1.0) -> Tensor:
        base = self.base_predict(x)
        delta = self.predict_delta(x, base)
        return self.combine(base, delta, alpha)


def build_cno(realpdebench_root: Path, device: torch.device) -> nn.Module:
    CNO3d = load_cno_class(realpdebench_root)
    return CNO3d(
        in_dim=3,
        out_dim=3,
        out_dim_mult=1,
        in_size=64,
        N_layers=3,
        activation="LeakyReLU",
    ).to(device)


def load_frozen_cno(checkpoint: Path, realpdebench_root: Path, device: torch.device) -> nn.Module:
    model = build_cno(realpdebench_root, device)
    load_cno_checkpoint(model, checkpoint, device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_full_residual_model(
    checkpoint_path: Path,
    realpdebench_root: Path,
    device: torch.device,
) -> tuple[ResidualCorrectionModel, dict[str, object]]:
    """Build and strictly restore an archived base-plus-corrector checkpoint."""
    metadata = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(metadata, dict):
        raise TypeError("residual checkpoint must be a mapping")
    raw_config = metadata.get("corrector_config")
    if not isinstance(raw_config, dict):
        raise KeyError("residual checkpoint is missing corrector_config")
    config = CorrectorConfig(**raw_config)
    model = ResidualCorrectionModel(
        build_cno(realpdebench_root, device),
        ResidualCorrector3D(config),
    ).to(device)
    restored = load_residual_checkpoint(model, checkpoint_path, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, restored


def load_frozen_base(model_name: str, checkpoint: Path, realpdebench_root: Path, device: torch.device) -> nn.Module:
    if model_name == "cno":
        return load_frozen_cno(checkpoint, realpdebench_root, device)
    if model_name == "sota_v2_mf":
        from strong_backbone_adapter import load_sota_v2_mf_backbone
        return load_sota_v2_mf_backbone(checkpoint, realpdebench_root, device)
    from realpde_loss_official_v9 import load_base
    model = load_base(model_name, Path("/runs/strict/kit_full"), checkpoint, device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def empty_tracker(abs_widths: Sequence[float], rel_widths: Sequence[float]) -> dict[str, object]:
    return {
        "metric_sums": {
            "n": 0,
            "rel_l2_sum": 0.0,
            "tke_sum": 0.0,
            "mvpe_sum": 0.0,
            "time_sum": 0.0,
            "time_n": 0,
        },
        "candidates": init_sps_candidates(abs_widths, rel_widths),
    }


def update_tracker(tracker: dict[str, object], pred_np: np.ndarray, target_np: np.ndarray, elapsed: float) -> None:
    pred_np = pred_np.astype(np.float32, copy=False)
    target_np = target_np.astype(np.float32, copy=False)
    pred_np[..., 2] = 0.0
    channels = measured_channels(target_np)
    rel = rel_l2_per_sample(pred_np, target_np, channels)
    tke = tke_rel_l2_per_sample(pred_np, target_np, channels)
    mvpe = mvpe_rel_l2_per_sample(pred_np, target_np)
    batch_n = int(pred_np.shape[0])
    metric_sums = tracker["metric_sums"]
    assert isinstance(metric_sums, dict)
    metric_sums["n"] += batch_n
    metric_sums["rel_l2_sum"] += float(np.sum(rel))
    metric_sums["tke_sum"] += float(np.sum(tke))
    metric_sums["mvpe_sum"] += float(np.sum(mvpe))
    metric_sums["time_sum"] += float(elapsed)
    metric_sums["time_n"] += batch_n
    update_sps_candidates(tracker["candidates"], pred_np, target_np, channels, rel, tke, mvpe)


@torch.no_grad()
def evaluate_alphas(
    model: ResidualCorrectionModel,
    loader: DataLoader,
    device: torch.device,
    *,
    alphas: Sequence[float],
    abs_widths: Sequence[float],
    rel_widths: Sequence[float],
    max_batches: int | None = None,
    fixed_time_seconds: float | None = None,
) -> list[dict[str, object]]:
    model.eval()
    trackers = {float(alpha): empty_tracker(abs_widths, rel_widths) for alpha in alphas}
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        base = model.base_predict(x)
        delta = model.predict_delta(x, base)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if fixed_time_seconds is not None:
            elapsed = float(fixed_time_seconds) * int(x.shape[0])

        base_np = base.detach().cpu().numpy().astype(np.float32, copy=False)
        delta_np = delta.detach().cpu().numpy().astype(np.float32, copy=False)
        target_np = y.detach().cpu().numpy().astype(np.float32, copy=False)
        for alpha, tracker in trackers.items():
            pred_np = base_np + float(alpha) * delta_np
            update_tracker(tracker, pred_np, target_np, elapsed)

    summaries: list[dict[str, object]] = []
    for alpha, tracker in trackers.items():
        summary = finalize_scores(tracker["metric_sums"], tracker["candidates"])
        summary["alpha"] = alpha
        summary["best_final_est"] = summary["best_bounds"][0]["final_est"]
        summary["best_bound_abs"] = summary["best_bounds"][0]["abs"]
        summary["best_bound_rel"] = summary["best_bounds"][0]["rel"]
        summary["point_score"] = float(np.mean([
            summary["rel_l2_score"],
            summary["tke_score"],
            summary["mvpe_score"],
        ]))
        summaries.append(summary)
    summaries.sort(key=lambda row: float(row["best_final_est"]), reverse=True)
    return summaries


def checkpoint_selection_score(summary: dict[str, object], metric: str) -> float:
    if metric == "point_score":
        return float(np.mean([
            float(summary["rel_l2_score"]),
            float(summary["tke_score"]),
            float(summary["mvpe_score"]),
        ]))
    if metric == "final_est":
        return float(summary["best_final_est"])
    raise ValueError(f"unsupported checkpoint selection metric: {metric}")


@torch.no_grad()
def write_final_evidence(
    model: ResidualCorrectionModel,
    loader: DataLoader,
    device: torch.device,
    *,
    out_dir: Path,
    experiment: str,
    alpha: float = 1.0,
) -> dict[str, object]:
    """Write the primary final-step aggregate, trajectory and horizon evidence."""
    model.eval()
    base_predictions: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        base = model.base_predict(x)
        delta = model.predict_delta(x, base)
        base_predictions.append(base.cpu().numpy().astype(np.float32))
        predictions.append(model.combine(base, delta, alpha).cpu().numpy().astype(np.float32))
        targets.append(y.numpy().astype(np.float32))
    base_pred = np.concatenate(base_predictions, axis=0)
    pred = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    channels = measured_channels(target)
    rel = rel_l2_per_sample(pred, target, channels)
    tke = tke_rel_l2_per_sample(pred, target, channels)
    mvpe = mvpe_rel_l2_per_sample(pred, target)
    refs = loader.dataset.refs
    names = [ref.path.name for ref in refs]
    starts = [ref.start for ref in refs]
    if len(names) != pred.shape[0]:
        raise RuntimeError("validation window metadata is misaligned with predictions")

    base_rel = rel_l2_per_sample(base_pred, target, channels)
    base_tke = tke_rel_l2_per_sample(base_pred, target, channels)
    base_mvpe = mvpe_rel_l2_per_sample(base_pred, target)
    aggregate = {
        "experiment": experiment,
        "alpha": float(alpha),
        "windows": int(pred.shape[0]),
        "trajectories": len(set(names)),
        "rel_l2_raw": float(np.mean(rel)),
        "tke_raw": float(np.mean(tke)),
        "mvpe_raw": float(np.mean(mvpe)),
        "base_rel_l2_raw": float(np.mean(base_rel)),
        "base_tke_raw": float(np.mean(base_tke)),
        "base_mvpe_raw": float(np.mean(base_mvpe)),
        "rel_l2_improvement_vs_base": float(1.0 - np.mean(rel) / np.mean(base_rel)),
        "tke_improvement_vs_base": float(1.0 - np.mean(tke) / np.mean(base_tke)),
        "mvpe_improvement_vs_base": float(1.0 - np.mean(mvpe) / np.mean(base_mvpe)),
    }
    (out_dir / "final_primary_metrics.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    trajectory_rows = []
    for name in sorted(set(names)):
        indices = np.asarray([index for index, value in enumerate(names) if value == name])
        trajectory_rows.append({
            "experiment": experiment,
            "trajectory": name,
            "windows": int(indices.size),
            "rel_l2_raw": float(np.mean(rel[indices])),
            "tke_raw": float(np.mean(tke[indices])),
            "mvpe_raw": float(np.mean(mvpe[indices])),
        })
    with (out_dir / "by_trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectory_rows[0]))
        writer.writeheader()
        writer.writerows(trajectory_rows)

    window_horizon_rows = compute_window_horizon_metrics(pred, target, names, starts)
    horizon_rows = aggregate_by_horizon(
        window_horizon_rows,
        experiment=experiment,
        trajectories=len(set(names)),
    )
    horizon_fields = list(horizon_rows[0])
    window_fields = ["experiment", *list(window_horizon_rows[0])]
    write_csv(out_dir / "by_horizon.csv", horizon_rows, horizon_fields)
    write_csv(
        out_dir / "by_trajectory_horizon.csv",
        [{"experiment": experiment, **row} for row in window_horizon_rows],
        window_fields,
    )
    write_post_train_diagnostics(
        out_dir=out_dir / "diagnostics",
        experiment=experiment,
        prediction=pred,
        target=target,
        trajectories=names,
        starts=starts,
        base_prediction=base_pred,
    )
    return aggregate


@torch.no_grad()
def write_horizon_snapshot(
    model: ResidualCorrectionModel,
    loader: DataLoader,
    device: torch.device,
    *,
    out_dir: Path,
    experiment: str,
    alpha: float = 1.0,
) -> dict[str, object]:
    """Write lightweight F1-F20 evidence for a training milestone."""
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        base = model.base_predict(x)
        delta = model.predict_delta(x, base)
        predictions.append(
            model.combine(base, delta, alpha).cpu().numpy().astype(np.float32)
        )
        targets.append(y.numpy().astype(np.float32))
    pred = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    refs = loader.dataset.refs
    names = [ref.path.name for ref in refs]
    starts = [ref.start for ref in refs]
    rows = aggregate_by_horizon(
        compute_window_horizon_metrics(pred, target, names, starts),
        experiment=experiment,
        trajectories=len(set(names)),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "by_horizon.csv", rows, list(rows[0]))
    summary = {
        "windows": int(pred.shape[0]),
        "trajectories": len(set(names)),
        "alpha": float(alpha),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def save_checkpoint(
    path: Path,
    model: ResidualCorrectionModel,
    *,
    iteration: int,
    best_score: float,
    best_alpha: float,
    best_bound_abs: float,
    best_bound_rel: float,
    train_log: list[dict[str, object]],
    eval_log: list[dict[str, object]],
    run_config: dict[str, object],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    payload = {
            "model_state_dict": model.state_dict(),
            "corrector_config": asdict(model.corrector.config),
            "run_config": run_config,
            "train_losses": train_log,
            "val_losses": eval_log,
            "iteration": int(iteration),
            "best_iteration": int(iteration),
            "best_score": float(best_score),
            "best_alpha": float(best_alpha),
            "best_bound_abs": float(best_bound_abs),
            "best_bound_rel": float(best_bound_rel),
        }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)



def full_batch_indices(indices: Sequence[int], batch_size: int) -> list[int]:
    usable = (len(indices) // int(batch_size)) * int(batch_size)
    return [int(index) for index in indices[:usable]]


def trajectory_quotas_from_indices(
    dataset: H5WindowDataset,
    indices: Sequence[int],
) -> dict[str, int]:
    quotas = {path.name: 0 for path in dataset.paths}
    for index in indices:
        quotas[dataset.refs[int(index)].path.name] += 1
    return quotas


def summarize_sampling_exposure(
    dataset: H5WindowDataset,
    epoch_plans: Sequence[Sequence[int]],
    *,
    updates: int,
    batch_size: int,
    sampling_policy: str,
) -> dict[str, object]:
    required = int(updates) * int(batch_size)
    consumed: list[int] = []
    for plan in epoch_plans:
        if len(consumed) >= required:
            break
        remaining = required - len(consumed)
        consumed.extend(int(index) for index in plan[:remaining])
    if len(consumed) != required:
        raise RuntimeError(
            f"sampling audit has {len(consumed)} samples but expected {required}"
        )

    legal_by_trajectory: dict[str, set[int]] = {path.name: set() for path in dataset.paths}
    consumed_by_trajectory: dict[str, list[int]] = {path.name: [] for path in dataset.paths}
    for ref in dataset.refs:
        legal_by_trajectory[ref.path.name].add(int(ref.start))
    for index in consumed:
        ref = dataset.refs[int(index)]
        consumed_by_trajectory[ref.path.name].append(int(ref.start))

    duplicate_trajectory_batches = 0
    for offset in range(0, len(consumed), int(batch_size)):
        batch = consumed[offset : offset + int(batch_size)]
        names = [dataset.refs[index].path.name for index in batch]
        if len(set(names)) != len(names):
            duplicate_trajectory_batches += 1

    per_trajectory: list[dict[str, object]] = []
    unique_counts: list[int] = []
    unique_fractions: list[float] = []
    draw_counts: list[int] = []
    for path in dataset.paths:
        name = path.name
        starts = consumed_by_trajectory[name]
        unique = len(set(starts))
        legal = len(legal_by_trajectory[name])
        draw_counts.append(len(starts))
        unique_counts.append(unique)
        fraction = unique / max(legal, 1)
        unique_fractions.append(fraction)
        per_trajectory.append(
            {
                "trajectory": name,
                "draws": len(starts),
                "unique_starts": unique,
                "legal_starts": legal,
                "unique_start_fraction": fraction,
                "repeated_draws": len(starts) - unique,
            }
        )

    spatial_phase_counts = dataset.spatial_phase_counts(consumed)
    return {
        "sampling_policy": sampling_policy,
        "updates": int(updates),
        "batch_size": int(batch_size),
        "samples_consumed": len(consumed),
        "unique_windows_consumed": len(set(consumed)),
        "candidate_legal_windows": len(dataset.refs),
        "unique_window_fraction": len(set(consumed)) / max(len(dataset.refs), 1),
        "duplicate_trajectory_batches": duplicate_trajectory_batches,
        "trajectory_draw_min": min(draw_counts),
        "trajectory_draw_median": float(np.median(draw_counts)),
        "trajectory_draw_max": max(draw_counts),
        "unique_start_min": min(unique_counts),
        "unique_start_median": float(np.median(unique_counts)),
        "unique_start_max": max(unique_counts),
        "unique_start_fraction_median": float(np.median(unique_fractions)),
        "spatial_phase_counts": spatial_phase_counts,
        "spatial_phase_fractions": {
            key: float(value / max(len(consumed), 1))
            for key, value in spatial_phase_counts.items()
        },
        "per_trajectory": per_trajectory,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--allow-train-dev-overlap", action="store_true")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--realpdebench-root", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        choices=("cno", "fno", "unet", "sota_v2_mf"),
        default="cno",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--preload-to-ram", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-delta", type=float, default=0.05)
    parser.add_argument("--drop-pressure-feature", action="store_true")
    parser.add_argument("--history-context", action="store_true")
    parser.add_argument("--include-pressure-data", action="store_true")
    parser.add_argument("--in-steps", type=int, default=20)
    parser.add_argument("--out-steps", type=int, default=20)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument(
        "--eval-stride",
        type=int,
        default=None,
        help=(
            "Validation/evaluation stride. Defaults to --stride for backward compatibility. "
            "Use this only when an experiment changes training-window density while preserving "
            "the historical evaluation protocol."
        ),
    )
    parser.add_argument(
        "--train-window-mode",
        choices=("fixed", "random_phase", "trajectory_stratified_random_start"),
        default="fixed",
    )
    parser.add_argument(
        "--disable-phase-count-equalization",
        action="store_true",
        help=(
            "Preserve every fixed-stride window instead of trimming each trajectory "
            "to the minimum count across phases. Use this to reproduce the original "
            "colleague Stage-2 3341-window stride20 baseline."
        ),
    )
    parser.add_argument(
        "--angle-aug-max-deg",
        type=float,
        default=0.0,
        help=(
            "Training-only global u/v direction perturbation. For each training window, "
            "sample one angle uniformly from [-max,+max] degrees and rotate both Past20 "
            "and Future20 velocity components by the same angle. Validation is never augmented."
        ),
    )
    parser.add_argument(
        "--aoa-meanfield-aug-prob",
        type=float,
        default=0.0,
        help=(
            "Training-only adjacent-AoA mean-field interpolation probability. "
            "Uses same-Re neighboring real trajectories and never exposes AoA/Re to the model."
        ),
    )
    parser.add_argument("--aoa-meanfield-lambda-min", type=float, default=0.2)
    parser.add_argument("--aoa-meanfield-lambda-max", type=float, default=0.5)
    parser.add_argument("--aoa-neighbor-max-gap-deg", type=float, default=5.1)
    parser.add_argument("--aoa-min-eligible-fraction", type=float, default=0.8)
    parser.add_argument("--aoa-bridge-low", type=float, default=None)
    parser.add_argument("--aoa-bridge-high", type=float, default=None)
    parser.add_argument("--sub-sample", type=int, default=2)
    parser.add_argument(
        "--spatial-phase-mix-prob",
        type=float,
        default=0.0,
        help=(
            "Training-only probability mass assigned to non-P00 2x spatial phases. "
            "When >0, each legal temporal window is deterministically assigned P00 "
            "with probability 1-p and a balanced P01/P10/P11 phase with total probability p. "
            "Validation always remains official P00."
        ),
    )
    parser.add_argument(
        "--spatial-phase-seed",
        type=int,
        default=41,
        help="Independent deterministic seed for spatial phase assignment.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--train-on-all", action="store_true")
    parser.add_argument("--max-windows-per-trajectory", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--train-alpha", type=float, default=1.0)
    parser.add_argument("--eval-alphas", default="0,0.025,0.05,0.075,0.1,0.15,0.2,0.3,0.5,0.75,1.0")
    parser.add_argument("--bound-abs", default="0.005,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.025,0.03,0.04")
    parser.add_argument("--bound-rel", default="0,0.0025,0.005,0.0075,0.01,0.0125,0.015,0.02,0.025,0.03,0.04,0.05,0.08,0.1")
    parser.add_argument("--point", type=float, default=1.0)
    parser.add_argument("--mse", type=float, default=0.05)
    parser.add_argument("--tke", type=float, default=0.06)
    parser.add_argument("--temporal", type=float, default=0.03)
    parser.add_argument("--grad", type=float, default=0.015)
    parser.add_argument("--p-zero", type=float, default=0.01)
    parser.add_argument("--residual-mse", type=float, default=0.25)
    parser.add_argument("--delta-penalty", type=float, default=0.02)
    parser.add_argument(
        "--selection-metric",
        choices=("final_est", "point_score"),
        default="final_est",
        help=(
            "Checkpoint selection rule. final_est preserves the historical colleague "
            "behavior including SPS/time. point_score uses only the mean of official-v9 "
            "Rel-L2/TKE/MVPE subscores and is the frozen clean-baseline rule."
        ),
    )
    parser.add_argument(
        "--gradient-mode",
        choices=("scalar", "project_tke"),
        default="scalar",
        help=(
            "scalar preserves the historical objective exactly. project_tke "
            "keeps the primary residual gradient unchanged and projects only "
            "a conflicting weighted TKE gradient."
        ),
    )
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--fixed-time-seconds", type=float, default=None)
    parser.add_argument(
        "--benchmark-mode",
        action="store_true",
        help=(
            "Measure synchronized training-step throughput/VRAM and skip the "
            "expensive full final diagnostics. Intended only for environment "
            "profiling before a frozen experiment."
        ),
    )
    parser.add_argument("--benchmark-warmup", type=int, default=0)
    args = parser.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(f"out_dir already exists, refusing to overwrite: {args.out_dir}")
    if args.angle_aug_max_deg > 0 and args.aoa_meanfield_aug_prob > 0:
        raise ValueError("global velocity rotation and AoA mean-field augmentation are mutually exclusive")
    if not 0.0 <= args.spatial_phase_mix_prob <= 1.0:
        raise ValueError("--spatial-phase-mix-prob must be in [0, 1]")
    if args.spatial_phase_mix_prob > 0.0 and args.sub_sample != 2:
        raise ValueError("--spatial-phase-mix-prob requires --sub-sample 2")
    if (args.aoa_bridge_low is None) != (args.aoa_bridge_high is None):
        raise ValueError("--aoa-bridge-low and --aoa-bridge-high must be set together")
    args.out_dir.mkdir(parents=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.benchmark_mode and device.type != "cuda":
        raise RuntimeError("--benchmark-mode requires CUDA")

    if args.split_manifest is not None:
        train_paths, val_paths = paths_from_split_manifest(
            args.real_root,
            args.split_manifest,
            allow_train_dev_overlap=args.allow_train_dev_overlap,
        )
        paths = train_paths
    else:
        paths = list_h5(args.real_root, BAD_TRAIN_FILES)
        train_paths, val_paths = split_paths(paths, args.val_fraction, args.seed)
    fit_paths = paths if args.train_on_all else train_paths
    base_train_dataset = H5WindowDataset(
        fit_paths,
        in_steps=args.in_steps,
        out_steps=args.out_steps,
        stride=args.stride,
        sub_sample=args.sub_sample,
        max_windows_per_trajectory=args.max_windows_per_trajectory,
        include_pressure=args.include_pressure_data,
        window_mode="random_phase",
        preload_to_ram=args.preload_to_ram,
        spatial_phase_mix_prob=args.spatial_phase_mix_prob,
        spatial_phase_seed=args.spatial_phase_seed,
    )
    aoa_aug_dataset = None
    train_dataset = base_train_dataset
    if args.aoa_meanfield_aug_prob > 0:
        aoa_aug_dataset = AoAMeanFieldShiftDataset(
            base_train_dataset,
            probability=args.aoa_meanfield_aug_prob,
            lambda_min=args.aoa_meanfield_lambda_min,
            lambda_max=args.aoa_meanfield_lambda_max,
            seed=args.seed,
            max_gap_deg=args.aoa_neighbor_max_gap_deg,
            min_eligible_fraction=args.aoa_min_eligible_fraction,
            bridge_pair=(
                (args.aoa_bridge_low, args.aoa_bridge_high)
                if args.aoa_bridge_low is not None
                else None
            ),
        )
        train_dataset = aoa_aug_dataset
        (args.out_dir / "aoa_augmentation_audit.json").write_text(
            json.dumps(aoa_aug_dataset.audit(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    eval_stride = args.stride if args.eval_stride is None else int(args.eval_stride)
    if eval_stride < 1:
        raise ValueError("--eval-stride must be positive")

    val_dataset = H5WindowDataset(
        val_paths,
        in_steps=args.in_steps,
        out_steps=args.out_steps,
        stride=eval_stride,
        sub_sample=args.sub_sample,
        max_windows_per_trajectory=args.max_windows_per_trajectory,
        include_pressure=args.include_pressure_data,
        preload_to_ram=args.preload_to_ram,
    )
    control_reference_sampler = RandomPhaseWindowSampler(
        base_train_dataset,
        seed=args.seed,
        equalize_phase_counts=not args.disable_phase_count_equalization,
    )
    fixed_phases = {path.name: 0 for path in fit_paths}
    train_sampler: RandomPhaseWindowSampler | None = None
    train_batch_sampler: TrajectoryStratifiedRandomStartBatchSampler | None = None

    if args.train_window_mode == "trajectory_stratified_random_start":
        control_reference_sampler.set_epoch(0, phases=fixed_phases)
        reference_indices = full_batch_indices(
            control_reference_sampler.selected_indices,
            args.batch_size,
        )
        train_batch_sampler = TrajectoryStratifiedRandomStartBatchSampler(
            base_train_dataset,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        train_batch_sampler.set_epoch(
            0,
            trajectory_quotas=trajectory_quotas_from_indices(
                base_train_dataset,
                reference_indices,
            ),
        )
        worker_kwargs: dict[str, object] = {}
        if args.workers > 0:
            worker_kwargs = {
                "persistent_workers": True,
                "prefetch_factor": args.prefetch_factor,
            }
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            **worker_kwargs,
        )
        train_samples_per_epoch = len(reference_indices)
    else:
        train_sampler = control_reference_sampler
        train_sampler.set_epoch(
            0,
            phases=fixed_phases if args.train_window_mode == "fixed" else None,
        )
        worker_kwargs = {}
        if args.workers > 0:
            worker_kwargs = {
                "persistent_workers": True,
                "prefetch_factor": args.prefetch_factor,
            }
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
            **worker_kwargs,
        )
        train_samples_per_epoch = (
            len(train_sampler) // args.batch_size
        ) * args.batch_size
    val_worker_kwargs: dict[str, object] = {}
    if args.workers > 0:
        val_worker_kwargs = {
            "persistent_workers": True,
            "prefetch_factor": args.prefetch_factor,
        }
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        **val_worker_kwargs,
    )

    base_model = load_frozen_base(args.base_model, args.checkpoint, args.realpdebench_root, device)
    corrector_config = CorrectorConfig(
        hidden=args.hidden,
        blocks=args.blocks,
        dropout=args.dropout,
        include_pressure=not args.drop_pressure_feature,
        max_delta=args.max_delta,
        history_context=args.history_context,
    )
    model = ResidualCorrectionModel(base_model, ResidualCorrector3D(corrector_config)).to(device)
    resume_metadata: dict[str, object] = {}
    if args.resume_checkpoint is not None:
        resume_metadata = load_residual_checkpoint(model, args.resume_checkpoint, device)
    optimizer = torch.optim.AdamW(model.corrector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.updates))

    weights = {
        "point": args.point,
        "mse": args.mse,
        "tke": args.tke,
        "temporal": args.temporal,
        "grad": args.grad,
        "p_zero": args.p_zero,
    }
    alphas = parse_float_list(args.eval_alphas)
    abs_widths = parse_float_list(args.bound_abs)
    rel_widths = parse_float_list(args.bound_rel)
    if args.selection_metric == "point_score" and len(alphas) != 1:
        raise ValueError(
            "--selection-metric point_score requires exactly one --eval-alphas value "
            "so correction alpha cannot be tuned on the dev set"
        )

    run_config = {
        "real_root": str(args.real_root),
        "split_manifest": str(args.split_manifest) if args.split_manifest else None,
        "allow_train_dev_overlap": bool(args.allow_train_dev_overlap),
        "checkpoint": str(args.checkpoint),
        "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
        "resume_iteration": int(resume_metadata.get("iteration", 0)),
        "optimizer_policy": "reset_identically_for_each_screen_arm",
        "realpdebench_root": str(args.realpdebench_root),
        "device": str(device),
        "train_trajectories": len(train_paths),
        "fit_trajectories": len(fit_paths),
        "val_trajectories": len(val_paths),
        "train_windows": (
            len(base_train_dataset)
            if train_batch_sampler is not None
            else len(train_sampler)  # type: ignore[arg-type]
        ),
        "candidate_legal_windows": len(base_train_dataset),
        "train_samples_per_epoch": int(train_samples_per_epoch),
        "reference_fixed_stride_windows": len(control_reference_sampler),
        "val_windows": len(val_dataset),
        "preload_to_ram": bool(args.preload_to_ram),
        "train_ram_cache": base_train_dataset.cache_summary(),
        "val_ram_cache": val_dataset.cache_summary(),
        "workers": args.workers,
        "persistent_workers": bool(args.workers > 0),
        "prefetch_factor": args.prefetch_factor if args.workers > 0 else None,
        "train_stride": int(args.stride),
        "eval_stride": int(eval_stride),
        "train_window_mode": args.train_window_mode,
        "sampling_policy": (
            "baseline_fixed_stride_global_shuffle"
            if args.train_window_mode == "fixed"
            else (
                "epoch_random_phase_global_shuffle"
                if args.train_window_mode == "random_phase"
                else "baseline_weight_matched_trajectory_stratified_random_start"
            )
        ),
        "batch_unique_trajectory_required": bool(
            args.train_window_mode == "trajectory_stratified_random_start"
        ),
        "angle_augmentation": {
            "kind": "global_uv_component_rotation",
            "max_abs_degrees": float(args.angle_aug_max_deg),
            "training_only": True,
            "same_angle_for_past_and_future": True,
            "spatial_grid_rotated": False,
            "physical_exact_aoa_claimed": False,
        },
        "spatial_phase_augmentation": {
            "kind": "2x_subsample_phase_mix",
            "non_p00_probability": float(args.spatial_phase_mix_prob),
            "p00_probability": float(1.0 - args.spatial_phase_mix_prob),
            "alternate_phases": ["P01", "P10", "P11"],
            "spatial_phase_seed": int(args.spatial_phase_seed),
            "training_only": True,
            "same_phase_for_past_and_future": True,
            "validation_phase": "P00",
            "changes_temporal_window_sampling": False,
        },
        "aoa_meanfield_augmentation": {
            "kind": "same_re_adjacent_aoa_past20_mean_field_shift",
            "probability": float(args.aoa_meanfield_aug_prob),
            "lambda_min": float(args.aoa_meanfield_lambda_min),
            "lambda_max": float(args.aoa_meanfield_lambda_max),
            "max_neighbor_gap_deg": float(args.aoa_neighbor_max_gap_deg),
            "min_eligible_fraction": float(args.aoa_min_eligible_fraction),
            "bridge_pair": (
                [float(args.aoa_bridge_low), float(args.aoa_bridge_high)]
                if args.aoa_bridge_low is not None
                else None
            ),
            "training_only": True,
            "aoa_re_model_inputs": False,
            "future_used_to_construct_input_shift": False,
            "audit": aoa_aug_dataset.audit() if aoa_aug_dataset is not None else None,
        },
        "phase_counts_equalized": not args.disable_phase_count_equalization,
        "updates": args.updates,
        "eval_interval": args.eval_interval,
        "batch_size": args.batch_size,
        "test_batch_size": args.test_batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "corrector_config": asdict(corrector_config),
        "train_alpha": args.train_alpha,
        "eval_alphas": alphas,
        "abs_widths": abs_widths,
        "rel_widths": rel_widths,
        "loss_weights": weights,
        "selection_metric": args.selection_metric,
        "sps_used_for_checkpoint_selection": args.selection_metric == "final_est",
        "runtime_used_for_checkpoint_selection": args.selection_metric == "final_est",
        "gradient_mode": args.gradient_mode,
        "gradient_projection": (
            "project weighted TKE gradient orthogonal to primary residual gradient "
            "only when their dot product is negative"
            if args.gradient_mode == "project_tke"
            else "none"
        ),
        "residual_mse": args.residual_mse,
        "delta_penalty": args.delta_penalty,
        "fixed_time_seconds": args.fixed_time_seconds,
        "benchmark_mode": bool(args.benchmark_mode),
        "trainable_parameters": sum(p.numel() for p in model.corrector.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")
    print(json.dumps(run_config, indent=2, default=str), flush=True)

    train_log: list[dict[str, object]] = []
    eval_log: list[dict[str, object]] = []

    summaries = evaluate_alphas(
        model,
        val_loader,
        device,
        alphas=alphas,
        abs_widths=abs_widths,
        rel_widths=rel_widths,
        max_batches=args.max_eval_batches,
        fixed_time_seconds=args.fixed_time_seconds,
    )
    top = summaries[0]
    top["iteration"] = 0
    eval_log.append({"iteration": 0, "summaries": summaries[:5]})
    best_score = checkpoint_selection_score(top, args.selection_metric)
    best_iter = 0
    best_alpha = float(top["alpha"])
    best_bound_abs = float(top["best_bound_abs"])
    best_bound_rel = float(top["best_bound_rel"])
    print("EVAL_TOP " + json.dumps(top, sort_keys=True), flush=True)
    (args.out_dir / "eval_step_00000.json").write_text(
        json.dumps(summaries, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    if args.selection_metric == "point_score" and not args.benchmark_mode:
        write_horizon_snapshot(
            model,
            val_loader,
            device,
            out_dir=args.out_dir / "milestone_00000",
            experiment=f"{args.out_dir.name}_step0",
            alpha=float(top["alpha"]),
        )
    save_checkpoint(
        args.out_dir / "model_best.pth",
        model,
        iteration=0,
        best_score=best_score,
        best_alpha=best_alpha,
        best_bound_abs=best_bound_abs,
        best_bound_rel=best_bound_rel,
        train_log=train_log,
        eval_log=eval_log,
        run_config=run_config,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    # Preserve the zero-residual initialization explicitly.  This represents the
    # arm-specific CNO backbone before residual learning and is useful for clean
    # stage attribution in held-out generalization benchmarks.
    save_checkpoint(
        args.out_dir / "model_init.pth",
        model,
        iteration=0,
        best_score=best_score,
        best_alpha=best_alpha,
        best_bound_abs=best_bound_abs,
        best_bound_rel=best_bound_rel,
        train_log=train_log,
        eval_log=eval_log,
        run_config=run_config,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    train_iter = iter(train_loader)
    sampler_epoch = 0
    sampling_epoch_plans: list[list[int]] = []
    benchmark_step_seconds = 0.0
    benchmark_measured_updates = 0
    if args.benchmark_mode and device.type == "cuda":
        if args.benchmark_warmup < 0 or args.benchmark_warmup >= args.updates:
            raise ValueError("--benchmark-warmup must be in [0, updates)")
        torch.cuda.reset_peak_memory_stats(device)
    audit_path = args.out_dir / "window_audit.jsonl"

    def current_sampling_plan() -> list[int]:
        if train_batch_sampler is not None:
            return [
                int(index)
                for batch_indices in train_batch_sampler.current_plan
                for index in batch_indices
            ]
        assert train_sampler is not None
        return full_batch_indices(train_sampler.selected_indices, args.batch_size)

    def current_sampling_epoch_record() -> dict[str, object]:
        if train_batch_sampler is not None:
            return {
                "epoch": sampler_epoch,
                "policy": "trajectory_stratified_random_start",
                "trajectory_quotas": train_batch_sampler.trajectory_quotas,
                "batches": len(train_batch_sampler),
            }
        assert train_sampler is not None
        return {
            "epoch": sampler_epoch,
            "policy": args.train_window_mode,
            "phases": train_sampler.phases,
            "samples": len(current_sampling_plan()),
        }

    sampling_epoch_plans.append(current_sampling_plan())
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(current_sampling_epoch_record(), sort_keys=True) + "\n")
    accum: dict[str, list[float]] = {}
    angle_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 20260922]))
    for step in range(1, args.updates + 1):
        model.corrector.train()
        model.base_model.eval()
        if args.benchmark_mode and device.type == "cuda":
            torch.cuda.synchronize(device)
        benchmark_step_started = time.perf_counter()
        try:
            batch = next(train_iter)
        except StopIteration:
            sampler_epoch += 1
            if train_batch_sampler is not None:
                control_reference_sampler.set_epoch(
                    sampler_epoch,
                    phases=fixed_phases,
                )
                reference_indices = full_batch_indices(
                    control_reference_sampler.selected_indices,
                    args.batch_size,
                )
                train_batch_sampler.set_epoch(
                    sampler_epoch,
                    trajectory_quotas=trajectory_quotas_from_indices(
                        base_train_dataset,
                        reference_indices,
                    ),
                )
            else:
                assert train_sampler is not None
                train_sampler.set_epoch(
                    sampler_epoch,
                    phases=fixed_phases if args.train_window_mode == "fixed" else None,
                )
            if aoa_aug_dataset is not None:
                aoa_aug_dataset.set_epoch(sampler_epoch)
            sampling_epoch_plans.append(current_sampling_plan())
            with audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(current_sampling_epoch_record(), sort_keys=True) + "\n")
            train_iter = iter(train_loader)
            batch = next(train_iter)
        if len(batch) == 3:
            x, y, aoa_meta = batch
        else:
            x, y = batch
            aoa_meta = None
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        angle_abs_mean = 0.0
        if args.angle_aug_max_deg < 0:
            raise ValueError("--angle-aug-max-deg must be non-negative")
        if args.angle_aug_max_deg > 0:
            sampled_degrees = angle_rng.uniform(
                -float(args.angle_aug_max_deg),
                float(args.angle_aug_max_deg),
                size=int(x.shape[0]),
            ).astype(np.float32)
            angle_abs_mean = float(np.mean(np.abs(sampled_degrees)))
            sampled_degrees_t = torch.from_numpy(sampled_degrees).to(device=device, dtype=x.dtype)
            x = rotate_velocity_uv(x, sampled_degrees_t)
            y = rotate_velocity_uv(y, sampled_degrees_t)

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            base = model.base_predict(x)
        delta = model.predict_delta(x, base)
        pred = model.combine(base, delta, args.train_alpha)
        if args.gradient_mode == "project_tke":
            primary_loss, energy_loss, tensor_parts = split_residual_objective(
                pred,
                y,
                base,
                delta,
                weights=weights,
                residual_mse_weight=args.residual_mse,
                delta_penalty_weight=args.delta_penalty,
            )
            loss = primary_loss + energy_loss
            parts = {
                key: float(value.detach().cpu())
                for key, value in tensor_parts.items()
            }
            parts["primary_loss"] = float(primary_loss.detach().cpu())
            parts["energy_loss"] = float(energy_loss.detach().cpu())
            parts.update(
                project_tke_backward(
                    primary_loss,
                    energy_loss,
                    model.corrector.parameters(),
                )
            )
        else:
            loss, parts = physics_loss(pred, y, weights)
            residual_target = y[..., :2] - base[..., :2]
            residual_mse = torch.mean((delta[..., :2] - residual_target) ** 2)
            delta_penalty = torch.mean(delta[..., :2] ** 2)
            loss = (
                loss
                + args.residual_mse * residual_mse
                + args.delta_penalty * delta_penalty
            )
            parts["residual_mse"] = float(residual_mse.detach().cpu())
            parts["delta_penalty"] = float(delta_penalty.detach().cpu())
        parts["angle_aug_abs_deg"] = angle_abs_mean
        if aoa_meta is not None:
            applied = aoa_meta["applied"].float()
            parts["aoa_meanfield_aug_fraction"] = float(applied.mean().item())
            parts["aoa_meanfield_lambda_mean"] = float(aoa_meta["lambda"].float().mean().item())
            parts["aoa_effective_shift_abs_deg_mean"] = float(
                aoa_meta["effective_shift_deg"].float().abs().mean().item()
            )
        else:
            parts["aoa_meanfield_aug_fraction"] = 0.0
            parts["aoa_meanfield_lambda_mean"] = 0.0
            parts["aoa_effective_shift_abs_deg_mean"] = 0.0
        parts["loss"] = float(loss.detach().cpu())
        if args.gradient_mode == "scalar":
            loss.backward()
        if args.clip_grad and args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.corrector.parameters(), args.clip_grad)
        optimizer.step()
        scheduler.step()
        if args.benchmark_mode and device.type == "cuda":
            torch.cuda.synchronize(device)
            if step > args.benchmark_warmup:
                benchmark_step_seconds += time.perf_counter() - benchmark_step_started
                benchmark_measured_updates += 1

        for key, value in parts.items():
            accum.setdefault(key, []).append(float(value))
        if step % 20 == 0:
            row = {
                "iteration": step,
                "lr": optimizer.param_groups[0]["lr"],
                **{key: float(np.mean(values[-20:])) for key, values in accum.items()},
            }
            train_log.append(row)
            print("TRAIN " + json.dumps(row, sort_keys=True), flush=True)

        if step % args.eval_interval == 0 or step == args.updates:
            summaries = evaluate_alphas(
                model,
                val_loader,
                device,
                alphas=alphas,
                abs_widths=abs_widths,
                rel_widths=rel_widths,
                max_batches=args.max_eval_batches,
                fixed_time_seconds=args.fixed_time_seconds,
            )
            top = summaries[0]
            top["iteration"] = step
            eval_log.append({"iteration": step, "summaries": summaries[:5]})
            current_score = checkpoint_selection_score(top, args.selection_metric)
            print("EVAL_TOP " + json.dumps({
                **top,
                "selection_metric": args.selection_metric,
                "selection_score": current_score,
            }, sort_keys=True), flush=True)
            (args.out_dir / f"eval_step_{step:05d}.json").write_text(
                json.dumps(summaries, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            if args.selection_metric == "point_score" and not args.benchmark_mode:
                write_horizon_snapshot(
                    model,
                    val_loader,
                    device,
                    out_dir=args.out_dir / f"milestone_{step:05d}",
                    experiment=f"{args.out_dir.name}_step{step}",
                    alpha=float(top["alpha"]),
                )
            save_checkpoint(
                args.out_dir / "model_latest.pth",
                model,
                iteration=step,
                best_score=current_score,
                best_alpha=float(top["alpha"]),
                best_bound_abs=float(top["best_bound_abs"]),
                best_bound_rel=float(top["best_bound_rel"]),
                train_log=train_log,
                eval_log=eval_log,
                run_config=run_config,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            if current_score > best_score:
                best_score = current_score
                best_iter = step
                best_alpha = float(top["alpha"])
                best_bound_abs = float(top["best_bound_abs"])
                best_bound_rel = float(top["best_bound_rel"])
                save_checkpoint(
                    args.out_dir / "model_best.pth",
                    model,
                    iteration=step,
                    best_score=best_score,
                    best_alpha=best_alpha,
                    best_bound_abs=best_bound_abs,
                    best_bound_rel=best_bound_rel,
                    train_log=train_log,
                    eval_log=eval_log,
                    run_config=run_config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
                print(
                    f"BEST iteration={best_iter} {args.selection_metric}={best_score:.6f} "
                    f"alpha={best_alpha} abs={best_bound_abs} rel={best_bound_rel}",
                    flush=True,
                )
            (args.out_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "best_iter": best_iter,
                        "best_score": best_score,
                        "selection_metric": args.selection_metric,
                        "best_alpha": best_alpha,
                        "best_bound_abs": best_bound_abs,
                        "best_bound_rel": best_bound_rel,
                        "latest_top": top,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

    save_checkpoint(
        args.out_dir / "model_final.pth",
        model,
        iteration=args.updates,
        best_score=best_score,
        best_alpha=best_alpha,
        best_bound_abs=best_bound_abs,
        best_bound_rel=best_bound_rel,
        train_log=train_log,
        eval_log=eval_log,
        run_config=run_config,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    sampling_audit = summarize_sampling_exposure(
        base_train_dataset,
        sampling_epoch_plans,
        updates=args.updates,
        batch_size=args.batch_size,
        sampling_policy=str(run_config["sampling_policy"]),
    )
    (args.out_dir / "sampling_audit.json").write_text(
        json.dumps(sampling_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.benchmark_mode:
        total_memory = int(torch.cuda.get_device_properties(device).total_memory)
        runtime = {
            "benchmark_mode": True,
            "updates": int(args.updates),
            "warmup_updates": int(args.benchmark_warmup),
            "measured_updates": int(benchmark_measured_updates),
            "batch_size": int(args.batch_size),
            "samples_seen": int(args.updates * args.batch_size),
            "training_step_seconds": float(benchmark_step_seconds),
            "samples_per_second": float(
                benchmark_measured_updates * args.batch_size / max(benchmark_step_seconds, 1e-12)
            ),
            "train_ram_cache": base_train_dataset.cache_summary(),
            "val_ram_cache": val_dataset.cache_summary(),
            "workers": args.workers,
            "prefetch_factor": args.prefetch_factor if args.workers > 0 else None,
            "peak_gpu_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
            "peak_gpu_memory_reserved": int(torch.cuda.max_memory_reserved(device)),
            "gpu_total_memory": total_memory,
            "peak_allocated_fraction": float(
                torch.cuda.max_memory_allocated(device) / total_memory
            ),
            "peak_reserved_fraction": float(
                torch.cuda.max_memory_reserved(device) / total_memory
            ),
        }
        (args.out_dir / "runtime.json").write_text(
            json.dumps(runtime, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        final_evidence = {
            "benchmark_mode": True,
            "runtime": runtime,
        }
    else:
        final_evidence = write_final_evidence(
            model,
            val_loader,
            device,
            out_dir=args.out_dir,
            experiment=args.out_dir.name,
            alpha=1.0,
        )
    (args.out_dir / "DONE").touch()
    print(
        f"DONE out_dir={args.out_dir} best_iter={best_iter} "
        f"best_score={best_score:.6f} alpha={best_alpha} abs={best_bound_abs} "
        f"rel={best_bound_rel} primary={json.dumps(final_evidence, sort_keys=True)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
