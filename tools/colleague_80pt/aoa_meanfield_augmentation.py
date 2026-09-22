#!/usr/bin/env python3
"""Training-only adjacent-AoA mean-field interpolation for RealPDE Track 1.

The augmentation changes the *mean spatial flow field* toward a real neighboring
angle-of-attack trajectory at the same Reynolds number, while preserving the
anchor window's temporal fluctuation sequence.

For anchor A and same-Re neighboring-AoA trajectory B:

    delta_mean = mean_t(Past20_B) - mean_t(Past20_A)
    X_aug = X_A + lambda * delta_mean
    Y_aug = Y_A + lambda * delta_mean

Only Past20 is used to estimate the mean-field shift.  The exact same spatial
shift is applied to Past20 and Future20, so no Future20 information is used to
construct the input transformation.  AoA/Re metadata are training-time only and
are never model inputs or inference-time features.

This is intentionally more conservative than raw frame-wise Mixup: arbitrary
vortex/shedding phases from two trajectories are not averaged together.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from realpde_h5_feature_adapter_train import H5WindowDataset, WindowRef, h5_field


@dataclass(frozen=True)
class FlowCondition:
    re: float
    aoa: float


def _scalar_field(handle: h5py.File, key: str) -> float:
    value = np.asarray(h5_field(handle, key))
    if value.size < 1:
        raise ValueError(f"empty HDF5 metadata field: {key}")
    flat = value.reshape(-1)
    if not np.allclose(flat, flat[0], rtol=0.0, atol=1e-6):
        raise ValueError(f"HDF5 metadata field {key!r} is not scalar/constant")
    return float(flat[0])


def read_condition(path: Path) -> FlowCondition:
    with h5py.File(path, "r") as handle:
        return FlowCondition(re=_scalar_field(handle, "re"), aoa=_scalar_field(handle, "aoa"))


def read_grid(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    with h5py.File(path, "r") as handle:
        try:
            x = np.asarray(h5_field(handle, "x"), dtype=np.float64)
            y = np.asarray(h5_field(handle, "y"), dtype=np.float64)
        except KeyError:
            return None
    return x, y


def grids_compatible(
    first: tuple[np.ndarray, np.ndarray] | None,
    second: tuple[np.ndarray, np.ndarray] | None,
) -> bool:
    if first is None or second is None:
        return True
    ax, ay = first
    bx, by = second
    return (
        ax.shape == bx.shape
        and ay.shape == by.shape
        and np.allclose(ax, bx, rtol=0.0, atol=1e-7)
        and np.allclose(ay, by, rtol=0.0, atol=1e-7)
    )


def _same_re(a: float, b: float) -> bool:
    return bool(np.isclose(a, b, rtol=0.0, atol=1e-5))


def build_adjacent_aoa_neighbors(
    paths: Sequence[Path],
    *,
    max_gap_deg: float = 5.1,
) -> tuple[dict[Path, tuple[Path, ...]], dict[Path, FlowCondition]]:
    """Map each trajectory to nearest same-Re, different-AoA neighbor(s)."""
    if max_gap_deg <= 0:
        raise ValueError("max_gap_deg must be positive")
    conditions = {path: read_condition(path) for path in paths}
    grids = {path: read_grid(path) for path in paths}
    mapping: dict[Path, tuple[Path, ...]] = {}
    for path in paths:
        source = conditions[path]
        candidates = [
            other
            for other in paths
            if other != path
            and _same_re(conditions[other].re, source.re)
            and abs(conditions[other].aoa - source.aoa) > 1e-6
            and grids_compatible(grids[path], grids[other])
        ]
        if not candidates:
            mapping[path] = ()
            continue
        nearest_gap = min(abs(conditions[p].aoa - source.aoa) for p in candidates)
        if nearest_gap > max_gap_deg + 1e-6:
            mapping[path] = ()
            continue
        nearest = tuple(
            sorted(
                (
                    p
                    for p in candidates
                    if abs(abs(conditions[p].aoa - source.aoa) - nearest_gap) <= 1e-6
                ),
                key=lambda p: p.name,
            )
        )
        mapping[path] = nearest
    return mapping, conditions


def load_window_like(dataset: H5WindowDataset, ref: WindowRef) -> tuple[Tensor, Tensor]:
    total = dataset.in_steps + dataset.out_steps
    with h5py.File(ref.path, "r") as handle:
        sl = slice(ref.start, ref.start + total)
        ss = dataset.sub_sample
        u = np.asarray(h5_field(handle, "u")[sl, ::ss, ::ss], dtype=np.float32)
        v = np.asarray(h5_field(handle, "v")[sl, ::ss, ::ss], dtype=np.float32)
        if dataset.include_pressure:
            try:
                p = np.asarray(h5_field(handle, "p")[sl, ::ss, ::ss], dtype=np.float32)
            except KeyError:
                p = np.zeros_like(u)
        else:
            p = np.zeros_like(u)
    full = torch.from_numpy(np.stack([u, v, p], axis=-1))
    return full[: dataset.in_steps], full[dataset.in_steps :]


class AoAMeanFieldShiftDataset(Dataset):
    """Wrap H5WindowDataset with deterministic adjacent-AoA mean-field shifts."""

    def __init__(
        self,
        base: H5WindowDataset,
        *,
        probability: float,
        lambda_min: float,
        lambda_max: float,
        seed: int,
        max_gap_deg: float = 5.1,
        min_eligible_fraction: float = 0.8,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0,1]")
        if not 0.0 <= lambda_min <= lambda_max <= 0.5:
            raise ValueError("require 0 <= lambda_min <= lambda_max <= 0.5")
        if not 0.0 <= min_eligible_fraction <= 1.0:
            raise ValueError("min_eligible_fraction must be in [0,1]")
        self.base = base
        self.probability = float(probability)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.seed = int(seed)
        self.max_gap_deg = float(max_gap_deg)
        self.epoch = 0
        self.neighbors, self.conditions = build_adjacent_aoa_neighbors(
            self.base.paths,
            max_gap_deg=self.max_gap_deg,
        )
        eligible = sum(bool(self.neighbors[path]) for path in self.base.paths)
        self.eligible_fraction = eligible / max(len(self.base.paths), 1)
        if self.probability > 0 and self.eligible_fraction < min_eligible_fraction:
            raise RuntimeError(
                "adjacent-AoA coverage too low for a valid augmentation experiment: "
                f"{eligible}/{len(self.base.paths)}={self.eligible_fraction:.3f} "
                f"< required {min_eligible_fraction:.3f}"
            )

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def audit(self) -> dict[str, object]:
        rows = []
        for path in self.base.paths:
            source = self.conditions[path]
            rows.append({
                "trajectory": path.name,
                "re": source.re,
                "aoa": source.aoa,
                "neighbor_trajectories": [p.name for p in self.neighbors[path]],
                "neighbor_aoa": [self.conditions[p].aoa for p in self.neighbors[path]],
                "eligible": bool(self.neighbors[path]),
                "coordinate_grid_checked": read_grid(path) is not None,
            })
        return {
            "kind": "adjacent_aoa_mean_field_shift",
            "trajectories": len(self.base.paths),
            "eligible_trajectories": sum(row["eligible"] for row in rows),
            "eligible_fraction": self.eligible_fraction,
            "max_neighbor_gap_deg": self.max_gap_deg,
            "probability": self.probability,
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
            "same_re_required": True,
            "metadata_used_at_inference": False,
            "rows": rows,
        }

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, int(index), 20260922])
        )

    def __getitem__(self, index: int):
        x, y = self.base[index]
        ref = self.base.refs[index]
        source_condition = self.conditions[ref.path]
        meta = {
            "applied": torch.tensor(0.0, dtype=torch.float32),
            "lambda": torch.tensor(0.0, dtype=torch.float32),
            "aoa_from": torch.tensor(source_condition.aoa, dtype=torch.float32),
            "aoa_to": torch.tensor(source_condition.aoa, dtype=torch.float32),
            "effective_shift_deg": torch.tensor(0.0, dtype=torch.float32),
        }
        candidates = self.neighbors[ref.path]
        if self.probability <= 0 or not candidates:
            return x, y, meta

        rng = self._rng(index)
        if float(rng.random()) >= self.probability:
            return x, y, meta

        neighbor_path = candidates[int(rng.integers(0, len(candidates)))]
        neighbor_condition = self.conditions[neighbor_path]
        total = self.base.in_steps + self.base.out_steps
        max_start = self.base.lengths[neighbor_path] - total
        if max_start < 0:
            return x, y, meta
        neighbor_start = min(ref.start, max_start)
        neighbor_x, _ = load_window_like(
            self.base,
            WindowRef(path=neighbor_path, start=neighbor_start),
        )

        lam = float(rng.uniform(self.lambda_min, self.lambda_max))
        # Use only Past20 to estimate the AoA-related mean-field change.
        delta_mean_uv = (
            neighbor_x[..., :2].mean(dim=0, keepdim=True)
            - x[..., :2].mean(dim=0, keepdim=True)
        )
        x_aug = x.clone()
        y_aug = y.clone()
        x_aug[..., :2] = x_aug[..., :2] + lam * delta_mean_uv
        y_aug[..., :2] = y_aug[..., :2] + lam * delta_mean_uv
        if x_aug.shape[-1] >= 3:
            x_aug[..., 2] = 0.0
            y_aug[..., 2] = 0.0

        shift = lam * (neighbor_condition.aoa - source_condition.aoa)
        meta = {
            "applied": torch.tensor(1.0, dtype=torch.float32),
            "lambda": torch.tensor(lam, dtype=torch.float32),
            "aoa_from": torch.tensor(source_condition.aoa, dtype=torch.float32),
            "aoa_to": torch.tensor(neighbor_condition.aoa, dtype=torch.float32),
            "effective_shift_deg": torch.tensor(shift, dtype=torch.float32),
        }
        return x_aug, y_aug, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-gap-deg", type=float, default=5.1)
    args = parser.parse_args()

    paths = [args.real_root / name for name in args.files]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[:10])
    mapping, conditions = build_adjacent_aoa_neighbors(paths, max_gap_deg=args.max_gap_deg)
    rows = []
    for path in paths:
        rows.append({
            "trajectory": path.name,
            "re": conditions[path].re,
            "aoa": conditions[path].aoa,
            "neighbors": [p.name for p in mapping[path]],
            "neighbor_aoa": [conditions[p].aoa for p in mapping[path]],
            "eligible": bool(mapping[path]),
        })
    payload = {
        "trajectories": len(paths),
        "eligible_trajectories": sum(row["eligible"] for row in rows),
        "eligible_fraction": sum(row["eligible"] for row in rows) / max(len(rows), 1),
        "max_gap_deg": args.max_gap_deg,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
