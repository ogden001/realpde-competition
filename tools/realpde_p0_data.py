"""Official-HDF5 window loader used by the P0 feature experiment.

This intentionally reads the released top-level HDF5 schema rather than the
reference RealPDEBench loader, whose historical ``measured_data`` layout does
not match the competition archives in this workspace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class WindowRef:
    path: Path
    start: int


def list_h5(root: Path) -> list[Path]:
    paths = sorted(root.glob("*.h5"))
    if not paths:
        raise ValueError(f"no HDF5 trajectories found beneath {root}")
    return paths


def split_paths(paths: Sequence[Path], val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    """Trajectory-level deterministic split; never splits a trajectory by time."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    if len(paths) < 2:
        raise ValueError("at least two trajectories are required for a trajectory split")
    rng = np.random.default_rng(seed)
    order = np.arange(len(paths)); rng.shuffle(order)
    n_val = min(max(1, round(len(paths) * val_fraction)), len(paths) - 1)
    val_ids = set(order[:n_val].tolist())
    return [p for i, p in enumerate(paths) if i not in val_ids], [p for i, p in enumerate(paths) if i in val_ids]


def read_grid(path: Path, sub_sample: int) -> tuple[Tensor, Tensor]:
    with h5py.File(path, "r") as f:
        return (
            torch.as_tensor(f["x"][::sub_sample, ::sub_sample], dtype=torch.float32),
            torch.as_tensor(f["y"][::sub_sample, ::sub_sample], dtype=torch.float32),
        )


class H5WindowDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Reads paired input/target windows without materializing derived features."""

    def __init__(self, paths: Sequence[Path], *, in_steps: int = 20, out_steps: int = 20, stride: int = 20,
                 sub_sample: int = 2, max_windows_per_trajectory: int | None = None, include_pressure: bool = False,
                 window_mode: str = "fixed"):
        if min(in_steps, out_steps, stride, sub_sample) < 1:
            raise ValueError("in_steps, out_steps, stride, and sub_sample must be positive")
        if window_mode not in {"fixed", "random_phase"}:
            raise ValueError("window_mode must be 'fixed' or 'random_phase'")
        self.in_steps, self.out_steps, self.sub_sample = in_steps, out_steps, sub_sample
        self.stride = stride
        self.window_mode = window_mode
        self.max_windows_per_trajectory = max_windows_per_trajectory
        self.include_pressure = include_pressure
        self.refs: list[WindowRef] = []
        self.paths = list(paths)
        self.lengths: dict[Path, int] = {}
        for path in self.paths:
            with h5py.File(path, "r") as f:
                field = f["u"] if "u" in f else f["measured_data/u"]
                length = int(field.shape[0])
            self.lengths[path] = length
            if window_mode == "fixed":
                starts = list(range(0, length - in_steps - out_steps + 1, stride))
                if max_windows_per_trajectory is not None:
                    starts = starts[:max_windows_per_trajectory]
            else:
                # Random-phase selection happens in the main-process sampler.
                # Keep every legal start available as a lightweight WindowRef so
                # workers never need synchronized mutable Dataset state.
                starts = list(range(0, length - in_steps - out_steps + 1))
            self.refs.extend(WindowRef(path, start) for start in starts)
        if not self.refs:
            raise ValueError("the requested windows do not fit into supplied trajectories")

    def __len__(self) -> int:
        return len(self.refs)

    @staticmethod
    def _field(f: h5py.File, key: str):
        return f[key] if key in f else f[f"measured_data/{key}"]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ref = self.refs[index]
        total = self.in_steps + self.out_steps
        with h5py.File(ref.path, "r") as f:
            sl = slice(ref.start, ref.start + total)
            u = np.asarray(self._field(f, "u")[sl, ::self.sub_sample, ::self.sub_sample], dtype=np.float32)
            v = np.asarray(self._field(f, "v")[sl, ::self.sub_sample, ::self.sub_sample], dtype=np.float32)
            if self.include_pressure and ("p" in f or "measured_data/p" in f):
                p = np.asarray(self._field(f, "p")[sl, ::self.sub_sample, ::self.sub_sample], dtype=np.float32)
            else:
                p = np.zeros_like(u)
            re, aoa = float(f["re"][()]), float(f["aoa"][()])
        full = torch.from_numpy(np.stack([u, v, p], axis=-1))
        condition = torch.tensor([re, aoa], dtype=torch.float32)
        return full[:self.in_steps], full[self.in_steps:], condition, torch.tensor(index, dtype=torch.long)


class RandomPhaseWindowSampler(Sampler[int]):
    """Select one independent phase per trajectory for a deterministic epoch."""

    def __init__(self, dataset: H5WindowDataset, *, seed: int):
        if dataset.window_mode != "random_phase":
            raise ValueError("RandomPhaseWindowSampler requires window_mode='random_phase'")
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        self.phases: dict[str, int] = {}
        self._selected_indices: list[int] = []
        self._index_by_path_start = {(str(ref.path), ref.start): index for index, ref in enumerate(dataset.refs)}
        self.set_epoch(0)

    def set_epoch(self, epoch: int, *, phases: dict[str, int] | None = None) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)
        if phases is None:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
            self.phases = {path.name: int(rng.integers(0, self.dataset.stride)) for path in self.dataset.paths}
        else:
            expected = {path.name for path in self.dataset.paths}
            if set(phases) != expected:
                raise ValueError("phases must contain exactly one phase for each trajectory")
            self.phases = {name: int(phase) for name, phase in phases.items()}
            if any(phase < 0 or phase >= self.dataset.stride for phase in self.phases.values()):
                raise ValueError("each phase must be in [0, stride)")

        selected: list[int] = []
        for path in self.dataset.paths:
            phase = self.phases[path.name]
            starts = list(range(phase, self.dataset.lengths[path] - self.dataset.in_steps - self.dataset.out_steps + 1,
                               self.dataset.stride))
            if self.dataset.max_windows_per_trajectory is not None:
                starts = starts[:self.dataset.max_windows_per_trajectory]
            selected.extend(self._index_by_path_start[(str(path), start)] for start in starts)
        # Preserve independent per-trajectory phase selection above, then
        # match the fixed-window train loader's global shuffled presentation.
        # A separate epoch-derived stream keeps phase draws unchanged.
        shuffle_rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 1]))
        shuffle_rng.shuffle(selected)
        self._selected_indices = selected

    def audit_records(self) -> list[dict[str, object]]:
        records = []
        for path in self.dataset.paths:
            phase = self.phases[path.name]
            starts = list(range(phase, self.dataset.lengths[path] - self.dataset.in_steps - self.dataset.out_steps + 1,
                               self.dataset.stride))
            if self.dataset.max_windows_per_trajectory is not None:
                starts = starts[:self.dataset.max_windows_per_trajectory]
            strides = [right - left for left, right in zip(starts, starts[1:])]
            records.append({
                "epoch": self.epoch,
                "trajectory": path.name,
                "phase": phase,
                "window_count": len(starts),
                "first_start": starts[0] if starts else None,
                "last_start": starts[-1] if starts else None,
                "min_stride": min(strides) if strides else self.dataset.stride,
                "max_stride": max(strides) if strides else self.dataset.stride,
                "invalid_window_count": sum(
                    not (0 <= start and start + self.dataset.in_steps + self.dataset.out_steps - 1 < self.dataset.lengths[path])
                    for start in starts
                ),
            })
        return records

    def __iter__(self):
        return iter(self._selected_indices)

    def __len__(self) -> int:
        return len(self._selected_indices)
