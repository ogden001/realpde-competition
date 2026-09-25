"""Spatial-complete training data for the RealPDE SOTA merge.

Stage A uses every legal temporal window under every real 2x sampling phase:
P00, P01, P10, P11. This is exhaustive expansion, not probabilistic
augmentation. The same module is used by clean Train51 and full refits.
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

SPATIAL_PHASES: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))
PHASE_NAMES: tuple[str, ...] = ("P00", "P01", "P10", "P11")


@dataclass(frozen=True)
class SpatialViewRef:
    path: Path
    start: int
    phase_code: int

    @property
    def phase(self) -> tuple[int, int]:
        return SPATIAL_PHASES[self.phase_code]

    @property
    def phase_name(self) -> str:
        return PHASE_NAMES[self.phase_code]


def _field(handle: h5py.File, key: str):
    if key in handle:
        return handle[key]
    nested = f"measured_data/{key}"
    if nested in handle:
        return handle[nested]
    raise KeyError(key)


class SpatialPhaseExpandedDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Dense stride-1 temporal windows expanded over all four real 2x phases."""

    def __init__(
        self,
        paths: Sequence[Path],
        *,
        in_steps: int = 20,
        out_steps: int = 20,
        stride: int = 1,
        sub_sample: int = 2,
        include_pressure: bool = False,
        preload_to_ram: bool = False,
    ) -> None:
        if min(in_steps, out_steps, stride, sub_sample) < 1:
            raise ValueError("in_steps, out_steps, stride and sub_sample must be positive")
        if sub_sample != 2:
            raise ValueError("SpatialPhaseExpandedDataset is defined only for sub_sample=2")
        self.in_steps = int(in_steps)
        self.out_steps = int(out_steps)
        self.stride = int(stride)
        self.include_pressure = bool(include_pressure)
        self.preload_to_ram = bool(preload_to_ram)
        self.paths = list(paths)
        if not self.paths:
            raise ValueError("at least one HDF5 trajectory is required")

        self.base_refs: list[tuple[Path, int]] = []
        self._ram_cache: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
        self._cache_bytes = 0
        total = self.in_steps + self.out_steps

        for path in self.paths:
            with h5py.File(path, "r") as handle:
                u_field = _field(handle, "u")
                v_field = _field(handle, "v")
                if u_field.shape != v_field.shape or len(u_field.shape) != 3:
                    raise ValueError(f"invalid u/v field geometry in {path}")
                length, height, width = map(int, u_field.shape)
                if height % 2 or width % 2:
                    raise ValueError(f"2x phase expansion requires even H/W, got {height}x{width} in {path}")
                starts = list(range(0, length - total + 1, self.stride))
                self.base_refs.extend((path, start) for start in starts)

                if self.preload_to_ram:
                    u = np.asarray(u_field[:], dtype=np.float32)
                    v = np.asarray(v_field[:], dtype=np.float32)
                    p: np.ndarray | None = None
                    if self.include_pressure:
                        try:
                            p = np.asarray(_field(handle, "p")[:], dtype=np.float32)
                        except KeyError:
                            p = None
                    self._ram_cache[path] = (u, v, p)
                    self._cache_bytes += int(u.nbytes + v.nbytes + (0 if p is None else p.nbytes))

        if not self.base_refs:
            raise ValueError("no legal temporal windows fit the provided trajectories")

    @property
    def base_window_count(self) -> int:
        return len(self.base_refs)

    @property
    def cache_bytes(self) -> int:
        return int(self._cache_bytes)

    def cache_summary(self) -> dict[str, object]:
        return {
            "enabled": self.preload_to_ram,
            "trajectories": len(self._ram_cache),
            "bytes": int(self._cache_bytes),
            "gib": float(self._cache_bytes / (1024 ** 3)),
            "full_resolution_spatial_cache": self.preload_to_ram,
        }

    def phase_counts(self) -> dict[str, int]:
        return {name: self.base_window_count for name in PHASE_NAMES}

    def __len__(self) -> int:
        return 4 * len(self.base_refs)

    def ref(self, index: int) -> SpatialViewRef:
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        base_index, phase_code = divmod(int(index), 4)
        path, start = self.base_refs[base_index]
        return SpatialViewRef(path=path, start=start, phase_code=phase_code)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ref = self.ref(index)
        total = self.in_steps + self.out_steps
        sl = slice(ref.start, ref.start + total)
        dy, dx = ref.phase

        if self.preload_to_ram:
            u_all, v_all, p_all = self._ram_cache[ref.path]
            u = u_all[sl, dy::2, dx::2]
            v = v_all[sl, dy::2, dx::2]
            p = p_all[sl, dy::2, dx::2] if p_all is not None else np.zeros_like(u)
            with h5py.File(ref.path, "r") as handle:
                re = float(handle["re"][()]) if "re" in handle else 0.0
                aoa = float(handle["aoa"][()]) if "aoa" in handle else 0.0
        else:
            with h5py.File(ref.path, "r") as handle:
                u = np.asarray(_field(handle, "u")[sl, dy::2, dx::2], dtype=np.float32)
                v = np.asarray(_field(handle, "v")[sl, dy::2, dx::2], dtype=np.float32)
                if self.include_pressure:
                    try:
                        p = np.asarray(_field(handle, "p")[sl, dy::2, dx::2], dtype=np.float32)
                    except KeyError:
                        p = np.zeros_like(u)
                else:
                    p = np.zeros_like(u)
                re = float(handle["re"][()]) if "re" in handle else 0.0
                aoa = float(handle["aoa"][()]) if "aoa" in handle else 0.0

        if u.shape != v.shape or u.shape[1:] != (32, 64):
            raise ValueError(f"phase {ref.phase_name} produced unexpected shape {u.shape} from {ref.path}")
        full = torch.from_numpy(np.stack([u, v, p], axis=-1))
        condition = torch.tensor([re, aoa], dtype=torch.float32)
        return (
            full[: self.in_steps],
            full[self.in_steps :],
            condition,
            torch.tensor(int(index), dtype=torch.long),
        )


class SpatialPhaseExpandedSampler(Sampler[int]):
    """Shuffle one complete four-phase epoch with exact resume support.

    If 4*N is not divisible by batch 8, one complete four-phase quartet from a
    deterministic base window is repeated. This preserves exact phase balance
    and prevents silent view dropping.
    """

    def __init__(self, dataset: SpatialPhaseExpandedDataset, *, seed: int) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        self.start_index = 0
        self._order: list[int] = []
        self.pad_base_index: int | None = None
        self.set_epoch(0, start_index=0)

    def set_epoch(self, epoch: int, *, start_index: int = 0) -> None:
        if epoch < 0 or start_index < 0:
            raise ValueError("invalid sampler state")
        self.epoch = int(epoch)
        self.start_index = int(start_index)
        order = np.arange(len(self.dataset), dtype=np.int64).tolist()
        self.pad_base_index = None
        if len(order) % 8:
            self.pad_base_index = self.epoch % self.dataset.base_window_count
            first = 4 * self.pad_base_index
            order.extend([first + phase for phase in range(4)])
        if len(order) % 8:
            raise RuntimeError("spatial epoch padding failed to preserve batch=8")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 4]))
        rng.shuffle(order)
        self._order = order
        if self.start_index > len(self._order):
            raise ValueError("start_index outside padded spatial epoch")

    @property
    def full_epoch_size(self) -> int:
        return len(self._order)

    def __iter__(self):
        return iter(self._order[self.start_index :])

    def __len__(self) -> int:
        return len(self._order) - self.start_index

    def audit(self) -> dict[str, object]:
        unique = self.dataset.phase_counts()
        pad = 1 if self.pad_base_index is not None else 0
        return {
            "epoch": self.epoch,
            "start_index": self.start_index,
            "remaining_views": len(self),
            "unique_views": len(self.dataset),
            "stream_views_full_epoch": len(self._order),
            "base_windows": self.dataset.base_window_count,
            "balance_pad_views": len(self._order) - len(self.dataset),
            "balance_pad_base_index": self.pad_base_index,
            "phase_counts_stream_full_epoch": {name: count + pad for name, count in unique.items()},
            "phase_fraction_stream_full_epoch": {
                name: (count + pad) / max(1, len(self._order)) for name, count in unique.items()
            },
        }
