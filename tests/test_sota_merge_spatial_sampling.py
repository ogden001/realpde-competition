from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from realpde_sota_merge_spatial import (  # noqa: E402
    PHASE_NAMES,
    SPATIAL_PHASES,
    SpatialPhaseExpandedDataset,
    SpatialPhaseExpandedSampler,
)


def _write_case(path: Path, frames: int = 42) -> None:
    h, w = 64, 128
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    base = 1000.0 * yy + xx
    u = np.stack([base + 100_000.0 * t for t in range(frames)]).astype(np.float32)
    v = (u + 0.5).astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle["u"] = u
        handle["v"] = v
        handle["re"] = 6300.0
        handle["aoa"] = 5.0


def test_every_temporal_window_has_all_four_real_phases(tmp_path: Path) -> None:
    case = tmp_path / "case.h5"
    _write_case(case)
    ds = SpatialPhaseExpandedDataset([case], preload_to_ram=True)
    assert ds.base_window_count == 3
    assert len(ds) == 12
    assert ds.phase_counts() == {name: 3 for name in PHASE_NAMES}

    for phase_code, (dy, dx) in enumerate(SPATIAL_PHASES):
        past, future, _, encoded = ds[phase_code]
        assert int(encoded) == phase_code
        assert tuple(past.shape) == (20, 32, 64, 3)
        assert tuple(future.shape) == (20, 32, 64, 3)
        assert float(past[0, 0, 0, 0]) == float(1000 * dy + dx)
        assert float(future[0, 0, 0, 0]) == float(2_000_000 + 1000 * dy + dx)


def test_sampler_keeps_phase_balance_and_batch8_without_dropping_views(tmp_path: Path) -> None:
    case = tmp_path / "case.h5"
    _write_case(case)
    ds = SpatialPhaseExpandedDataset([case])
    sampler = SpatialPhaseExpandedSampler(ds, seed=41)

    assert sampler.full_epoch_size == 16
    order = list(iter(sampler))
    assert len(order) == 16
    assert set(range(12)).issubset(set(order))
    counts = {name: 0 for name in PHASE_NAMES}
    for index in order:
        counts[ds.ref(index).phase_name] += 1
    assert counts == {name: 4 for name in PHASE_NAMES}


def test_sampler_resume_is_exact(tmp_path: Path) -> None:
    case = tmp_path / "case.h5"
    _write_case(case)
    ds = SpatialPhaseExpandedDataset([case])
    sampler = SpatialPhaseExpandedSampler(ds, seed=41)
    complete = list(iter(sampler))
    sampler.set_epoch(0, start_index=8)
    assert list(iter(sampler)) == complete[8:]
    sampler.set_epoch(1, start_index=0)
    assert list(iter(sampler)) != complete
