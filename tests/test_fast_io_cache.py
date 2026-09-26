from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import realpde_p0_data as p0_data  # noqa: E402
import realpde_sota_merge_spatial as spatial  # noqa: E402
from realpde_p0_data import DenseAllWindowSampler, H5WindowDataset  # noqa: E402


def _write_trajectory(path: Path, *, length: int = 45, height: int = 64, width: int = 128) -> None:
    base = np.arange(length, dtype=np.float32)[:, None, None]
    yy = np.arange(height, dtype=np.float32)[None, :, None] * 0.01
    xx = np.arange(width, dtype=np.float32)[None, None, :] * 0.001
    u = base + yy + xx
    v = 1000.0 + 2.0 * base + yy - xx
    with h5py.File(path, "w") as handle:
        handle.create_dataset("u", data=u)
        handle.create_dataset("v", data=v)
        handle.create_dataset("x", data=np.arange(width, dtype=np.float32)[None, :].repeat(height, axis=0))
        handle.create_dataset("y", data=np.arange(height, dtype=np.float32)[:, None].repeat(width, axis=1))
        handle.create_dataset("re", data=6300.0)
        handle.create_dataset("aoa", data=10.0)


def test_h5_window_ram_cache_is_bitwise_equivalent_and_hot_path_has_no_h5_open(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "case.h5"
    _write_trajectory(path)

    disk = H5WindowDataset(
        [path], in_steps=20, out_steps=20, stride=20, sub_sample=2,
        include_pressure=False, window_mode="dense_all", preload_to_ram=False,
    )
    ram = H5WindowDataset(
        [path], in_steps=20, out_steps=20, stride=20, sub_sample=2,
        include_pressure=False, window_mode="dense_all", preload_to_ram=True,
    )

    assert ram.cache_summary()["enabled"] is True
    assert ram.cache_summary()["trajectories"] == 1
    assert ram.cache_bytes > 0
    for index in (0, len(ram) - 1):
        disk_values = disk[index]
        ram_values = ram[index]
        for left, right in zip(disk_values, ram_values):
            assert torch.equal(left, right)

    def forbidden_open(*args, **kwargs):
        raise AssertionError("cached __getitem__ reopened HDF5")

    monkeypatch.setattr(p0_data.h5py, "File", forbidden_open)
    past, future, condition, index = ram[1]
    assert past.shape == (20, 32, 64, 3)
    assert future.shape == (20, 32, 64, 3)
    assert condition.tolist() == [6300.0, 10.0]
    assert index.item() == 1


def test_dense_all_sampler_start_index_is_exact_suffix(tmp_path: Path):
    paths = [tmp_path / "a.h5", tmp_path / "b.h5"]
    for path in paths:
        _write_trajectory(path, length=48)

    dataset = H5WindowDataset(
        paths, in_steps=20, out_steps=20, stride=20, sub_sample=2,
        include_pressure=False, window_mode="dense_all", preload_to_ram=True,
    )
    full = DenseAllWindowSampler(dataset, seed=20260901)
    full.set_epoch(3)
    full_order = list(full)

    resumed = DenseAllWindowSampler(dataset, seed=20260901)
    resumed.set_epoch(3, start_index=16)
    assert list(resumed) == full_order[16:]
    assert len(resumed) == len(full_order) - 16
    assert resumed.full_epoch_size == len(full_order)


def test_spatial_ram_cache_hot_path_has_no_h5_open(tmp_path: Path, monkeypatch):
    path = tmp_path / "case.h5"
    _write_trajectory(path)

    dataset = spatial.SpatialPhaseExpandedDataset(
        [path],
        in_steps=20,
        out_steps=20,
        stride=1,
        sub_sample=2,
        include_pressure=False,
        preload_to_ram=True,
    )
    assert dataset.cache_summary()["enabled"] is True

    expected = dataset[0]

    def forbidden_open(*args, **kwargs):
        raise AssertionError("spatial cached __getitem__ reopened HDF5")

    monkeypatch.setattr(spatial.h5py, "File", forbidden_open)
    observed = dataset[0]
    for left, right in zip(expected, observed):
        assert torch.equal(left, right)
