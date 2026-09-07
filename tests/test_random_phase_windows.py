from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from realpde_p0_data import (  # noqa: E402
    H5WindowDataset,
    RandomPhaseWindowSampler,
)


def _write_trajectory(path: Path, length: int) -> None:
    values = np.arange(length * 2 * 3, dtype=np.float32).reshape(length, 2, 3)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("u", data=values)
        handle.create_dataset("v", data=values + 1000)
        handle.create_dataset("p", data=values + 2000)
        handle.create_dataset("x", data=np.arange(3, dtype=np.float32)[None, :].repeat(2, axis=0))
        handle.create_dataset("y", data=np.arange(2, dtype=np.float32)[:, None].repeat(3, axis=1))
        handle.create_dataset("re", data=1000.0)
        handle.create_dataset("aoa", data=5.0)


def _paths(tmp_path: Path) -> list[Path]:
    paths = [tmp_path / "a.h5", tmp_path / "b.h5", tmp_path / "c.h5"]
    for path, length in zip(paths, (100, 121, 83), strict=True):
        _write_trajectory(path, length)
    return paths


def _starts(dataset: H5WindowDataset, indices: list[int]) -> list[tuple[str, int]]:
    return [(dataset.refs[index].path.name, dataset.refs[index].start) for index in indices]


class _IndexDataset(torch.utils.data.Dataset):
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index


def test_fixed_mode_preserves_existing_stride_twenty_refs(tmp_path: Path):
    dataset = H5WindowDataset(_paths(tmp_path), window_mode="fixed", stride=20, sub_sample=1)

    assert _starts(dataset, list(range(len(dataset)))) == [
        ("a.h5", 0),
        ("a.h5", 20),
        ("a.h5", 40),
        ("a.h5", 60),
        ("b.h5", 0),
        ("b.h5", 20),
        ("b.h5", 40),
        ("b.h5", 60),
        ("b.h5", 80),
        ("c.h5", 0),
        ("c.h5", 20),
        ("c.h5", 40),
    ]


def test_phase_seven_selects_only_seven_modulo_twenty_windows(tmp_path: Path):
    dataset = H5WindowDataset(_paths(tmp_path), window_mode="random_phase", stride=20, sub_sample=1)
    sampler = RandomPhaseWindowSampler(dataset, seed=20260901)
    sampler.set_epoch(0, phases={path.name: 7 for path in _paths(tmp_path)})
    selected = _starts(dataset, list(iter(sampler)))

    assert selected == [
        ("a.h5", 7),
        ("a.h5", 27),
        ("a.h5", 47),
        ("b.h5", 7),
        ("b.h5", 27),
        ("b.h5", 47),
        ("b.h5", 67),
        ("c.h5", 7),
        ("c.h5", 27),
    ]
    assert all(start % 20 == 7 for _, start in selected)


def test_random_phase_windows_respect_boundary_and_contiguous_past_future(tmp_path: Path):
    dataset = H5WindowDataset(_paths(tmp_path), window_mode="random_phase", stride=20, sub_sample=1)
    sampler = RandomPhaseWindowSampler(dataset, seed=20260901)
    sampler.set_epoch(3)

    selected_by_path: dict[Path, list[int]] = {}
    for index in sampler:
        ref = dataset.refs[index]
        selected_by_path.setdefault(ref.path, []).append(ref.start)

    for path, selected_starts in selected_by_path.items():
        length = dataset.lengths[path]
        assert selected_starts == list(range(selected_starts[0], selected_starts[-1] + 1, 20))
        assert all(0 <= start and start + 39 < length for start in selected_starts)
        with h5py.File(path, "r") as handle:
            for start in selected_starts:
                past = np.asarray(handle["u"][start:start + 20])
                future = np.asarray(handle["u"][start + 20:start + 40])
                assert past.shape[0] == future.shape[0] == 20
                np.testing.assert_array_equal(past, handle["u"][start:start + 20])
                np.testing.assert_array_equal(future, handle["u"][start + 20:start + 40])


def test_same_seed_and_epoch_reproduce_per_trajectory_phases_and_starts(tmp_path: Path):
    paths = _paths(tmp_path)
    dataset = H5WindowDataset(paths, window_mode="random_phase", stride=20, sub_sample=1)
    first = RandomPhaseWindowSampler(dataset, seed=20260901)
    second = RandomPhaseWindowSampler(dataset, seed=20260901)
    first.set_epoch(4)
    second.set_epoch(4)

    assert first.phases == second.phases
    assert _starts(dataset, list(iter(first))) == _starts(dataset, list(iter(second)))
    assert len(set(first.phases.values())) > 1


def test_different_epoch_rephases_the_trajectories(tmp_path: Path):
    dataset = H5WindowDataset(_paths(tmp_path), window_mode="random_phase", stride=20, sub_sample=1)
    sampler = RandomPhaseWindowSampler(dataset, seed=20260901)
    sampler.set_epoch(4)
    first = sampler.phases.copy()
    sampler.set_epoch(5)

    assert sampler.phases != first


def test_workers_zero_and_two_produce_the_same_sampler_window_set(tmp_path: Path):
    dataset = H5WindowDataset(_paths(tmp_path), window_mode="random_phase", stride=20, sub_sample=1)
    sampler_zero = RandomPhaseWindowSampler(dataset, seed=20260901)
    sampler_two = RandomPhaseWindowSampler(dataset, seed=20260901)
    sampler_zero.set_epoch(6)
    sampler_two.set_epoch(6)
    index_dataset = _IndexDataset(len(dataset.refs))
    loader_zero = DataLoader(index_dataset, batch_size=3, sampler=sampler_zero, num_workers=0)
    loader_two = DataLoader(index_dataset, batch_size=3, sampler=sampler_two, num_workers=2)

    zero_indices = [index for batch in loader_zero for index in batch.tolist()]
    two_indices = [index for batch in loader_two for index in batch.tolist()]

    assert _starts(dataset, zero_indices) == _starts(dataset, two_indices)


def test_dev_dataset_remains_fixed_when_train_dataset_is_random_phase(tmp_path: Path):
    paths = _paths(tmp_path)
    train = H5WindowDataset(paths[:2], window_mode="random_phase", stride=20, sub_sample=1)
    train_sampler = RandomPhaseWindowSampler(train, seed=20260901)
    train_sampler.set_epoch(0)
    dev = H5WindowDataset(paths[2:], window_mode="fixed", stride=20, sub_sample=1)

    assert any(ref.start % 20 != 0 for ref in train.refs)
    assert [ref.start for ref in dev.refs] == [0, 20, 40]
