from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from realpde_p0_data import DenseAllWindowSampler, H5WindowDataset  # noqa: E402


def _write_trajectory(path: Path, length: int, offset: float = 0.0) -> None:
    values = offset + np.arange(length, dtype=np.float32)[:, None, None] * np.ones((1, 2, 3), dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("u", data=values)
        handle.create_dataset("v", data=values + 1000)
        handle.create_dataset("p", data=values + 2000)
        handle.create_dataset("x", data=np.arange(3, dtype=np.float32)[None, :].repeat(2, axis=0))
        handle.create_dataset("y", data=np.arange(2, dtype=np.float32)[:, None].repeat(3, axis=1))
        handle.create_dataset("re", data=1000.0)
        handle.create_dataset("aoa", data=5.0)


def test_dense_all_t868_has_all_829_legal_starts_and_exact_boundary_windows(tmp_path: Path):
    """Fails if Dense-All omits a legal start or slices the final window incorrectly."""
    path = tmp_path / "long.h5"
    _write_trajectory(path, length=868)

    dataset = H5WindowDataset([path], window_mode="dense_all", stride=20, sub_sample=1)

    assert len(dataset) == 829
    assert [ref.start for ref in dataset.refs] == list(range(829))
    past, future, _, _ = dataset[828]
    assert past[:, 0, 0, 0].tolist() == list(range(828, 848))
    assert future[:, 0, 0, 0].tolist() == list(range(848, 868))


def test_dense_all_never_crosses_trajectory_and_fixed_dev_stays_canonical(tmp_path: Path):
    """Fails if dense refs leak across files or Dev inherits the dense protocol."""
    first, second, dev_path = tmp_path / "a.h5", tmp_path / "b.h5", tmp_path / "dev.h5"
    _write_trajectory(first, length=45, offset=0.0)
    _write_trajectory(second, length=45, offset=10000.0)
    _write_trajectory(dev_path, length=100, offset=20000.0)

    train = H5WindowDataset([first, second], window_mode="dense_all", stride=20, sub_sample=1)
    dev = H5WindowDataset([dev_path], window_mode="fixed", stride=20, sub_sample=1)

    assert [(ref.path.name, ref.start) for ref in train.refs] == [
        ("a.h5", 0), ("a.h5", 1), ("a.h5", 2), ("a.h5", 3), ("a.h5", 4), ("a.h5", 5),
        ("b.h5", 0), ("b.h5", 1), ("b.h5", 2), ("b.h5", 3), ("b.h5", 4), ("b.h5", 5),
    ]
    assert train[5][0][-1, 0, 0, 0].item() == 24.0
    assert train[6][0][0, 0, 0, 0].item() == 10000.0
    assert [ref.start for ref in dev.refs] == [0, 20, 40, 60]


def test_dense_all_global_shuffle_is_reproducible_and_workers_read_same_window_set(tmp_path: Path):
    """Fails if shuffle is trajectory-blocked/nondeterministic or worker reads change the window set."""
    paths = [tmp_path / "a.h5", tmp_path / "b.h5", tmp_path / "c.h5"]
    for index, path in enumerate(paths):
        _write_trajectory(path, length=83, offset=10000.0 * index)
    dataset = H5WindowDataset(paths, window_mode="dense_all", stride=20, sub_sample=1)
    first = DenseAllWindowSampler(dataset, seed=20260901)
    second = DenseAllWindowSampler(dataset, seed=20260901)
    first.set_epoch(4)
    second.set_epoch(4)

    first_order = list(first)
    assert first_order == list(second)
    assert sorted(first_order) == list(range(len(dataset)))
    assert len({dataset.refs[index].path for index in first_order[:12]}) > 1

    loader_zero = DataLoader(dataset, batch_size=5, sampler=first, num_workers=0)
    loader_two = DataLoader(dataset, batch_size=5, sampler=second, num_workers=2)
    zero_indices = [index for *_, indices in loader_zero for index in indices.tolist()]
    two_indices = [index for *_, indices in loader_two for index in indices.tolist()]
    assert zero_indices == two_indices == first_order
