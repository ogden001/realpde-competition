from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

warnings.filterwarnings("ignore", message="Failed to initialize NumPy:.*", category=UserWarning)
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from realpde_p0_data import H5WindowDataset  # noqa: E402
from realpde_p0_features import P0FeatureBuilder, P0FeatureConfig  # noqa: E402


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "realpde"
EXPECTED_SCHEMA = {"u", "v", "p", "x", "y", "re", "aoa"}
FROZEN_TRAIN_SOURCES = {"3750_10.h5", "20325_0.h5", "16500_0.h5"}
EXPECTED_FILES = {
    "train_sample_a.h5": (80, 3, "3750_10.h5"),
    "train_sample_b.h5": (80, 3, "20325_0.h5"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_fixture(path: Path) -> dict[str, np.ndarray | float]:
    with h5py.File(path, "r") as handle:
        assert set(handle.keys()) == EXPECTED_SCHEMA
        for name in ("u", "v", "p", "x", "y"):
            assert handle[name].compression == "gzip"
        return {
            "u": handle["u"][...],
            "v": handle["v"][...],
            "p": handle["p"][...],
            "x": handle["x"][...],
            "y": handle["y"][...],
            "re": float(handle["re"][()]),
            "aoa": float(handle["aoa"][()]),
        }


def test_train_fixtures_have_schema_shape_windows_and_provenance() -> None:
    for filename, (frames, fixed_count, source_name) in EXPECTED_FILES.items():
        path = FIXTURE_ROOT / filename
        summary_path = path.with_suffix(".json")
        assert path.is_file(), f"missing fixture: {path}"
        assert summary_path.is_file(), f"missing fixture summary: {summary_path}"

        data = _read_fixture(path)
        u, v, p = data["u"], data["v"], data["p"]
        x, y = data["x"], data["y"]
        assert isinstance(u, np.ndarray)
        assert u.shape == v.shape == p.shape == (frames, 32, 64)
        assert x.shape == y.shape == (32, 64)
        assert all(np.isfinite(field).all() for field in (u, v, p, x, y))
        assert np.array_equal(p, np.zeros_like(p)), "real-data pressure must be exactly zero"

        # The first legal sample is exactly Past20 followed by Future20.
        past_u, future_u = u[:20], u[20:40]
        past_v, future_v = v[:20], v[20:40]
        assert past_u.shape == future_u.shape == (20, 32, 64)
        assert past_v.shape == future_v.shape == (20, 32, 64)
        assert np.array_equal(np.concatenate((past_u, future_u)), u[:40])
        assert np.array_equal(np.concatenate((past_v, future_v)), v[:40])

        dx = float(np.median(np.diff(x[0])))
        dy = float(np.median(np.diff(y[:, 0])))
        # Use a Python-list bridge because some local PyTorch/NumPy pairs do
        # not expose the NumPy C API required by torch.from_numpy().
        input_window = torch.tensor(np.stack((u[:20], v[:20], p[:20]), axis=-1).tolist()).unsqueeze(0).float()
        features = P0FeatureBuilder(P0FeatureConfig(include_p0_b=False, dx=dx, dy=dy))(input_window)
        assert features.shape == (1, 20, 32, 64, 20)
        assert torch.isfinite(features).all()

        fixed = H5WindowDataset([path], in_steps=20, out_steps=20, stride=20, sub_sample=1, window_mode="fixed")
        dense = H5WindowDataset([path], in_steps=20, out_steps=20, stride=1, sub_sample=1, window_mode="dense_all")
        assert len(fixed) == fixed_count
        assert len(dense) == (829 if frames == 868 else 41)

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["source_trajectory_filename"] == source_name
        assert summary["source_split"] == "train"
        assert source_name in FROZEN_TRAIN_SOURCES
        assert summary["retained_frame_range"] == {"start": 0, "stop_exclusive": frames}
        assert summary["spatial_subsampling"] == {
            "source_shape": [64, 128],
            "step": 2,
            "output_shape": [32, 64],
        }
        assert summary["u_shape"] == list(u.shape)
        assert summary["v_shape"] == list(v.shape)
        assert summary["p_shape"] == list(p.shape)
        assert summary["x_shape"] == list(x.shape)
        assert summary["y_shape"] == list(y.shape)
        assert summary["re"] == data["re"]
        assert summary["aoa"] == data["aoa"]
        assert summary["pressure_semantics"] == "real PIV has no measured pressure; p is exactly zero"
        for name, field in (("u_stats", u), ("v_stats", v)):
            for stat in ("min", "max", "mean", "std"):
                assert summary[name][stat] == pytest.approx(float(getattr(field, stat)()))
        assert summary["dx"] == pytest.approx(dx)
        assert summary["dy"] == pytest.approx(dy)
        assert summary["fixed_window_protocol"] == {"tin": 20, "tout": 20, "stride": 20}
        assert summary["dense_all_window_protocol"] == {"tin": 20, "tout": 20, "stride": 1}
        assert summary["fixed_window_count"] == fixed_count
        assert summary["dense_all_legal_window_count"] == (829 if frames == 868 else 41)
        assert summary["sha256"] == _sha256(path)


def test_full_868_fixture_has_all_dense_all_legal_windows() -> None:
    path = FIXTURE_ROOT / "train_full_868.h5"
    if not path.is_file():
        pytest.skip("full 868-frame fixture is intentionally omitted when it exceeds 20 MB")
    dataset = H5WindowDataset([path], in_steps=20, out_steps=20, stride=1, sub_sample=1, window_mode="dense_all")

    assert len(dataset) == 829
    assert [ref.start for ref in dataset.refs] == list(range(829))
    past, future, _, _ = dataset[828]
    assert past.shape == future.shape == (20, 32, 64, 3)
