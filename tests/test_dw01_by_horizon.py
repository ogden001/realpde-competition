from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from dw01_by_horizon import (  # noqa: E402
    aggregate_by_horizon,
    compute_window_horizon_metrics,
    official_probe_geometry,
)


def test_perfect_prediction_is_zero_and_has_twenty_horizons():
    target = np.ones((2, 20, 32, 64, 3), dtype=np.float32)
    rows = compute_window_horizon_metrics(target, target, ["a.h5", "b.h5"], [0, 20])
    assert len(rows) == 40
    assert all(row["frame_rel_l2"] == pytest.approx(0.0) for row in rows)
    assert all(row["frame_rmse"] == pytest.approx(0.0) for row in rows)
    assert all(row["tke_contrib_rel_l2"] == pytest.approx(0.0) for row in rows)
    assert all(row["tke_contrib_ratio"] == pytest.approx(1.0) for row in rows)
    assert all(row["mvpe_probe_rel_l2"] == pytest.approx(0.0) for row in rows)


def test_metrics_use_uv_only_and_tke_uses_full_future20_mean():
    target = np.zeros((1, 20, 32, 64, 3), dtype=np.float32)
    target[..., 0] = 1.0
    prediction = target.copy()
    prediction[..., 2] = 999.0  # pressure must not affect diagnostics
    prediction[:, 0, ..., 0] = 2.0
    rows = compute_window_horizon_metrics(prediction, target, ["a.h5"], [0])
    assert rows[0]["frame_rel_l2"] == pytest.approx(1.0)
    assert rows[0]["frame_rmse"] == pytest.approx(1.0 / np.sqrt(2.0))
    assert rows[1]["frame_rel_l2"] == pytest.approx(0.0)
    assert rows[0]["tke_contrib_ratio"] > 1.0
    assert rows[0]["tke_contrib_rel_l2"] > 0.0


def test_official_v9_probe_geometry_is_used_for_32x64():
    ys, xs = official_probe_geometry(32, 64)
    assert ys == [8, 10, 12, 14, 16, 18, 20, 22, 24]
    assert xs == [13, 21, 29, 37]


def test_aggregate_has_one_row_per_horizon_and_mean_window_metrics():
    target = np.ones((2, 20, 4, 8, 3), dtype=np.float32)
    prediction = target.copy()
    prediction[0, :, :, :, 0] *= 2.0
    rows = compute_window_horizon_metrics(prediction, target, ["a.h5", "b.h5"], [0, 20])
    aggregate = aggregate_by_horizon(rows, experiment="DW-01", trajectories=2)
    assert len(aggregate) == 20
    assert [row["horizon"] for row in aggregate] == list(range(1, 21))
    assert aggregate[0]["experiment"] == "DW-01"
    assert aggregate[0]["windows"] == 2
    assert aggregate[0]["frame_rel_l2"] == pytest.approx(0.3535533906)
