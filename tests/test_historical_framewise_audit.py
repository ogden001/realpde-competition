from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def test_frame_and_window_metrics_use_velocity_channels_only():
    from historical_framewise_audit import frame_metrics, per_window_frame_metrics

    prediction = np.zeros((2, 2, 1, 1, 3), dtype=np.float32)
    target = np.zeros_like(prediction)
    target[..., 0] = 1.0
    prediction[0, ..., 0] = 2.0
    prediction[1, ..., 0] = 4.0
    # Pressure must not affect the requested velocity metrics.
    prediction[..., 2] = 1000.0

    frame = frame_metrics(prediction, target)
    rows = per_window_frame_metrics("E", ["a.h5", "b.h5"], [0, 20], prediction, target)

    assert [row["rel_l2"] for row in frame] == pytest.approx([2.0, 2.0])
    assert [row["velocity_squared_error"] for row in frame] == pytest.approx([5.0, 5.0])
    assert len(rows) == 4
    assert rows[0] == {
        "experiment": "E", "trajectory": "a.h5", "window_start": 0, "horizon": 1,
        "rel_l2": 1.0, "mvpe": 1.0, "velocity_squared_error": 1.0,
        "pred_speed_mean": 2.0, "target_speed_mean": 1.0,
    }
    assert rows[-1]["rel_l2"] == pytest.approx(3.0)


def test_inventory_rows_require_the_requested_provenance_fields():
    from historical_framewise_audit import INVENTORY_FIELDS, validate_inventory_row

    assert INVENTORY_FIELDS == [
        "experiment", "architecture", "features", "loss", "initialization", "train_split",
        "update", "checkpoint", "checkpoint_sha", "provenance_status", "replay_status",
    ]
    with pytest.raises(ValueError, match="checkpoint_sha"):
        validate_inventory_row({field: "x" for field in INVENTORY_FIELDS if field != "checkpoint_sha"})
