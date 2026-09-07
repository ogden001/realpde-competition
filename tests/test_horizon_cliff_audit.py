from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def test_windowwise_horizon_rel_l2_means_per_window_not_global():
    from horizon_cliff_audit import windowwise_horizon_rel_l2

    prediction = np.zeros((2, 2, 1, 1, 3), dtype=np.float32)
    target = np.zeros_like(prediction)
    target[:, :, 0, 0, 0] = 1.0
    prediction[0, :, 0, 0, 0] = 2.0  # Rel-L2 = 1 for both horizons.
    prediction[1, :, 0, 0, 0] = 4.0  # Rel-L2 = 3 for both horizons.

    assert windowwise_horizon_rel_l2(prediction, target) == pytest.approx([2.0, 2.0])


def test_window_alignment_requires_future_to_begin_at_start_plus_20():
    from horizon_cliff_audit import verify_window_alignment

    trajectory = np.arange(50 * 2, dtype=np.float32).reshape(50, 1, 1, 2)
    past = trajectory[0:20]
    future = trajectory[20:40]
    details = verify_window_alignment(past, future, trajectory, start=0)

    assert details["future_frame_indices"] == [20, 38, 39]
    with pytest.raises(AssertionError, match="Future20"):
        verify_window_alignment(past, trajectory[19:39], trajectory, start=0)


def test_p0a_config_restores_legacy_checkpoint_without_boolean_flags():
    from horizon_cliff_audit import p0a_config_from_checkpoint

    config = p0a_config_from_checkpoint({"feature_set": "P0-A", "feature_config": {"dx": 0.1, "dy": -0.1}})

    assert config.include_p0_a is True
    assert config.include_p0_b is False


def test_prepare_out_dir_accepts_empty_mountpoint_and_rejects_nonempty(tmp_path: Path):
    from horizon_cliff_audit import prepare_out_dir

    empty = tmp_path / "empty"; empty.mkdir()
    prepare_out_dir(empty)
    (empty / "prior.txt").write_text("evidence")
    with pytest.raises(FileExistsError):
        prepare_out_dir(empty)
