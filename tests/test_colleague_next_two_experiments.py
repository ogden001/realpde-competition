from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
COLLEAGUE_TOOLS = ROOT / "tools" / "colleague_80pt"
sys.path.insert(0, str(COLLEAGUE_TOOLS))
sys.path.insert(0, str(ROOT / "tools"))

from fluctuation_calibration import mean_preserving_ramp_scale, write_csv  # noqa: E402
from residual_multi import rotate_velocity_uv  # noqa: E402


def test_fluctuation_scale_one_is_identity() -> None:
    rng = np.random.default_rng(7)
    pred = rng.normal(size=(3, 20, 4, 5, 3)).astype(np.float32)
    pred[..., 2] = 0.0
    out = mean_preserving_ramp_scale(pred, 1.0)
    np.testing.assert_allclose(out, pred, atol=2e-6, rtol=0.0)


def test_fluctuation_ramp_preserves_temporal_mean() -> None:
    rng = np.random.default_rng(11)
    pred = rng.normal(size=(2, 20, 3, 4, 3)).astype(np.float32)
    pred[..., 2] = 0.0
    out = mean_preserving_ramp_scale(pred, 1.6)
    np.testing.assert_allclose(
        out[..., :2].mean(axis=1),
        pred[..., :2].mean(axis=1),
        atol=2e-6,
        rtol=0.0,
    )
    assert np.all(out[..., 2] == 0.0)


def test_velocity_rotation_preserves_speed_and_pressure() -> None:
    x = torch.zeros((2, 1, 1, 1, 3), dtype=torch.float32)
    x[0, ..., 0] = 1.0
    x[1, ..., 1] = 2.0
    x[..., 2] = 3.0
    out = rotate_velocity_uv(x, torch.tensor([90.0, -90.0]))
    before_speed = torch.linalg.vector_norm(x[..., :2], dim=-1)
    after_speed = torch.linalg.vector_norm(out[..., :2], dim=-1)
    torch.testing.assert_close(after_speed, before_speed, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(out[..., 2], x[..., 2], atol=0.0, rtol=0.0)
    torch.testing.assert_close(out[0, ..., 0], torch.zeros_like(out[0, ..., 0]), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(out[0, ..., 1], torch.ones_like(out[0, ..., 1]), atol=1e-6, rtol=0.0)


def test_velocity_rotation_requires_one_angle_per_sample() -> None:
    x = torch.zeros((2, 2, 2, 2, 3), dtype=torch.float32)
    try:
        rotate_velocity_uv(x, torch.tensor([1.0, 2.0, 3.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched batch angles")



def test_write_csv_accepts_union_schema(tmp_path) -> None:
    path = tmp_path / "mixed.csv"
    rows = [
        {"experiment": "baseline", "horizon": 1, "frame_rel_l2": 0.1},
        {
            "experiment": "candidate",
            "horizon": 1,
            "frame_rel_l2": 0.09,
            "base_frame_rel_l2": 0.1,
            "delta_rms": 0.01,
        },
    ]
    write_csv(path, rows)
    text = path.read_text(encoding="utf-8")
    assert "base_frame_rel_l2" in text.splitlines()[0]
    assert "delta_rms" in text.splitlines()[0]
    assert len(text.splitlines()) == 3
