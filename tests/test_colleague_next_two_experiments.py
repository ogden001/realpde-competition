from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
COLLEAGUE_TOOLS = ROOT / "tools" / "colleague_80pt"
sys.path.insert(0, str(COLLEAGUE_TOOLS))
sys.path.insert(0, str(ROOT / "tools"))

from fluctuation_calibration import mean_preserving_ramp_scale, write_csv  # noqa: E402
from aoa_meanfield_augmentation import (  # noqa: E402
    AoAMeanFieldShiftDataset,
    build_adjacent_aoa_neighbors,
)
from realpde_h5_feature_adapter_train import H5WindowDataset  # noqa: E402
from residual_multi import rotate_velocity_uv  # noqa: E402
from run_next_two_experiments import alpha_one_metrics  # noqa: E402


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


def test_alpha_one_metrics_accepts_eval_schema(tmp_path) -> None:
    path = tmp_path / "eval_step.json"
    path.write_text(
        '[{"alpha": 1.0, "rel_l2": 0.08, "tke": 0.45, '
        '"mvpe": 0.07, "best_final_est": 81.5}]',
        encoding="utf-8",
    )
    assert alpha_one_metrics(path) == {
        "rel_l2_raw": 0.08,
        "tke_raw": 0.45,
        "mvpe_raw": 0.07,
        "final_est": 81.5,
        "best_alpha": 1.0,
    }



def _write_condition_h5(
    path: Path,
    *,
    re_value: float,
    aoa: float,
    past_u: float,
    future_u: float,
) -> None:
    u = np.full((40, 4, 6), past_u, dtype=np.float32)
    u[20:] = future_u
    v = np.zeros_like(u)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("u", data=u)
        handle.create_dataset("v", data=v)
        handle.create_dataset("x", data=np.arange(6, dtype=np.float32)[None, :].repeat(4, axis=0))
        handle.create_dataset("y", data=np.arange(4, dtype=np.float32)[:, None].repeat(6, axis=1))
        handle.create_dataset("re", data=np.asarray(re_value, dtype=np.float32))
        handle.create_dataset("aoa", data=np.asarray(aoa, dtype=np.float32))


def test_adjacent_aoa_neighbors_require_same_re(tmp_path: Path) -> None:
    a = tmp_path / "1000_0.h5"
    b = tmp_path / "1000_5.h5"
    c = tmp_path / "2000_5.h5"
    _write_condition_h5(a, re_value=1000, aoa=0, past_u=1, future_u=2)
    _write_condition_h5(b, re_value=1000, aoa=5, past_u=3, future_u=4)
    _write_condition_h5(c, re_value=2000, aoa=5, past_u=9, future_u=9)

    mapping, _ = build_adjacent_aoa_neighbors([a, b, c], max_gap_deg=5.1)
    assert mapping[a] == (b,)
    assert mapping[b] == (a,)
    assert mapping[c] == ()


def test_aoa_meanfield_shift_uses_past_mean_and_preserves_fluctuation_phase(tmp_path: Path) -> None:
    a = tmp_path / "1000_0.h5"
    b = tmp_path / "1000_5.h5"
    _write_condition_h5(a, re_value=1000, aoa=0, past_u=1, future_u=2)
    # Deliberately make neighbor Future20 absurdly different.  The augmentation
    # must still derive its shift from neighbor Past20 only.
    _write_condition_h5(b, re_value=1000, aoa=5, past_u=3, future_u=99)

    base = H5WindowDataset(
        [a, b],
        in_steps=20,
        out_steps=20,
        stride=20,
        sub_sample=1,
        include_pressure=False,
        window_mode="fixed",
    )
    wrapped = AoAMeanFieldShiftDataset(
        base,
        probability=1.0,
        lambda_min=0.5,
        lambda_max=0.5,
        seed=41,
        max_gap_deg=5.1,
        min_eligible_fraction=1.0,
    )

    x_raw, y_raw = base[0]
    x_aug, y_aug, meta = wrapped[0]
    # Past mean changes from 1 toward 3 by lambda=.5, so the same +1 spatial
    # shift is applied to both Past20 and Future20 of the anchor.
    torch.testing.assert_close(x_aug[..., 0], x_raw[..., 0] + 1.0)
    torch.testing.assert_close(y_aug[..., 0], y_raw[..., 0] + 1.0)
    assert float(meta["applied"]) == 1.0
    assert abs(float(meta["effective_shift_deg"]) - 2.5) < 1e-6
    # No pressure channel is created by the augmentation.
    assert torch.count_nonzero(x_aug[..., 2]) == 0
    assert torch.count_nonzero(y_aug[..., 2]) == 0


def test_aoa_meanfield_coverage_guard_rejects_sparse_pairing(tmp_path: Path) -> None:
    a = tmp_path / "1000_0.h5"
    b = tmp_path / "2000_5.h5"
    _write_condition_h5(a, re_value=1000, aoa=0, past_u=1, future_u=2)
    _write_condition_h5(b, re_value=2000, aoa=5, past_u=3, future_u=4)
    base = H5WindowDataset(
        [a, b],
        in_steps=20,
        out_steps=20,
        stride=20,
        sub_sample=1,
        include_pressure=False,
        window_mode="fixed",
    )
    try:
        AoAMeanFieldShiftDataset(
            base,
            probability=0.5,
            lambda_min=0.2,
            lambda_max=0.5,
            seed=41,
            max_gap_deg=5.1,
            min_eligible_fraction=0.8,
        )
    except RuntimeError as error:
        assert "coverage too low" in str(error)
    else:
        raise AssertionError("expected sparse same-Re/AoA coverage to be rejected")



def test_adjacent_aoa_neighbors_reject_mismatched_coordinate_grid(tmp_path: Path) -> None:
    a = tmp_path / "1000_0.h5"
    b = tmp_path / "1000_5.h5"
    _write_condition_h5(a, re_value=1000, aoa=0, past_u=1, future_u=2)
    _write_condition_h5(b, re_value=1000, aoa=5, past_u=3, future_u=4)
    with h5py.File(b, "r+") as handle:
        handle["x"][...] = handle["x"][...] + 0.25
    mapping, _ = build_adjacent_aoa_neighbors([a, b], max_gap_deg=5.1)
    assert mapping[a] == ()
    assert mapping[b] == ()
