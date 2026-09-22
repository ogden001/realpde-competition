from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from post_train_diagnostics import (  # noqa: E402
    residual_summary,
    spatial_maps,
    write_post_train_diagnostics,
)


def _arrays():
    rng = np.random.default_rng(7)
    target = rng.normal(size=(4, 20, 3, 5, 3)).astype(np.float32)
    target[..., 2] = 0.0
    base = target.copy()
    base[..., 0] += 0.20
    base[..., 1] -= 0.20
    corrected = target.copy()
    corrected[..., 0] += 0.10
    corrected[..., 1] -= 0.10
    return base, corrected, target


def test_residual_summary_detects_uniformly_better_correction() -> None:
    base, corrected, target = _arrays()
    summary = residual_summary(base, corrected, target)
    assert summary["correction_help_fraction"] == 1.0
    assert summary["corrected_rel_l2"] < summary["base_rel_l2"]


def test_spatial_maps_have_grid_shape() -> None:
    _, corrected, target = _arrays()
    maps = spatial_maps(corrected, target)
    assert set(maps) == {
        "u_rmse", "v_rmse", "velocity_rmse", "mean_field_rmse",
        "fluctuation_rmse", "tke_abs_error", "target_tke_mean",
    }
    assert all(value.shape == (3, 5) for value in maps.values())


def test_standard_bundle_is_written(tmp_path: Path) -> None:
    base, corrected, target = _arrays()
    summary = write_post_train_diagnostics(
        out_dir=tmp_path,
        experiment="unit",
        prediction=corrected,
        target=target,
        trajectories=["a.h5", "a.h5", "b.h5", "b.h5"],
        starts=[0, 20, 0, 20],
        base_prediction=base,
    )
    assert summary["windows"] == 4
    assert summary["horizons"] == 20
    assert (tmp_path / "by_horizon.csv").is_file()
    assert (tmp_path / "by_trajectory.csv").is_file()
    assert (tmp_path / "by_trajectory_horizon.csv").is_file()
    assert (tmp_path / "spatial_maps.npz").is_file()
    assert (tmp_path / "summary.json").is_file()
