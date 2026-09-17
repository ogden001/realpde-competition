from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import realpde_tmr01a_trajectory_horizon_audit as audit  # noqa: E402


def _nontrivial_target(n: int = 2, t: int = 20, h: int = 32, w: int = 64) -> np.ndarray:
    y = np.zeros((n, t, h, w, 3), dtype=np.float32)
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[None, None, :, None]
    xx = np.linspace(-0.5, 0.5, w, dtype=np.float32)[None, None, None, :]
    tt = np.linspace(-1.0, 1.0, t, dtype=np.float32)[None, :, None, None]
    y[..., 0] = 1.5 + 0.30 * tt + 0.10 * yy + 0.05 * xx
    y[..., 1] = -0.7 - 0.20 * tt + 0.04 * yy - 0.08 * xx
    return y


def test_window_horizon_diagnostics_identity_is_exact():
    target = _nontrivial_target()
    d = audit.window_horizon_diagnostics(target.copy(), target)

    assert d["frame_rel_l2"].shape == (2, 20)
    assert d["frame_rmse"].shape == (2, 20)
    assert d["tke_contrib_rel_l2"].shape == (2, 20)
    assert d["tke_contrib_ratio"].shape == (2, 20)
    assert d["mvpe_probe_rel_l2"].shape == (2, 20)
    assert np.max(np.abs(d["frame_rel_l2"])) == pytest.approx(0.0)
    assert np.max(np.abs(d["frame_rmse"])) == pytest.approx(0.0)
    assert np.max(np.abs(d["tke_contrib_rel_l2"])) == pytest.approx(0.0)
    assert np.max(np.abs(d["mvpe_probe_rel_l2"])) == pytest.approx(0.0)
    assert np.allclose(d["tke_contrib_ratio"], 1.0, rtol=1e-6, atol=1e-6)


def test_tke_diagnostic_uses_each_windows_full_future20_temporal_mean():
    target = _nontrivial_target(n=1)
    pred = target.copy()
    # Scale only the temporal fluctuation around each pixel's Future20 mean.
    uv = pred[..., :2]
    mean = uv.mean(axis=1, keepdims=True)
    pred[..., :2] = mean + 0.5 * (uv - mean)

    d = audit.window_horizon_diagnostics(pred, target)

    # Energy is quadratic in fluctuation amplitude: 0.5^2 = 0.25.
    assert np.allclose(d["tke_contrib_ratio"], 0.25, rtol=1e-5, atol=1e-5)
    assert np.allclose(d["tke_contrib_rel_l2"], 0.75, rtol=1e-5, atol=1e-5)


def test_mvpe_probe_diagnostic_reacts_only_at_frozen_probe_geometry():
    target = _nontrivial_target(n=1)
    pred = target.copy()
    ys, xs = audit.mvpe_probe_geometry(32, 64)
    pred[:, :, ys, xs[0], 0] += 0.4

    d = audit.window_horizon_diagnostics(pred, target)
    assert np.all(d["mvpe_probe_rel_l2"] > 0.0)

    untouched = target.copy()
    untouched[:, :, 0, 0, 0] += 10.0
    d2 = audit.window_horizon_diagnostics(untouched, target)
    assert np.max(np.abs(d2["mvpe_probe_rel_l2"])) == pytest.approx(0.0)


def test_build_long_rows_keeps_trajectory_horizon_and_paired_improvement():
    target = _nontrivial_target(n=4)
    base = target.copy()
    base[..., 0] += 0.20
    half = target.copy()
    half[..., 0] += 0.10
    full = target.copy()
    names = np.asarray(["a.h5", "a.h5", "b.h5", "b.h5"])

    rows = audit.build_trajectory_horizon_rows(
        predictions={0.0: base, 0.5: half, 1.0: full},
        target=target,
        trajectory_names=names,
    )

    assert len(rows) == 2 * 20 * 3
    sample = next(r for r in rows if r["trajectory_id"] == "a.h5" and r["horizon"] == 1 and r["alpha"] == 1.0)
    assert sample["windows"] == 2
    assert sample["frame_rel_l2"] == pytest.approx(0.0)
    assert sample["frame_rel_l2_improvement_vs_base"] == pytest.approx(1.0)


def test_horizon_stability_counts_are_trajectory_paired():
    rows = []
    for trajectory, imp in [("a", 0.10), ("b", -0.20), ("c", 0.05)]:
        rows.append({
            "trajectory_id": trajectory,
            "horizon": 1,
            "alpha": 1.0,
            "frame_rel_l2_improvement_vs_base": imp,
            "frame_rmse_improvement_vs_base": imp,
            "tke_contrib_rel_l2_improvement_vs_base": imp,
            "mvpe_probe_rel_l2_improvement_vs_base": imp,
        })
    stats = audit.horizon_trajectory_stability(rows)
    row = stats[0]
    assert row["trajectory_count"] == 3
    assert row["frame_rel_l2_improved_count"] == 2
    assert row["frame_rel_l2_improved_fraction"] == pytest.approx(2 / 3)
    assert row["frame_rel_l2_median_improvement"] == pytest.approx(0.05)
