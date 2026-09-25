from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "colleague_80pt"))


def load_module():
    path = ROOT / "tools" / "diagnose_tail_fluctuation_coherence.py"
    spec = importlib.util.spec_from_file_location("tail_fluctuation_coherence", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_module()


def make_sequence(amplitude: float = 1.0, phase_shift: int = 0, mean_bias: float = 0.0):
    x = np.zeros((2, 20, 2, 2, 3), dtype=np.float32)
    mode_a = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
    mode_b = np.asarray([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    for h in range(20):
        phase = 2.0 * np.pi * (h + phase_shift) / 20.0
        field = amplitude * (np.sin(phase) * mode_a + np.cos(phase) * mode_b)
        x[:, h, ..., 0] = mean_bias + field
        x[:, h, ..., 1] = 0.5 * mean_bias + 0.5 * field
    return x


def test_decompose_has_zero_temporal_mean():
    x = make_sequence(amplitude=2.0, mean_bias=3.0)
    mean, fluct = M.decompose(x)
    assert mean.shape == (2, 1, 2, 2, 2)
    assert np.max(np.abs(fluct.mean(axis=1))) < 1e-12


def test_identical_prediction_has_unit_amplitude_and_cosine():
    y = make_sequence()
    rows = M.fluctuation_horizon_rows(y, y)
    for row in rows:
        assert abs(row["amplitude_ratio_global"] - 1.0) < 1e-10
        assert abs(row["fluctuation_cosine_global"] - 1.0) < 1e-10
        assert abs(row["fluctuation_scale_star"] - 1.0) < 1e-10
        assert row["fluctuation_rel_l2"] < 1e-10


def test_amplitude_error_is_separated_from_coherence():
    y = make_sequence(amplitude=1.0)
    p = make_sequence(amplitude=2.0)
    rows = M.fluctuation_horizon_rows(p, y)
    usable = [r for r in rows if abs(r["target_fluct_rms"]) > 1e-6]
    for row in usable:
        assert abs(row["amplitude_ratio_global"] - 2.0) < 1e-6
        assert row["fluctuation_cosine_global"] > 0.999999
        assert abs(row["fluctuation_scale_star"] - 0.5) < 1e-6


def test_mean_bias_does_not_destroy_fluctuation_coherence():
    y = make_sequence(amplitude=1.0, mean_bias=0.0)
    p = make_sequence(amplitude=1.0, mean_bias=2.0)
    mean = M.mean_field_metrics(p, y)
    rows = M.fluctuation_horizon_rows(p, y)
    usable = [r for r in rows if abs(r["target_fluct_rms"]) > 1e-6]
    assert mean["mean_field_rel_l2_global"] > 0.1
    for row in usable:
        assert row["fluctuation_cosine_global"] > 0.999999


def test_phase_similarity_detects_shifted_temporal_phase():
    target = make_sequence(amplitude=1.0, phase_shift=0)
    pred = make_sequence(amplitude=1.0, phase_shift=-2)
    gm, wm = M.phase_similarity_matrix(pred, target)
    rows = M.best_phase_rows(gm, wm)
    # phase_shift=-2 means Pred F10 carries the phase of GT F8.
    row = rows[9]
    assert row["best_gt_horizon_global"] == 8
    assert row["best_offset_global"] == -2
    assert row["best_global_cosine"] > row["diagonal_global_cosine"]


def test_trajectory_tail_rows_cover_three_tail_horizons_per_trajectory():
    y = make_sequence()
    names = ["a.h5", "a.h5"]
    rows = M.trajectory_tail_rows(y, y, names)
    assert len(rows) == 3
    assert {row["horizon"] for row in rows} == {18, 19, 20}
    for row in rows:
        assert abs(row["fluctuation_cosine_global"] - 1.0) < 1e-10
        assert abs(row["amplitude_ratio_global"] - 1.0) < 1e-10


def test_exp1_hashes_remain_frozen():
    assert M.EXP1_BACKBONE_SHA256 == "d340effd68031aba1cd2bc610c676878fb9105dd7f81b9ce15767280e665a5c0"
    assert M.EXP1_RESIDUAL_SHA256 == "d2b4ddd0df6064d053ec918c2671a93b7d9c2a40a15afe854bb3a5badf1438fc"
