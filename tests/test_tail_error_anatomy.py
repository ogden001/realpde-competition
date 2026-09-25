from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "colleague_80pt"))


def load_module():
    path = ROOT / "tools" / "diagnose_tail_error_anatomy.py"
    spec = importlib.util.spec_from_file_location("tail_error_anatomy", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_module()


def test_nonwrapping_shift_replicates_border_not_wraps():
    x = np.zeros((3, 4, 3), dtype=np.float32)
    x[..., 0] = np.arange(12, dtype=np.float32).reshape(3, 4)
    y = M.nonwrapping_shift_uv(x, dx=1, dy=0)
    assert np.allclose(y[:, 0, 0], x[:, 0, 0])
    assert np.allclose(y[:, 1:, 0], x[:, :-1, 0])
    assert not np.allclose(y[:, 0, 0], x[:, -1, 0])


def test_global_shift_oracle_recovers_known_shift():
    target = np.zeros((2, 20, 5, 6, 3), dtype=np.float32)
    target[:, 19, 2, 3, 0] = 1.0
    pred = target.copy()
    pred[:, 19] = M.nonwrapping_shift_uv(target[:, 19], dx=1, dy=-1)
    result = M.global_shift_oracle(pred, target, horizon=20)
    assert result["best_dx"] == -1
    assert result["best_dy"] == 1
    assert result["sse_gain_pct"] > 99.0


def test_temporal_lag_probe_detects_one_frame_lag():
    target = np.zeros((1, 20, 2, 2, 3), dtype=np.float32)
    for h in range(20):
        target[:, h, ..., 0] = float(h)
    pred = target.copy()
    pred[:, 19] = target[:, 18]
    result = M.temporal_lag_probe(pred, target, horizon=20)
    assert result["best_offset"] == -1
    assert result["best_target_horizon"] == 19
    assert result["best_vs_current_sse_gain_pct"] > 99.0


def test_temporal_extrapolation_recovers_linear_motion():
    pred = np.zeros((1, 20, 2, 2, 3), dtype=np.float32)
    target = np.zeros_like(pred)
    pred[:, 18, ..., 0] = 1.0
    pred[:, 19, ..., 0] = 2.0
    target[:, 19, ..., 0] = 3.0
    result = M.temporal_extrapolation_oracle(pred, target, horizon=20)
    assert result["best_gamma"] == 1.0
    assert result["sse_gain_pct"] > 99.0


def test_amplitude_oracle_recovers_fluctuation_scale():
    pred = np.zeros((1, 20, 1, 1, 3), dtype=np.float32)
    target = np.zeros_like(pred)
    # Future20 mean of u is zero, F20 fluctuation is +1 in prediction and +2 in target.
    pred[:, :19, 0, 0, 0] = -1.0 / 19.0
    pred[:, 19, 0, 0, 0] = 1.0
    target[:, :19, 0, 0, 0] = -2.0 / 19.0
    target[:, 19, 0, 0, 0] = 2.0
    result = M.amplitude_oracle(pred, target, horizon=20)
    assert abs(result["scale_star"] - 2.0) < 1e-6
    assert result["sse_gain_pct"] > 99.0


def test_metric_delta_lower_is_better_sign():
    baseline = {"rel_l2_raw": 1.0, "tke_raw": 2.0, "mvpe_raw": 4.0}
    candidate = {"rel_l2_raw": 0.9, "tke_raw": 1.8, "mvpe_raw": 3.6}
    delta = M.metric_delta(candidate, baseline)
    assert all(abs(v + 10.0) < 1e-12 for v in delta.values())


def test_exp1_hashes_reused_from_frozen_tail_diagnostic():
    assert M.EXP1_BACKBONE_SHA256 == "d340effd68031aba1cd2bc610c676878fb9105dd7f81b9ce15767280e665a5c0"
    assert M.EXP1_RESIDUAL_SHA256 == "d2b4ddd0df6064d053ec918c2671a93b7d9c2a40a15afe854bb3a5badf1438fc"


def test_candidate_residual_tail_alpha_changes_only_tail():
    base = np.zeros((1, 20, 1, 1, 3), dtype=np.float32)
    final = base.copy()
    final[:, 17:, 0, 0, 0] = 1.0
    geometry = [
        {"horizon": h, "alpha_star": (2.0 if h >= 18 else 1.0)}
        for h in range(1, 21)
    ]
    out = M.candidate_residual_tail_alpha(base, final, geometry)
    assert np.allclose(out[:, :17], final[:, :17])
    assert np.allclose(out[:, 17:, 0, 0, 0], 2.0)
    assert np.allclose(out[..., 2], 0.0)
