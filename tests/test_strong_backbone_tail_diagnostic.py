from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "tools" / "diagnose_strong_backbone_tail.py"
    spec = importlib.util.spec_from_file_location("strong_tail_diag", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


TAIL = load_module()


def arrays(scale: float):
    base = np.zeros((2, 20, 2, 2, 3), dtype=np.float32)
    target = np.zeros_like(base)
    target[..., 0] = 1.0
    target[..., 1] = 2.0
    final = base + scale * (target - base)
    return base, final, target


def test_residual_geometry_perfect_correction():
    base, final, target = arrays(1.0)
    rows = TAIL.residual_geometry_by_horizon(base, final, target)
    assert len(rows) == 20
    for row in rows:
        assert abs(row["delta_target_cosine"] - 1.0) < 1e-10
        assert abs(row["alpha_star"] - 1.0) < 1e-10
        assert abs(row["help_fraction"] - 1.0) < 1e-10
        assert abs(row["sse_gain_pct"] - 100.0) < 1e-10


def test_residual_geometry_detects_under_correction():
    base, final, target = arrays(0.5)
    row = TAIL.residual_geometry_by_horizon(base, final, target)[0]
    assert abs(row["delta_target_cosine"] - 1.0) < 1e-10
    assert abs(row["alpha_star"] - 2.0) < 1e-10
    assert abs(row["delta_to_target_rms_ratio"] - 0.5) < 1e-10
    assert abs(row["sse_gain_pct"] - 75.0) < 1e-10
    assert abs(row["oracle_alpha_sse_gain_pct"] - 100.0) < 1e-10


def test_residual_geometry_detects_wrong_direction():
    base, final, target = arrays(-0.25)
    row = TAIL.residual_geometry_by_horizon(base, final, target)[0]
    assert row["delta_target_cosine"] < -0.999999
    assert row["alpha_star"] < 0.0
    assert row["help_fraction"] == 0.0
    assert row["sse_gain_pct"] < 0.0


def test_tail_summary_flags_residual_gain_collapse():
    rows = []
    for h in range(1, 21):
        base_rel = 0.05 + 0.002 * h
        gain = 30.0
        cosine = 0.9
        if h == 19:
            gain = 18.0
            cosine = 0.82
        if h == 20:
            gain = 5.0
            cosine = 0.70
        final_rel = base_rel * (1.0 - gain / 100.0)
        rows.append({
            "horizon": h,
            "base_frame_rel_l2": base_rel,
            "final_frame_rel_l2": final_rel,
            "correction_gain_rel_pct": gain,
            "delta_target_cosine": cosine,
        })
    summary = TAIL.summarize_tail(rows)
    assert summary["diagnostic_flags"]["residual_effectiveness_drop_present"] is True
    assert summary["diagnostic_flags"]["residual_alignment_drop_present"] is True
    assert summary["correction_gain_drop_h18_to_h20_pp"] == 25.0


def test_exp1_checkpoint_hashes_are_frozen():
    assert TAIL.EXP1_BACKBONE_SHA256 == "d340effd68031aba1cd2bc610c676878fb9105dd7f81b9ce15767280e665a5c0"
    assert TAIL.EXP1_RESIDUAL_SHA256 == "d2b4ddd0df6064d053ec918c2671a93b7d9c2a40a15afe854bb3a5badf1438fc"
