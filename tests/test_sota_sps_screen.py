from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from sota_sps_screen import (  # noqa: E402
    CANDIDATES,
    build_bounds,
    p0a_only_config,
    summarize_candidates,
)


def test_candidate_set_is_small_and_contains_fallback_and_historical_neighbors():
    names = [row["name"] for row in CANDIDATES]

    assert len(CANDIDATES) <= 8
    assert "fallback" in names
    assert "abs0075_rel000" in names
    assert "abs0075_rel001" in names


def test_build_bounds_matches_abs_plus_rel_times_prediction_magnitude():
    prediction = np.asarray([[-2.0, 4.0]], dtype=np.float32)

    lower, upper = build_bounds(prediction, abs_width=0.5, rel_width=0.1)

    expected_half = np.asarray([[0.7, 0.9]], dtype=np.float32)
    np.testing.assert_allclose(prediction - lower, expected_half, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(upper - prediction, expected_half, rtol=0.0, atol=1e-6)


def test_p0a_only_config_disables_p0b_when_historical_checkpoint_omits_flags():
    config = p0a_only_config({"dx": 0.1, "dy": -0.2})

    assert config.include_p0_a is True
    assert config.include_p0_b is False
    assert config.dx == 0.1
    assert config.dy == -0.2


def test_p0a_only_config_rejects_checkpoint_that_explicitly_disables_p0a():
    try:
        p0a_only_config({"include_p0_a": False, "dx": 0.1, "dy": 0.2})
    except ValueError as exc:
        assert "P0-A" in str(exc)
    else:
        raise AssertionError("expected explicit non-P0-A config to be rejected")


def test_summary_never_auto_submits_and_reports_sps_delta_from_fallback():
    rows = [
        {"name": "fallback", "sps_score": 20.0, "coverage": 0.4},
        {"name": "abs0075_rel000", "sps_score": 28.0, "coverage": 0.5},
    ]

    result = summarize_candidates(rows)

    assert result["decision"] == "REVIEW_REQUIRED"
    assert result["best_local_sps_candidate"] == "abs0075_rel000"
    assert result["best_local_sps_delta_vs_fallback"] == 8.0
