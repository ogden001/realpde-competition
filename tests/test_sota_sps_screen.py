from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from sota_sps_screen import CANDIDATES, build_bounds, summarize_candidates  # noqa: E402


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
    np.testing.assert_allclose(prediction - lower, expected_half)
    np.testing.assert_allclose(upper - prediction, expected_half)


def test_summary_never_auto_submits_and_reports_sps_delta_from_fallback():
    rows = [
        {"name": "fallback", "sps_score": 20.0, "coverage": 0.4},
        {"name": "abs0075_rel000", "sps_score": 28.0, "coverage": 0.5},
    ]

    result = summarize_candidates(rows)

    assert result["decision"] == "REVIEW_REQUIRED"
    assert result["best_local_sps_candidate"] == "abs0075_rel000"
    assert result["best_local_sps_delta_vs_fallback"] == 8.0
