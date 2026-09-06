from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from audit_adaptive_sigma_error import (  # noqa: E402
    rankdata_average,
    safe_pearson,
    safe_spearman,
    sigma_decile_rows,
)


def test_rankdata_average_handles_ties():
    values = np.array([30.0, 10.0, 20.0, 20.0], dtype=np.float64)
    np.testing.assert_allclose(rankdata_average(values), [4.0, 1.0, 2.5, 2.5])


def test_correlations_detect_monotonic_error_ranking():
    sigma = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    error = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    assert np.isclose(safe_pearson(sigma, error), 1.0)
    assert np.isclose(safe_spearman(sigma, error), 1.0)


def test_sigma_deciles_cover_every_element_and_preserve_order():
    sigma = np.arange(1, 101, dtype=np.float64)
    error = sigma * 2.0
    inside = np.ones_like(sigma, dtype=bool)
    rows = sigma_decile_rows(sigma, error, inside, bins=10)
    assert len(rows) == 10
    assert sum(row["count"] for row in rows) == 100
    assert all(np.isclose(row["coverage"], 1.0) for row in rows)
    assert all(rows[i]["mean_sigma"] < rows[i + 1]["mean_sigma"] for i in range(9))
    assert all(rows[i]["mean_abs_error"] < rows[i + 1]["mean_abs_error"] for i in range(9))
