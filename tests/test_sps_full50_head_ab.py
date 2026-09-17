from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from realpde_sps_full50_head_ab import (
    BOUND_FLOOR,
    BOUND_MULT,
    PRIMARY_UPDATES,
    paired_summary,
)


def test_primary_ab_is_frozen_to_same_1600_step_and_packaged_bounds() -> None:
    assert PRIMARY_UPDATES == 1600
    assert BOUND_FLOOR == pytest.approx(0.0025)
    assert BOUND_MULT == pytest.approx(1.0)


def test_paired_summary_reports_wins_losses_and_distribution() -> None:
    summary = paired_summary(
        [
            {"delta_sps": 1.0},
            {"delta_sps": 0.5},
            {"delta_sps": 0.0},
            {"delta_sps": -0.25},
        ]
    )
    assert summary["trajectories"] == 4
    assert summary["candidate_wins"] == 2
    assert summary["ties"] == 1
    assert summary["candidate_losses"] == 1
    assert summary["median_delta_sps"] == pytest.approx(0.25)
    assert summary["mean_delta_sps"] == pytest.approx(0.3125)
    assert summary["min_delta_sps"] == pytest.approx(-0.25)
    assert summary["max_delta_sps"] == pytest.approx(1.0)


def test_paired_summary_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        paired_summary([])
