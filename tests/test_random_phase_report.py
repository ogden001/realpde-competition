from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from realpde_random_phase_report import (  # noqa: E402
    build_comparison_rows,
    classify_formal_gate,
    relative_improvements,
)


def test_relative_improvements_are_positive_when_random_phase_has_lower_errors():
    baseline = {"rel_l2": 1.0, "tke": 1.0, "mvpe": 1.0}
    candidate = {"rel_l2": 0.97, "tke": 1.01, "mvpe": 0.96}

    assert relative_improvements(baseline, candidate) == {
        "rel_l2": 3.0,
        "tke": -1.0,
        "mvpe": 4.0,
    }


def test_comparison_rows_include_both_arms_and_relative_improvements():
    rw00 = {"1000": {"rel_l2": 1.0, "tke": 1.0, "mvpe": 1.0}}
    rw01 = {"1000": {"rel_l2": 0.9, "tke": 1.0, "mvpe": 0.95}}

    rows = build_comparison_rows(rw00, rw01)

    assert rows == [{
        "update": 1000,
        "rw00_rel_l2": 1.0,
        "rw00_tke": 1.0,
        "rw00_mvpe": 1.0,
        "rw01_rel_l2": 0.9,
        "rw01_tke": 1.0,
        "rw01_mvpe": 0.95,
        "rel_l2_improvement_percent": 10.0,
        "tke_improvement_percent": 0.0,
        "mvpe_improvement_percent": 5.0,
    }]


def test_formal_gate_returns_go_for_any_registered_two_percent_metric():
    baseline = {"rel_l2": 1.0, "tke": 1.0, "mvpe": 1.0}
    candidate = {"rel_l2": 0.98, "tke": 1.0, "mvpe": 1.0}
    assert classify_formal_gate(baseline, candidate, rel_wins=9, mvpe_wins=8)["status"] == "GO"


def test_formal_gate_returns_borderline_for_stable_small_wins():
    baseline = {"rel_l2": 1.0, "tke": 1.0, "mvpe": 1.0}
    candidate = {"rel_l2": 0.985, "tke": 1.005, "mvpe": 0.985}
    result = classify_formal_gate(baseline, candidate, rel_wins=10, mvpe_wins=9)
    assert result["status"] == "BORDERLINE_KEEP"

