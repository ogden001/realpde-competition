from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from realpde_matched_phase_report import classify_matched_phase_decision  # noqa: E402


def test_matched_decision_marks_clear_double_rel_mvpe_degradation_as_no_go_candidate():
    result = classify_matched_phase_decision(
        {"rel_l2": 1.0, "tke": 1.0, "mvpe": 1.0},
        {"rel_l2": 1.051, "tke": 1.0, "mvpe": 1.051},
    )

    assert result["candidate_status"] == "NO_GO_CANDIDATE"
    assert result["final_status"] == "REVIEW_REQUIRED"


def test_matched_decision_marks_nonnegative_matched_result_as_need_long_review():
    result = classify_matched_phase_decision(
        {"rel_l2": 1.0, "tke": 1.0, "mvpe": 1.0},
        {"rel_l2": 0.99, "tke": 1.0, "mvpe": 1.0},
    )

    assert result["candidate_status"] == "NEED_LONG_7500"
    assert result["final_status"] == "REVIEW_REQUIRED"
