from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from realpde_random_phase_control_gate import control_gate  # noqa: E402


def test_control_gate_allows_small_matched_control_deviations():
    baseline = {"rel_l2": 0.2, "tke": 0.65, "mvpe": 0.16}
    candidate = {"rel_l2": 0.209, "tke": 0.70, "mvpe": 0.167}

    assert control_gate(baseline, candidate)["status"] == "CONTINUE"


def test_control_gate_stops_for_registered_rel_mvpe_or_tke_limits():
    baseline = {"rel_l2": 0.2, "tke": 0.65, "mvpe": 0.16}

    assert control_gate(baseline, {"rel_l2": 0.211, "tke": 0.65, "mvpe": 0.16})["status"] == "STOP_REVIEW_REQUIRED"
    assert control_gate(baseline, {"rel_l2": 0.2, "tke": 0.716, "mvpe": 0.16})["status"] == "STOP_REVIEW_REQUIRED"
