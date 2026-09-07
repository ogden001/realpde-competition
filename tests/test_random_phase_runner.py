from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from realpde_b1_p0a_n2 import (  # noqa: E402
    early_screen_gate,
    parse_eval_updates,
    summarize_window_audit,
)


def _record(epoch: int, trajectory: str, phase: int, count: int) -> dict[str, object]:
    return {
        "epoch": epoch,
        "trajectory": trajectory,
        "phase": phase,
        "window_count": count,
        "first_start": phase,
        "last_start": phase + 20 * (count - 1),
        "min_stride": 20,
        "max_stride": 20,
        "invalid_window_count": 0,
    }


def test_window_audit_summary_counts_epochs_trajectories_phases_and_windows():
    summary = summarize_window_audit(
        [_record(0, "a.h5", 7, 4), _record(0, "b.h5", 13, 5), _record(1, "a.h5", 2, 4), _record(1, "b.h5", 13, 5)],
        seed=20260901,
    )

    assert summary["seed"] == 20260901
    assert summary["epoch_count"] == 2
    assert summary["trajectory_count"] == 2
    assert summary["windows_per_epoch"] == {"0": 9, "1": 9}
    assert summary["phase_counts"] == {"0": 0, "1": 0, "2": 1, "3": 0, "4": 0, "5": 0, "6": 0, "7": 1, "8": 0, "9": 0,
                                       "10": 0, "11": 0, "12": 0, "13": 2, "14": 0, "15": 0, "16": 0, "17": 0, "18": 0, "19": 0}
    assert summary["min_window_count_per_trajectory"] == 4
    assert summary["max_window_count_per_trajectory"] == 5
    assert summary["invalid_window_count"] == 0
    assert summary["non_stride20_count"] == 0


def test_eval_updates_accepts_required_milestones_and_rejects_bad_order():
    assert parse_eval_updates("1000,2000,3000,5000,7500") == (1000, 2000, 3000, 5000, 7500)
    with pytest.raises(ValueError):
        parse_eval_updates("1000,3000,2000")
    with pytest.raises(ValueError):
        parse_eval_updates("1000,0")


def test_early_screen_stops_only_for_explicit_three_percent_negative_signal():
    baseline = {"rel_l2": 1.0, "tke": 1.0, "mvpe": 1.0}
    assert early_screen_gate(baseline, {"rel_l2": 1.04, "tke": 1.0, "mvpe": 1.04})["status"] == "STOP_EARLY"
    assert early_screen_gate(baseline, {"rel_l2": 1.02, "tke": 1.0, "mvpe": 1.01})["status"] == "CONTINUE"
    assert early_screen_gate(baseline, {"rel_l2": 1.04, "tke": 1.04, "mvpe": 1.04})["status"] == "STOP_EARLY"

