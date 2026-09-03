from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from sota_full_night import EXPECTED_N2, summarize_run, validate_resume_payload  # noqa: E402


def test_full_night_resume_requires_exact_15300_training_state():
    good = {
        "iteration": 15300,
        "feature_set": "P0-A",
        "loss_weights": EXPECTED_N2,
        "optimizer_state_dict": {"state": {1: {}}, "param_groups": [{"lr": 1e-5}]},
    }
    validate_resume_payload(good)

    with pytest.raises(ValueError, match="15300"):
        validate_resume_payload(good | {"iteration": 15299})
    with pytest.raises(ValueError, match="optimizer"):
        validate_resume_payload({key: value for key, value in good.items() if key != "optimizer_state_dict"})


def test_summary_reports_reached_milestones_without_selecting_checkpoint(tmp_path):
    (tmp_path / "status.json").write_text(
        '{"state":"TIME_CAPPED","stop_reason":"time_cap","update":40123}', encoding="utf-8"
    )
    (tmp_path / "run_metadata.json").write_text(
        '{"initial_update":15300,"train_trajectories":82,"train_windows":3383,"lr":1e-5}', encoding="utf-8"
    )
    for update in (31100, 36500, 37850):
        (tmp_path / f"model_update_{update:05d}.pth").touch()
    (tmp_path / "model_last.pth").touch()

    summary = summarize_run(tmp_path, milestones=(31100, 36500, 37850, 40560, 43260))

    assert summary["decision"] == "REVIEW_REQUIRED"
    assert summary["completed_update"] == 40123
    assert summary["reached_milestones"] == [31100, 36500, 37850]
    assert summary["missing_milestones"] == [40560, 43260]
