from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from sota_lr_screen import summarize_ab, validate_resume_payload, validate_run_dir  # noqa: E402


def test_validate_resume_payload_rejects_wrong_update_or_missing_optimizer():
    good = {
        "iteration": 18860,
        "feature_set": "P0-A",
        "loss_weights": {"mse": 1.0, "tke": 0.05, "rel": 0.027514, "mvpe": 0.009757},
        "optimizer_state_dict": {"state": {1: {}}, "param_groups": [{"lr": 1e-5}]},
    }

    validate_resume_payload(good)

    with pytest.raises(ValueError, match="18860"):
        validate_resume_payload(good | {"iteration": 18859})
    with pytest.raises(ValueError, match="optimizer"):
        validate_resume_payload({key: value for key, value in good.items() if key != "optimizer_state_dict"})


def test_summary_is_review_required_and_reports_decay_minus_control():
    control = {"rel_l2": 0.12, "tke": 0.50, "mvpe": 0.09}
    decay = {"rel_l2": 0.118, "tke": 0.495, "mvpe": 0.091}

    summary = summarize_ab(control, decay)

    assert summary["decision"] == "REVIEW_REQUIRED"
    assert summary["delta_decay_minus_control"] == pytest.approx(
        {"rel_l2": -0.002, "tke": -0.005, "mvpe": 0.001}
    )
    assert summary["improved_metrics"] == ["rel_l2", "tke"]


def test_validate_run_dir_requires_done_target_update_and_expected_lr(tmp_path):
    (tmp_path / "status.json").write_text(
        json.dumps({"state": "DONE", "stop_reason": "updates_complete", "update": 22960}),
        encoding="utf-8",
    )
    (tmp_path / "run_metadata.json").write_text(
        json.dumps({"initial_update": 18860, "lr": 5e-6, "optimizer_lrs": [5e-6]}),
        encoding="utf-8",
    )
    (tmp_path / "dev_22960.json").write_text(
        json.dumps({
            "update": 22960,
            "raw_errors": {"rel_l2": 0.11, "tke": 0.49, "mvpe": 0.088},
        }),
        encoding="utf-8",
    )

    metrics = validate_run_dir(tmp_path, expected_update=22960, expected_lr=5e-6)

    assert metrics == {"rel_l2": 0.11, "tke": 0.49, "mvpe": 0.088}
    with pytest.raises(ValueError, match="learning rate"):
        validate_run_dir(tmp_path, expected_update=22960, expected_lr=1e-5)
