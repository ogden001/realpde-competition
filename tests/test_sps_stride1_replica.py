from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from realpde_sps_stride1_replica import (  # noqa: E402
    CURRENT_DEV_MEAN_WIDTH,
    CURRENT_DEV_SPS,
    EXPECTED_PHASE1_DENSE_WINDOWS,
    EXPECTED_FULL_DENSE_WINDOWS,
    evaluate_replica_gate,
)
from build_sota_v2_adaptive_package import (  # noqa: E402
    validate_calibration_gate,
    validate_head_provenance,
)


def test_stride1_replica_freezes_expected_dense_window_counts():
    assert EXPECTED_PHASE1_DENSE_WINDOWS == 40_488
    assert EXPECTED_FULL_DENSE_WINDOWS == 66_755


def test_replica_gate_requires_at_least_one_point_five_sps_gain():
    no_go = evaluate_replica_gate(
        candidate_sps=CURRENT_DEV_SPS + 1.49,
        candidate_mean_width=CURRENT_DEV_MEAN_WIDTH,
    )
    assert no_go["status"] == "SPS_REPLICA_NO_GO"
    assert no_go["sps_gain_ok"] is False

    go = evaluate_replica_gate(
        candidate_sps=CURRENT_DEV_SPS + 1.50,
        candidate_mean_width=CURRENT_DEV_MEAN_WIDTH,
    )
    assert go["status"] == "SPS_REPLICA_GO"
    assert go["sps_gain_ok"] is True


def test_replica_gate_rejects_gain_that_depends_on_large_width_increase():
    result = evaluate_replica_gate(
        candidate_sps=CURRENT_DEV_SPS + 2.0,
        candidate_mean_width=CURRENT_DEV_MEAN_WIDTH * 1.151,
    )
    assert result["status"] == "SPS_REPLICA_NO_GO"
    assert result["width_guard_ok"] is False


def test_package_accepts_full_specific_head_only_when_it_matches_full_backbone():
    full_sha = "f" * 64
    metadata = {
        "head_scope": "full_specific",
        "updates": 1400,
        "backbone_checkpoint_iteration": 53582,
        "backbone_checkpoint_sha256": full_sha,
        "train_windows": 66755,
        "train_trajectories": 82,
        "window_mode": "dense_all",
    }
    assert validate_head_provenance(metadata, full_sha=full_sha) == "full_specific"

    broken = dict(metadata)
    broken["backbone_checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="full-specific head backbone SHA mismatch"):
        validate_head_provenance(broken, full_sha=full_sha)


def test_package_keeps_legacy_validation_head_compatibility():
    metadata = {
        "updates": 1400,
        "validation_checkpoint_iteration": 32500,
        "train_windows": 2052,
    }
    assert validate_head_provenance(metadata, full_sha="f" * 64) == "legacy_validation"


def test_package_accepts_replica_gate_and_legacy_adaptive_gate_only():
    assert validate_calibration_gate({"gate": "ADAPTIVE_GO"}) == "ADAPTIVE_GO"
    assert validate_calibration_gate({"gate": "SPS_REPLICA_GO"}) == "SPS_REPLICA_GO"
    with pytest.raises(ValueError, match="calibration gate is not GO"):
        validate_calibration_gate({"gate": "SPS_REPLICA_NO_GO"})
