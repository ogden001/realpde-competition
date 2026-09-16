from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from build_sota_v2_teammate_package import (
    validate_teammate_calibration,
    validate_teammate_head_provenance,
)


def test_full_specific_teammate_head_provenance() -> None:
    sha = "abc123"
    meta = {
        "head_scope": "full_specific_teammate35",
        "backbone_checkpoint_iteration": 53582,
        "backbone_checkpoint_sha256": sha,
        "train_windows": 3383,
        "train_trajectories": 82,
        "window_mode": "fixed",
        "recipe": "teammate35",
        "selected_updates": 1800,
    }
    assert validate_teammate_head_provenance(meta, full_sha=sha) == "full_specific_teammate35"


def test_full_specific_teammate_head_rejects_wrong_backbone_sha() -> None:
    meta = {
        "head_scope": "full_specific_teammate35",
        "backbone_checkpoint_iteration": 53582,
        "backbone_checkpoint_sha256": "wrong",
        "train_windows": 3383,
        "train_trajectories": 82,
        "window_mode": "fixed",
        "recipe": "teammate35",
        "selected_updates": 1800,
    }
    with pytest.raises(ValueError):
        validate_teammate_head_provenance(meta, full_sha="expected")


def test_teammate_calibration_requires_go_and_frozen_grid() -> None:
    assert validate_teammate_calibration(
        {"gate": "SPS_TEAMMATE_GO", "best": {"floor": 0.0025, "mult": 1.0}}
    ) == (0.0025, 1.0)

    with pytest.raises(ValueError):
        validate_teammate_calibration(
            {"gate": "SPS_TEAMMATE_NO_GO", "best": {"floor": 0.0025, "mult": 1.0}}
        )
    with pytest.raises(ValueError):
        validate_teammate_calibration(
            {"gate": "SPS_TEAMMATE_GO", "best": {"floor": 0.003, "mult": 1.0}}
        )
