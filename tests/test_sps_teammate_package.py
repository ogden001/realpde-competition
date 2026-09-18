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


def test_teammate_calibration_accepts_go_or_no_go_but_requires_frozen_grid() -> None:
    assert validate_teammate_calibration(
        {"gate": "SPS_TEAMMATE_GO", "best": {"floor": 0.0025, "mult": 1.0}}
    ) == (0.0025, 1.0, "SPS_TEAMMATE_GO")

    assert validate_teammate_calibration(
        {"gate": "SPS_TEAMMATE_NO_GO", "best": {"floor": 0.0025, "mult": 1.0}}
    ) == (0.0025, 1.0, "SPS_TEAMMATE_NO_GO")

    with pytest.raises(ValueError):
        validate_teammate_calibration(
            {"gate": "UNKNOWN", "best": {"floor": 0.0025, "mult": 1.0}}
        )
    with pytest.raises(ValueError):
        validate_teammate_calibration(
            {"gate": "SPS_TEAMMATE_GO", "best": {"floor": 0.003, "mult": 1.0}}
        )


def test_exact_train50_teammate_head_provenance() -> None:
    sha = "abc123"
    meta = {
        "head_scope": "full53582_train50_teammate35_exact",
        "backbone_checkpoint_iteration": 53582,
        "backbone_checkpoint_sha256": sha,
        "train_windows": 2052,
        "train_trajectories": 50,
        "dev_windows": 659,
        "dev_trajectories": 16,
        "window_mode": "fixed",
        "recipe": "teammate35_exact",
        "loss": "masked_gaussian_nll_nonzero_uv",
        "seed": 41,
        "max_updates": 2000,
        "eval_interval": 200,
        "selected_updates": 1800,
    }
    assert (
        validate_teammate_head_provenance(meta, full_sha=sha)
        == "full53582_train50_teammate35_exact"
    )


def test_exact_calibration_ready_gate_is_packageable() -> None:
    assert validate_teammate_calibration(
        {
            "gate": "SPS_TEAMMATE_EXACT_READY",
            "best": {"floor": 0.0025, "mult": 1.0},
        }
    ) == (0.0025, 1.0, "SPS_TEAMMATE_EXACT_READY")
