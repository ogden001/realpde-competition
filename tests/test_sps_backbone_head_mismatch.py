from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from realpde_sps_backbone_head_mismatch import (  # noqa: E402
    COMBINATION_KEYS,
    compute_mismatch,
    resolve_backbone_a,
)


def test_resolve_backbone_a_prefers_exact_30000(tmp_path: Path):
    for iteration in (25000, 30000, 31000):
        (tmp_path / f"model_update_{iteration:05d}.pth").touch()

    selected = resolve_backbone_a(tmp_path, backbone_b_iteration=32500)

    assert selected.path.name == "model_update_30000.pth"
    assert selected.iteration == 30000
    assert selected.selection_rule == "exact_30000"


def test_resolve_backbone_a_falls_back_to_nearest_earlier_checkpoint(tmp_path: Path):
    for iteration in (25000, 31000):
        (tmp_path / f"model_update_{iteration:05d}.pth").touch()

    selected = resolve_backbone_a(tmp_path, backbone_b_iteration=32500)

    assert selected.path.name == "model_update_31000.pth"
    assert selected.iteration == 31000
    assert selected.selection_rule == "nearest_earlier"


def test_resolve_backbone_a_rejects_missing_earlier_checkpoint(tmp_path: Path):
    (tmp_path / "model_update_35000.pth").touch()

    with pytest.raises(ValueError, match="earlier checkpoint"):
        resolve_backbone_a(tmp_path, backbone_b_iteration=32500)


def test_compute_mismatch_uses_matched_b_as_reference():
    rows = {
        "A_headA_Acal": {"sps": 40.0},
        "B_headB_Bcal": {"sps": 45.0},
        "B_headA_Acal": {"sps": 39.0},
        "B_headA_Bcal": {"sps": 44.0},
    }

    result = compute_mismatch(rows)

    assert tuple(rows) == COMBINATION_KEYS
    assert result == {
        "mismatch_total": 6.0,
        "mismatch_after_recalibration": 1.0,
    }
