from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from realpde_sps_teammate_exact_submit import (
    EVAL_INTERVAL,
    HEAD_LR,
    HEAD_SCOPE,
    HEAD_WEIGHT_DECAY,
    MAX_UPDATES,
    READY_GATE,
    RECIPE,
    SEED,
    SIGMA0,
    _select_best_checkpoint,
)


def test_exact_teammate_recipe_constants_are_frozen() -> None:
    assert SEED == 41
    assert MAX_UPDATES == 2000
    assert EVAL_INTERVAL == 200
    assert HEAD_LR == pytest.approx(1e-3)
    assert HEAD_WEIGHT_DECAY == pytest.approx(1e-5)
    assert SIGMA0 == pytest.approx(0.02)
    assert HEAD_SCOPE == "full53582_train50_teammate35_exact"
    assert RECIPE == "teammate35_exact"
    assert READY_GATE == "SPS_TEAMMATE_EXACT_READY"


def test_checkpoint_selection_uses_real_sps_then_narrower_width() -> None:
    rows = [
        {"iteration": 200, "best": {"sps": 45.0, "mean_width_uv": 0.020}},
        {"iteration": 400, "best": {"sps": 45.2, "mean_width_uv": 0.025}},
        {"iteration": 600, "best": {"sps": 45.2, "mean_width_uv": 0.023}},
    ]
    selected = _select_best_checkpoint(rows)
    assert selected["iteration"] == 600


def test_checkpoint_selection_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        _select_best_checkpoint([])
