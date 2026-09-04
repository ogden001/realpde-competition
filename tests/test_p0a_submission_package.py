from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from build_p0a_n2_submission import (  # noqa: E402
    generate_submission_module,
    validate_bounds,
    validate_checkpoint_for_submission,
)


N2 = {"mse": 1.0, "tke": 0.05, "rel": 0.027514, "mvpe": 0.009757}


def _checkpoint(path: Path, iteration: int = 43260) -> None:
    torch.save(
        {
            "model_state_dict": {"x": torch.ones(1)},
            "iteration": iteration,
            "feature_set": "P0-A",
            "feature_config": {"dx": 0.1, "dy": 0.2},
            "loss_weights": N2,
        },
        path,
    )


def test_checkpoint_validation_accepts_explicit_required_iteration(tmp_path):
    checkpoint = tmp_path / "model.pth"
    _checkpoint(checkpoint, 43260)

    payload = validate_checkpoint_for_submission(checkpoint, required_iteration=43260)

    assert payload["iteration"] == 43260
    with pytest.raises(ValueError, match="40560"):
        validate_checkpoint_for_submission(checkpoint, required_iteration=40560)


def test_bounds_must_be_nonnegative_and_supplied_as_a_pair():
    assert validate_bounds(0.0075, 0.0) == (0.0075, 0.0)
    assert validate_bounds(None, None) is None
    with pytest.raises(ValueError, match="together"):
        validate_bounds(0.0075, None)
    with pytest.raises(ValueError, match="non-negative"):
        validate_bounds(-0.1, 0.0)


def test_generated_submission_explicitly_returns_abs_rel_bounds(tmp_path):
    generate_submission_module(tmp_path, bound_abs=0.0075, bound_rel=0.01)
    source = (tmp_path / "submission.py").read_text(encoding="utf-8")

    assert "_BOUND_ABS = 0.0075" in source
    assert "_BOUND_REL = 0.01" in source
    assert '"lower"' in source and '"upper"' in source
    assert "np.abs(prediction)" in source


def test_generated_submission_can_keep_fallback_no_bounds(tmp_path):
    generate_submission_module(tmp_path, bound_abs=None, bound_rel=None)
    source = (tmp_path / "submission.py").read_text(encoding="utf-8")

    assert "_RETURN_BOUNDS = False" in source
    assert "return prediction" in source
