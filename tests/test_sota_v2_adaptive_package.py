from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from build_sota_v2_adaptive_package import (  # noqa: E402
    EXPECTED_FULL_CHECKPOINT_SHA,
    strip_mf_prefix,
    submission_source,
    validate_full_checkpoint_metadata,
)
from verify_sota_v2_adaptive_package import validate_package_outputs  # noqa: E402


def test_expected_full_checkpoint_sha_matches_recorded_digest():
    assert EXPECTED_FULL_CHECKPOINT_SHA == "f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce"
    assert len(EXPECTED_FULL_CHECKPOINT_SHA) == 64


def test_strip_mf_prefix_requires_cno_wrapped_state():
    state = {"cno.a": torch.tensor([1.0]), "cno.b": torch.tensor([2.0])}
    stripped = strip_mf_prefix(state)
    assert set(stripped) == {"a", "b"}
    with pytest.raises(ValueError, match="MF checkpoint"):
        strip_mf_prefix({"a": torch.tensor([1.0])})


def test_full_checkpoint_guard_is_exact():
    payload = {"iteration": 53582, "feature_set": "P0-A", "model_state_dict": {"cno.a": torch.tensor([1.0])}}
    validate_full_checkpoint_metadata(payload, EXPECTED_FULL_CHECKPOINT_SHA, EXPECTED_FULL_CHECKPOINT_SHA)
    with pytest.raises(ValueError, match="iteration"):
        validate_full_checkpoint_metadata(
            payload | {"iteration": 53581}, EXPECTED_FULL_CHECKPOINT_SHA, EXPECTED_FULL_CHECKPOINT_SHA
        )


def test_submission_source_is_mf_cno_and_never_mentions_corrector():
    source = submission_source()
    assert "out_dim=5" in source
    assert "mf_forward" in source
    assert "ResidualCorrector3D" not in source


def test_package_output_validator_enforces_parity_and_interval_invariants():
    pred = np.zeros((1, 20, 32, 64, 3), dtype=np.float32)
    lower = pred.copy(); upper = pred.copy()
    lower[..., :2] -= 0.01; upper[..., :2] += 0.01
    report = validate_package_outputs(
        pred,
        {"prediction": pred.copy(), "lower": lower, "upper": upper},
        tolerance=1e-6,
    )
    assert report["max_abs_prediction_diff"] == 0.0
    bad = {"prediction": pred.copy(), "lower": lower, "upper": upper.copy()}
    bad["upper"][..., 2] = 0.01
    with pytest.raises(ValueError, match="pressure interval"):
        validate_package_outputs(pred, bad, tolerance=1e-6)
