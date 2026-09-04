from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from realpde_p0a_submission_smoke import check_prediction_bundle  # noqa: E402


def _bundle() -> dict[str, np.ndarray]:
    prediction = np.asarray([[[[[0.5, -0.25, 0.0]]]]], dtype=np.float32)
    half = 0.0075 + 0.02 * np.abs(prediction)
    return {
        "prediction": prediction,
        "lower": (prediction - half).astype(np.float32),
        "upper": (prediction + half).astype(np.float32),
    }


def test_check_prediction_bundle_accepts_expected_abs_rel_bounds():
    result = check_prediction_bundle(_bundle(), bound_abs=0.0075, bound_rel=0.02)

    assert result["shape"] == [1, 1, 1, 1, 3]
    assert result["dtype"] == "float32"
    assert result["finite"] is True
    assert result["pressure_zero"] is True
    assert result["bounds_match"] is True


def test_check_prediction_bundle_rejects_missing_explicit_bounds():
    with pytest.raises(ValueError, match="lower/upper"):
        check_prediction_bundle({"prediction": _bundle()["prediction"]}, bound_abs=0.0075, bound_rel=0.02)


def test_check_prediction_bundle_rejects_wrong_bounds_formula():
    bundle = _bundle()
    bundle["upper"] = bundle["upper"] + np.float32(0.01)

    with pytest.raises(ValueError, match="bounds do not match"):
        check_prediction_bundle(bundle, bound_abs=0.0075, bound_rel=0.02)
