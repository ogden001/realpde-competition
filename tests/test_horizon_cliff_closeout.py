from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def test_tail_squared_error_contribution_uses_horizon_totals():
    from horizon_cliff_closeout import tail_squared_error_contribution

    values = np.array([1.0] * 18 + [2.0, 3.0])
    result = tail_squared_error_contribution(values)

    assert result["t19_fraction"] == pytest.approx(2 / 23)
    assert result["t20_fraction"] == pytest.approx(3 / 23)
    assert result["t19_t20_fraction"] == pytest.approx(5 / 23)
