import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import realpde_hybrid_cno_point_h1_scale_probe as probe


def test_scale_prediction_replays_endpoints_and_preserves_pressure():
    cno = np.zeros((2, 20, 2, 3, 3), dtype=np.float32)
    h1 = cno.copy()
    h1[..., :2] = 2.0
    assert np.array_equal(probe.scale_prediction(cno, h1, 0.0), cno)
    assert np.allclose(probe.scale_prediction(cno, h1, 1.0), h1, atol=1e-7)
    mid = probe.scale_prediction(cno, h1, 0.5)
    assert np.allclose(mid[..., :2], 1.0)
    assert np.array_equal(mid[..., 2], cno[..., 2])


def test_select_alpha_uses_tke_safety_then_rel_l2_and_tiebreaks():
    rows = [
        {"alpha": 0.0, "rel_impr_pct": 0.0, "tke_impr_pct": 0.0, "mvpe_impr_pct": 0.0},
        {"alpha": 0.2, "rel_impr_pct": 4.0, "tke_impr_pct": -4.9, "mvpe_impr_pct": 2.0},
        {"alpha": 0.4, "rel_impr_pct": 4.05, "tke_impr_pct": -5.1, "mvpe_impr_pct": 8.0},
        {"alpha": 0.3, "rel_impr_pct": 4.05, "tke_impr_pct": -4.0, "mvpe_impr_pct": 3.0},
    ]
    assert probe.select_alpha(rows) == pytest.approx(0.3)


def test_gate_treats_negative_tke_improvement_as_degradation():
    passed, details = probe.dev_gate(
        {"rel_l2": 1.0, "tke": 1.0, "mvpe": 1.0},
        {"rel_l2": 0.8, "tke": 1.04, "mvpe": 0.8},
    )
    assert passed
    assert details["improvement_pct"]["tke"] == pytest.approx(-4.0)
    passed, _ = probe.dev_gate(
        {"rel_l2": 1.0, "tke": 1.0, "mvpe": 1.0},
        {"rel_l2": 0.8, "tke": 1.06, "mvpe": 0.8},
    )
    assert not passed
