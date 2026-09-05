import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import realpde_horizon_curriculum as runner


def test_schedule_is_frozen_to_three_stages():
    assert runner.curriculum_stage(1) == {"short_horizon": 5, "short_weight": 0.75, "full_weight": 0.25}
    assert runner.curriculum_stage(1000) == {"short_horizon": 5, "short_weight": 0.75, "full_weight": 0.25}
    assert runner.curriculum_stage(1001) == {"short_horizon": 10, "short_weight": 0.5, "full_weight": 0.5}
    assert runner.curriculum_stage(2000) == {"short_horizon": 10, "short_weight": 0.5, "full_weight": 0.5}
    assert runner.curriculum_stage(2001) == {"short_horizon": 20, "short_weight": 0.0, "full_weight": 1.0}
    assert runner.curriculum_stage(3000) == {"short_horizon": 20, "short_weight": 0.0, "full_weight": 1.0}
    with pytest.raises(ValueError):
        runner.curriculum_stage(0)
    with pytest.raises(ValueError):
        runner.curriculum_stage(3001)


class FakeCore:
    @staticmethod
    def loss_parts(pred, target):
        horizon = pred.shape[1]
        value = pred.new_tensor(float(horizon))
        return {"mse": value, "rel": value * 2, "mvpe": value * 3, "tke": value * 4}


def test_tke_always_uses_full_horizon_while_reconstruction_is_curriculum_blended():
    pred = torch.zeros(2, 20, 4, 4, 3)
    target = torch.zeros_like(pred)
    loss, evidence = runner.curriculum_loss(pred, target, 1, FakeCore)
    blended_horizon = 0.75 * 5 + 0.25 * 20
    expected = (
        runner.N2_WEIGHTS["mse"] * blended_horizon
        + runner.N2_WEIGHTS["rel"] * (2 * blended_horizon)
        + runner.N2_WEIGHTS["mvpe"] * (3 * blended_horizon)
        + runner.N2_WEIGHTS["tke"] * (4 * 20)
    )
    assert float(loss) == pytest.approx(expected)
    assert evidence["tke_source"] == "full20"
    assert evidence["short_horizon"] == 5


def test_final_stage_is_exact_standard_n2():
    pred = torch.zeros(1, 20, 2, 2, 3)
    target = torch.zeros_like(pred)
    loss, evidence = runner.curriculum_loss(pred, target, 2500, FakeCore)
    full = FakeCore.loss_parts(pred, target)
    expected = sum(runner.N2_WEIGHTS[name] * full[name] for name in runner.N2_WEIGHTS)
    assert torch.equal(loss, expected)
    assert evidence["short_weight"] == 0.0


def test_gate_matches_breadth_first_thresholds():
    baseline = {"rel_l2": 1.0, "tke": 1.0, "mvpe": 1.0}
    assert runner.classify_gate(baseline, {"rel_l2": 0.96, "tke": 1.01, "mvpe": 0.96})["status"] == "PROMISING"
    assert runner.classify_gate(baseline, {"rel_l2": 0.99, "tke": 0.99, "mvpe": 0.98})["status"] == "WEAK_SIGNAL_PARKED"
    assert runner.classify_gate(baseline, {"rel_l2": 0.96, "tke": 1.03, "mvpe": 0.96})["status"] == "NO_GO"
