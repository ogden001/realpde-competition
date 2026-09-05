import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import realpde_arch_a2_multiscale as runner


def test_zero_init_preserves_direct_prediction_and_pressure_exactly():
    branch = runner.CoarseResidualBranch(hidden=8)
    x = torch.randn(2, 20, 8, 10, 3)
    direct = torch.randn(2, 20, 8, 10, 3)
    fused = runner.fuse_prediction(direct, x, branch)
    assert torch.equal(fused, direct)
    assert torch.equal(fused[..., 2], direct[..., 2])
    assert tuple(branch(x).shape) == (2, 20, 8, 10, 2)


def test_coarse_branch_uses_only_uv_and_really_downsamples_spatially():
    branch = runner.CoarseResidualBranch(hidden=4)
    with torch.no_grad():
        branch.output.weight.normal_(mean=0.0, std=0.01)
        branch.output.bias.zero_()
    x = torch.randn(1, 20, 8, 10, 3)
    x_changed_pressure = x.clone()
    x_changed_pressure[..., 2] += 1000.0
    y = branch(x)
    y_changed_pressure = branch(x_changed_pressure)
    assert torch.equal(y, y_changed_pressure)
    assert branch.last_coarse_shape == (20, 4, 5)


def test_optimizer_contains_global_and_coarse_parameters_exactly_once():
    global_model = torch.nn.Linear(3, 3)
    branch = runner.CoarseResidualBranch(hidden=4)
    optimizer = runner.make_optimizer(global_model, branch, lr=1e-5)
    expected_ids = {id(p) for p in global_model.parameters()} | {id(p) for p in branch.parameters()}
    actual_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert set(actual_ids) == expected_ids
    assert len(actual_ids) == len(set(actual_ids))
    assert runner.optimizer_audit(global_model, branch, optimizer)["passed"] is True


def test_gate_requires_rel_and_mvpe_three_percent_with_tke_protected():
    baseline = {"rel_l2": 1.0, "tke": 1.0, "mvpe": 1.0}
    good = {"rel_l2": 0.96, "tke": 1.01, "mvpe": 0.96}
    weak = {"rel_l2": 0.98, "tke": 0.99, "mvpe": 0.96}
    bad_tke = {"rel_l2": 0.96, "tke": 1.03, "mvpe": 0.96}

    improvements = runner.aggregate_improvements(baseline, good)
    assert improvements == pytest.approx({"rel_l2": 4.0, "tke": -1.0, "mvpe": 4.0})
    assert runner.classify_gate(baseline, good)["status"] == "PROMISING"
    assert runner.classify_gate(baseline, weak)["status"] == "WEAK_SIGNAL_PARKED"
    assert runner.classify_gate(baseline, bad_tke)["status"] == "NO_GO"
