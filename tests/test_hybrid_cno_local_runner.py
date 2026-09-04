import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import realpde_hybrid_cno_local_runner as runner


def test_zero_init_preserves_direct_prediction_exactly():
    global_model = torch.nn.Identity()
    local = runner.LocalResidualBranch()
    x = torch.randn(2, 20, 8, 10, 3)
    direct = torch.randn(2, 20, 8, 10, 3)
    fused = runner.fuse_prediction(direct, x, local)
    assert torch.equal(fused, direct)
    assert tuple(local(x).shape) == (2, 20, 8, 10, 2)


def test_local_branch_uses_only_past_uv_and_zero_init_is_trainable():
    local = runner.LocalResidualBranch()
    x = torch.randn(1, 20, 8, 10, 3, requires_grad=True)
    y = local(x)
    assert y.abs().max().item() == 0.0
    assert local.output.weight.requires_grad
    assert local.input.in_channels == 2
    y.sum().backward()
    assert local.output.weight.grad is not None


def test_metric_helpers_make_matched_deltas_and_wins():
    baseline = {"a": {"rel_l2": 2.0, "tke": 1.0, "mvpe": 3.0}, "b": {"rel_l2": 1.0, "tke": 2.0, "mvpe": 3.0}}
    candidate = {"a": {"rel_l2": 1.0, "tke": 1.0, "mvpe": 4.0}, "b": {"rel_l2": 2.0, "tke": 1.0, "mvpe": 2.0}}
    delta, wins = runner.compare_trajectory_metrics(baseline, candidate)
    assert delta["a"]["rel_l2_pct"] == pytest.approx(50.0)
    assert delta["a"]["mvpe_pct"] == pytest.approx(-33.3333333333)
    assert wins == {"rel_l2": 1, "tke": 1, "mvpe": 1}
