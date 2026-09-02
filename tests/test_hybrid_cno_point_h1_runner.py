import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import realpde_hybrid_cno_point_h1_runner as runner


def test_head_contract_and_zero_init_exact_equivalence():
    head = runner.Local3PointHead()
    x = torch.randn(2, 20, 32, 64, 3)
    cno = torch.randn_like(x)
    y = runner.hybrid_forward(cno, x, head)
    assert tuple(y.shape) == tuple(cno.shape)
    assert torch.equal(y, cno)
    assert head.net[0].in_features == 402
    assert head.net[-1].out_features == 40


def test_backbone_is_frozen_and_not_in_optimizer():
    backbone = nn.Linear(3, 3)
    runner.freeze_backbone(backbone)
    assert all(not p.requires_grad for p in backbone.parameters())
    head = runner.Local3PointHead()
    opt = runner.make_optimizer(head, backbone)
    opt_ids = {id(p) for group in opt.param_groups for p in group["params"]}
    assert opt_ids == {id(p) for p in head.parameters()}


def test_screening_gate_uses_error_improvement_sign():
    base = {"raw_errors": {"rel_l2": 1.0, "tke": 1.0, "mvpe": 1.0}}
    candidate = {"raw_errors": {"rel_l2": 0.99, "tke": 0.90, "mvpe": 0.99}}
    passed, detail = runner.screening_gate(base, candidate)
    assert passed is True
    assert detail["improvement_pct"]["tke"] == pytest.approx(10.0)
