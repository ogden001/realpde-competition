import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import realpde_point_local3_balanced_runner as runner


def test_balanced_loss_weight_is_frozen():
    assert runner.LAMBDA_TKE == 0.001
    assert runner.BATCH == 8
    assert runner.SEED == 20260901


def test_local3_contract_shape():
    model = runner.Local3MLP()
    x = torch.zeros(1, 20, 4, 6, 3)
    y = model(x)
    assert tuple(y.shape) == (1, 20, 4, 6, 3)
    assert model.net[0].in_features == 362
    assert model.net[-1].out_features == 40


def test_fixed_order_is_deterministic_and_shuffled():
    a = runner.fixed_order(11, 32, runner.SEED)
    b = runner.fixed_order(11, 32, runner.SEED)
    assert a == b
    assert a != list(range(32))


def test_screening_gate_requires_all_conditions():
    base = {"raw_errors": {"rel_l2": 1.0, "tke": 1.0, "mvpe": 1.0}}
    good = {"raw_errors": {"rel_l2": 0.99, "tke": 1.09, "mvpe": 0.99}}
    bad_rel = {"raw_errors": {"rel_l2": 1.01, "tke": 1.0, "mvpe": 0.99}}
    assert runner.screening_gate(base, good)[0] is True
    assert runner.screening_gate(base, bad_rel)[0] is False
