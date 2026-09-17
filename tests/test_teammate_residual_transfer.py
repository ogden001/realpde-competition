from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import realpde_teammate_residual_transfer as transfer  # noqa: E402


def test_fixed_variants_are_baseline_half_and_full_only():
    assert transfer.ALPHAS == (0.0, 0.5, 1.0)


def test_scaled_correction_preserves_pressure_and_alpha_zero_parity():
    base = torch.randn(2, 20, 4, 6, 3)
    base[..., 2] = 0.0
    delta = torch.randn_like(base)
    delta[..., 2] = 0.0

    out0 = transfer.apply_scaled_correction(base, delta, 0.0)
    out_half = transfer.apply_scaled_correction(base, delta, 0.5)
    out1 = transfer.apply_scaled_correction(base, delta, 1.0)

    assert torch.equal(out0, base)
    assert torch.allclose(out_half[..., :2], base[..., :2] + 0.5 * delta[..., :2])
    assert torch.allclose(out1[..., :2], base[..., :2] + delta[..., :2])
    assert torch.count_nonzero(out_half[..., 2]) == 0
    assert torch.count_nonzero(out1[..., 2]) == 0


def test_scaled_correction_rejects_non_fixed_alpha():
    base = torch.zeros(1, 20, 2, 2, 3)
    delta = torch.zeros_like(base)
    with pytest.raises(ValueError):
        transfer.apply_scaled_correction(base, delta, 0.25)


def test_freeze_module_disables_gradients_and_sets_eval():
    module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    module.train()
    transfer.freeze_module(module)
    assert not module.training
    assert all(not p.requires_grad for p in module.parameters())


def test_corrector_objective_uses_frozen_teammate_weights(monkeypatch):
    parts = {
        "point": torch.tensor(1.0),
        "mse": torch.tensor(2.0),
        "tke": torch.tensor(3.0),
        "temporal": torch.tensor(4.0),
        "grad": torch.tensor(5.0),
        "p_zero": torch.tensor(6.0),
        "residual_mse": torch.tensor(7.0),
        "delta_penalty": torch.tensor(8.0),
    }
    monkeypatch.setattr(transfer, "corrector_loss", lambda base, target, delta: parts)
    value, returned = transfer.corrector_objective(torch.empty(0), torch.empty(0), torch.empty(0))
    expected = sum(transfer.CORRECTOR_LOSS_WEIGHTS[k] * v for k, v in parts.items())
    assert value.item() == pytest.approx(expected.item())
    assert returned is parts


def test_registered_baselines_are_the_frozen_sota_v2_anchors():
    assert transfer.REGISTERED_BASELINES[30_000] == pytest.approx(
        {"rel_l2": 0.105989, "tke": 0.474373, "mvpe": 0.090337}
    )
    assert transfer.REGISTERED_BASELINES[32_500] == pytest.approx(
        {"rel_l2": 0.099935, "tke": 0.469293, "mvpe": 0.075778}
    )


def test_raw_physical_errors_loads_the_supplied_official_scorer(tmp_path):
    scorer = tmp_path / "scoring.py"
    scorer.write_text(
        """
import numpy as np

def measured_channels(target):
    return 2

def rel_l2_per_sample(pred, target, c):
    assert c == 2
    return np.asarray([1.0])

def tke_rel_l2_per_sample(pred, target, c):
    assert c == 2
    return np.asarray([2.0])

def mvpe_rel_l2(pred, target):
    return 3.0
""",
        encoding="utf-8",
    )
    pred = np.zeros((1, 20, 2, 2, 3), dtype=np.float32)
    target = np.zeros_like(pred)

    assert transfer.raw_physical_errors(tmp_path, pred, target) == {
        "rel_l2": 1.0,
        "tke": 2.0,
        "mvpe": 3.0,
    }
