from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from realpde_adaptive_probe import (  # noqa: E402
    AdaptiveUncertaintyHead,
    ResidualCorrector3D,
    adaptive_features,
    corrector_loss,
    evaluate_corrector_gate,
)


def test_adaptive_features_are_causal_and_have_frozen_channel_count():
    past = torch.randn(2, 20, 4, 6, 3)
    future = torch.randn(2, 20, 4, 6, 3)
    features = adaptive_features(past, future)
    assert features.shape == (2, 20, 4, 6, 42)
    changed = past.clone()
    changed[:, -1] += 10
    assert not torch.equal(features, adaptive_features(changed, future))


def test_corrector_forces_pressure_delta_to_zero():
    head = ResidualCorrector3D(in_channels=42, hidden=8, blocks=2)
    delta = head(torch.randn(2, 42, 20, 4, 6))
    assert delta.shape == (2, 3, 20, 4, 6)
    assert torch.count_nonzero(delta[:, 2]).item() == 0
    assert torch.isfinite(delta).all()


def test_uncertainty_head_clamps_sigma_and_has_two_uv_channels():
    head = AdaptiveUncertaintyHead(in_channels=15, hidden=8, blocks=2)
    sigma = head(torch.randn(2, 15, 20, 4, 6))
    assert sigma.shape == (2, 2, 20, 4, 6)
    assert float(sigma.min()) >= 1e-4
    assert float(sigma.max()) <= 1.0


def test_corrector_loss_is_finite_and_pressure_safe():
    prediction = torch.randn(2, 20, 4, 6, 3)
    target = torch.randn_like(prediction)
    delta = torch.randn_like(prediction)
    parts = corrector_loss(prediction, target, delta)
    assert set(parts) == {"point", "mse", "tke", "temporal", "grad", "p_zero", "residual_mse", "delta_penalty"}
    assert all(torch.isfinite(value) for value in parts.values())


def test_gate_requires_all_fixed_thresholds():
    good = evaluate_corrector_gate(
        baseline={"rel_l2": 1.0, "mvpe": 1.0, "tke": 1.0},
        candidate={"rel_l2": 0.97, "mvpe": 0.97, "tke": 1.01},
        trajectory_tke_degradations=[0.1, 0.14],
    )
    assert good["status"] == "PASS"
    bad = evaluate_corrector_gate(
        baseline={"rel_l2": 1.0, "mvpe": 1.0, "tke": 1.0},
        candidate={"rel_l2": 0.97, "mvpe": 0.97, "tke": 1.01},
        trajectory_tke_degradations=[0.16, 0.2, 0.18],
    )
    assert bad["status"] == "CORRECTOR_NO_GO"
