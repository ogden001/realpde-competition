from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from full43260_base_adaptive_runtime import (  # noqa: E402
    adaptive_bounds,
    AdaptiveUncertaintyHead,
)


def test_adaptive_bounds_use_frozen_uv_formula_and_zero_pressure():
    prediction = torch.randn(1, 20, 32, 64, 3)
    sigma = torch.rand(1, 20, 32, 64, 2) + 0.01
    lower, upper = adaptive_bounds(prediction, sigma, floor=0.0025, mult=1.0)
    half = 0.0025 + sigma
    expected_half = torch.cat([half, torch.zeros_like(half[..., :1])], dim=-1)
    assert torch.allclose(prediction - lower, expected_half)
    assert torch.allclose(upper - prediction, expected_half)
    assert torch.count_nonzero(upper[..., 2] - lower[..., 2]).item() == 0


def test_adaptive_head_matches_v5_shape_and_positive_sigma():
    torch.manual_seed(0)
    head = AdaptiveUncertaintyHead(hidden=32, blocks=2).eval()
    features = torch.randn(1, 15, 20, 4, 6)
    sigma = head(features)
    assert sigma.shape == (1, 2, 20, 4, 6)
    assert torch.isfinite(sigma).all()
    assert torch.all(sigma > 0)


def test_package_source_never_mentions_corrector_path():
    source = (Path(__file__).resolve().parents[1] / "tools" / "full43260_base_adaptive_runtime.py")
    if not source.exists():
        pytest.fail("package runtime has not been implemented")
    assert "ResidualCorrector3D" not in source.read_text(encoding="utf-8")
