from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from sota_v2_adaptive_runtime import (  # noqa: E402
    AdaptiveUncertaintyHead,
    adaptive_bounds,
    flow_features,
    mf_reconstruct,
)
from realpde_sota_v2_adaptive import (  # noqa: E402
    EXPECTED_DEV_RAW,
    FIXED_GRID,
    choose_best_calibration,
    uncertainty_features,
    validate_replay_metrics,
)


def test_mf_reconstruct_matches_frozen_definition_and_zeroes_pressure():
    raw = torch.randn(2, 20, 4, 6, 5)
    pred = mf_reconstruct(raw)
    mean = raw[..., :2].mean(dim=1, keepdim=True).expand(-1, 20, -1, -1, -1)
    fluct = raw[..., 2:4] - raw[..., 2:4].mean(dim=1, keepdim=True)
    assert torch.allclose(pred[..., :2], mean + fluct)
    assert torch.count_nonzero(pred[..., 2]).item() == 0


def test_uncertainty_features_are_exactly_raw3_plus_flow12():
    past = torch.randn(2, 20, 4, 6, 3)
    pred = torch.randn(2, 20, 4, 6, 3)
    features = uncertainty_features(past, pred)
    assert features.shape == (2, 20, 4, 6, 15)
    assert torch.equal(features[..., :3], past)
    assert torch.allclose(features[..., 3:], flow_features(pred[..., :2]))


def test_fresh_head_starts_at_sigma_002_and_bounds_have_zero_pressure_width():
    head = AdaptiveUncertaintyHead()
    sigma = head(torch.randn(1, 15, 20, 4, 6)).permute(0, 2, 3, 4, 1)
    assert torch.allclose(sigma, torch.full_like(sigma, 0.02), atol=1e-6)
    pred = torch.randn(1, 20, 4, 6, 3)
    pred[..., 2] = 0
    lower, upper = adaptive_bounds(pred, sigma, floor=0.0025, mult=1.5)
    assert torch.count_nonzero(upper[..., 2] - lower[..., 2]).item() == 0
    assert torch.all(lower <= pred)
    assert torch.all(pred <= upper)


def test_fixed_grid_is_exactly_28_rows():
    assert len(FIXED_GRID) == 28
    assert FIXED_GRID[0] == (0.0, 0.5)
    assert FIXED_GRID[-1] == (0.0075, 4.0)


def test_calibration_selection_prefers_sps_then_narrower_width():
    rows = [
        {"floor": 0.0, "mult": 1.0, "sps": 41.0, "mean_width_uv": 0.03},
        {"floor": 0.0025, "mult": 1.0, "sps": 42.0, "mean_width_uv": 0.04},
        {"floor": 0.005, "mult": 0.5, "sps": 42.0, "mean_width_uv": 0.025},
    ]
    best = choose_best_calibration(rows)
    assert best["floor"] == 0.005
    assert best["mult"] == 0.5


def test_validation_replay_metrics_are_hard_guarded():
    validate_replay_metrics(EXPECTED_DEV_RAW, atol=1e-8)
    bad = dict(EXPECTED_DEV_RAW)
    bad["rel_l2"] += 1e-3
    with pytest.raises(ValueError, match="replay metric mismatch"):
        validate_replay_metrics(bad, atol=1e-5)
