from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from sps_teammate_uncertainty_runtime import (
    TeammateUncertaintyHead,
    future_feature_count,
    masked_gaussian_nll_from_log_std,
    teammate_future_features,
)
from realpde_sps_teammate_final import (
    CURRENT_DEV_MEAN_WIDTH,
    CURRENT_DEV_SPS,
    MIN_DEV_SPS_GAIN,
    evaluate_phase_a_gate,
    execution_policy,
)


def test_teammate_future_features_are_exactly_35_channels() -> None:
    past = torch.zeros(2, 20, 32, 64, 3)
    prediction = torch.zeros(2, 20, 32, 64, 3)
    features = teammate_future_features(past, prediction)
    assert future_feature_count(include_pressure=True) == 35
    assert features.shape == (2, 20, 32, 64, 35)


def test_future_features_include_linear_and_prediction_deltas() -> None:
    past = torch.zeros(1, 20, 2, 3, 3)
    past[:, -2, ..., 0] = 1.0
    past[:, -1, ..., 0] = 3.0
    prediction = torch.zeros(1, 20, 2, 3, 3)
    features = teammate_future_features(past, prediction)

    # Two 13-channel augmented blocks, then linear(3), base-last(3), base-linear(3).
    linear_u = features[..., 26]
    base_minus_last_u = features[..., 29]
    base_minus_linear_u = features[..., 32]
    assert torch.allclose(linear_u[:, 0], torch.full_like(linear_u[:, 0], 3.1))
    assert torch.allclose(linear_u[:, -1], torch.full_like(linear_u[:, -1], 5.0))
    assert torch.allclose(base_minus_last_u, torch.full_like(base_minus_last_u, -3.0))
    assert torch.allclose(base_minus_linear_u, -linear_u)


def test_teammate_head_matches_packaged_architecture_contract() -> None:
    head = TeammateUncertaintyHead(hidden=32, blocks=2, dropout=0.0, include_pressure=True)
    assert tuple(head.input_norm.normalized_shape) == (35,)
    assert head.net[0].in_channels == 35
    assert head.net[0].out_channels == 32

    features = torch.zeros(1, 20, 4, 5, 35)
    log_std = head.forward_features(features)
    assert log_std.shape == (1, 20, 4, 5, 2)
    assert torch.isfinite(log_std).all()


def test_phase_a_gate_requires_plus_1_5_sps_and_width_guard() -> None:
    go = evaluate_phase_a_gate(
        candidate_sps=CURRENT_DEV_SPS + MIN_DEV_SPS_GAIN,
        candidate_mean_width=CURRENT_DEV_MEAN_WIDTH,
        prediction_parity_max_abs=0.0,
    )
    assert go["status"] == "SPS_TEAMMATE_GO"

    no_gain = evaluate_phase_a_gate(
        candidate_sps=CURRENT_DEV_SPS + MIN_DEV_SPS_GAIN - 1e-3,
        candidate_mean_width=CURRENT_DEV_MEAN_WIDTH,
        prediction_parity_max_abs=0.0,
    )
    assert no_gain["status"] == "SPS_TEAMMATE_NO_GO"

    too_wide = evaluate_phase_a_gate(
        candidate_sps=CURRENT_DEV_SPS + MIN_DEV_SPS_GAIN + 1.0,
        candidate_mean_width=CURRENT_DEV_MEAN_WIDTH * 1.151,
        prediction_parity_max_abs=0.0,
    )
    assert too_wide["status"] == "SPS_TEAMMATE_NO_GO"

    parity_fail = evaluate_phase_a_gate(
        candidate_sps=CURRENT_DEV_SPS + MIN_DEV_SPS_GAIN + 1.0,
        candidate_mean_width=CURRENT_DEV_MEAN_WIDTH,
        prediction_parity_max_abs=2e-7,
    )
    assert parity_fail["status"] == "SPS_TEAMMATE_NO_GO"


def test_phase_b_always_executes_but_submission_recommendation_follows_phase_a_gate() -> None:
    go = execution_policy("SPS_TEAMMATE_GO")
    assert go == {"run_phase_b": True, "submission_recommended": True}

    no_go = execution_policy("SPS_TEAMMATE_NO_GO")
    assert no_go == {"run_phase_b": True, "submission_recommended": False}

    with pytest.raises(ValueError):
        execution_policy("UNKNOWN")


def test_phase_a_gate_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        evaluate_phase_a_gate(
            candidate_sps=float("nan"),
            candidate_mean_width=CURRENT_DEV_MEAN_WIDTH,
            prediction_parity_max_abs=0.0,
        )


def test_masked_gaussian_nll_ignores_unmeasured_zero_targets() -> None:
    target = torch.tensor([[[[[0.0, 2.0]]]]])
    prediction = torch.tensor([[[[[100.0, 1.0]]]]])
    log_std = torch.zeros_like(target)
    loss = masked_gaussian_nll_from_log_std(target, prediction, log_std)
    # u is masked because target==0; v contributes 0.5 * (1 / 1)^2.
    assert loss.item() == pytest.approx(0.5)


def test_masked_gaussian_nll_rejects_all_unmeasured_batch() -> None:
    target = torch.zeros(1, 1, 1, 1, 2)
    prediction = torch.ones_like(target)
    log_std = torch.zeros_like(target)
    with pytest.raises(ValueError, match="no measured"):
        masked_gaussian_nll_from_log_std(target, prediction, log_std)
