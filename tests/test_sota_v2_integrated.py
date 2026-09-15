from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import realpde_sota_v2_integrated as sota  # noqa: E402


def test_stage_a_and_b_are_frozen_at_30000_boundary():
    assert sota.stage_config(30_000) == {"stage": "A", "lr": 1e-5, "extra_rel": 0.0}
    assert sota.stage_config(30_001) == {"stage": "B", "lr": 3e-6, "extra_rel": 0.027514}
    assert sota.stage_config(35_000)["stage"] == "B"
    with pytest.raises(ValueError):
        sota.stage_config(35_001)


def test_protocol_requires_effective_batch_8_and_exact_milestones():
    base = dict(seed=20260901, final_update=35_000, milestones=list(sota.MILESTONES), workers=2)
    sota.validate_protocol(Namespace(**base, micro_batch=8, accumulation_steps=1))
    sota.validate_protocol(Namespace(**base, micro_batch=4, accumulation_steps=2))
    with pytest.raises(ValueError):
        sota.validate_protocol(Namespace(**base, micro_batch=4, accumulation_steps=1))
    with pytest.raises(ValueError):
        sota.validate_protocol(Namespace(**(base | {"milestones": [30_000, 35_000]}),
                                         micro_batch=8, accumulation_steps=1))


def test_lift_expansion_preserves_three_direct_channels_and_zero_inits_p0a():
    source = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    target = torch.randn(2, 20, 2)
    result = sota.expand_lift_weight(source, target)
    assert torch.equal(result[:, :3], source)
    assert torch.count_nonzero(result[:, 3:]) == 0


def test_lift_expansion_keeps_existing_p0a_checkpoint_exactly():
    source = torch.randn(2, 20, 2)
    target = torch.randn_like(source)
    assert torch.equal(sota.expand_lift_weight(source, target), source)


def test_project_expansion_maps_direct_projection_to_both_mf_heads():
    source = torch.randn(3, 4, 1, 1, 1)
    target = torch.zeros(5, 4, 1, 1, 1)
    out = sota.expand_project_parameter(source, target)
    assert torch.equal(out[:2], source[:2])
    assert torch.equal(out[2:4], source[:2])
    assert torch.equal(out[4], source[2])

    source_bias = torch.randn(3)
    target_bias = torch.zeros(5)
    out_bias = sota.expand_project_parameter(source_bias, target_bias)
    assert torch.equal(out_bias[:2], source_bias[:2])
    assert torch.equal(out_bias[2:4], source_bias[:2])
    assert torch.equal(out_bias[4], source_bias[2])


def test_horizon_error_summary_reports_last_two_fraction_exactly():
    target = np.zeros((1, 20, 1, 1, 3), dtype=np.float32)
    pred = np.zeros_like(target)
    pred[:, 18, ..., 0] = 3.0
    pred[:, 19, ..., 0] = 4.0
    summary = sota.horizon_error_summary(pred, target)
    assert summary["total_squared_error"] == 25.0
    assert summary["h19"]["fraction"] == pytest.approx(9 / 25)
    assert summary["h20"]["fraction"] == pytest.approx(16 / 25)
    assert summary["h19_h20"]["fraction"] == pytest.approx(1.0)


def test_horizon_error_summary_rejects_non_future20():
    x = np.zeros((1, 19, 1, 1, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        sota.horizon_error_summary(x, x)
