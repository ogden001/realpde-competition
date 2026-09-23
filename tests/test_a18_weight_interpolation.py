from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "colleague_80pt"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(ROOT / "tools"))

from eval_a18_weight_interpolation import (  # noqa: E402
    BASELINE_CURRENT80,
    BETAS,
    interpolate_residual_state,
    mechanical_gate,
    select_candidate,
)


def test_frozen_beta_set_is_small_and_predeclared() -> None:
    assert BETAS == (0.50, 0.65, 0.80, 1.00)


def test_interpolation_changes_only_corrector_state() -> None:
    current = {
        "base_model.weight": torch.tensor([1.0, 2.0]),
        "corrector.weight": torch.tensor([2.0, 4.0]),
        "corrector.buffer": torch.tensor([3], dtype=torch.int64),
    }
    target = {
        "base_model.weight": torch.tensor([1.0, 2.0]),
        "corrector.weight": torch.tensor([6.0, 8.0]),
        "corrector.buffer": torch.tensor([3], dtype=torch.int64),
    }

    state = interpolate_residual_state(current, target, 0.25)

    torch.testing.assert_close(state["base_model.weight"], current["base_model.weight"])
    torch.testing.assert_close(
        state["corrector.weight"],
        torch.tensor([3.0, 5.0]),
    )
    assert torch.equal(state["corrector.buffer"], current["corrector.buffer"])


def test_interpolation_rejects_backbone_drift() -> None:
    current = {
        "base_model.weight": torch.tensor([1.0]),
        "corrector.weight": torch.tensor([2.0]),
    }
    target = {
        "base_model.weight": torch.tensor([1.1]),
        "corrector.weight": torch.tensor([3.0]),
    }
    with pytest.raises(ValueError, match="identical frozen backbone"):
        interpolate_residual_state(current, target, 0.5)


def test_interpolation_endpoints_match_source_and_target_corrector() -> None:
    current = {
        "base_model.weight": torch.tensor([1.0]),
        "corrector.weight": torch.tensor([2.0, 4.0]),
    }
    target = {
        "base_model.weight": torch.tensor([1.0]),
        "corrector.weight": torch.tensor([6.0, 8.0]),
    }
    beta0 = interpolate_residual_state(current, target, 0.0)
    beta1 = interpolate_residual_state(current, target, 1.0)
    torch.testing.assert_close(beta0["corrector.weight"], current["corrector.weight"])
    torch.testing.assert_close(beta1["corrector.weight"], target["corrector.weight"])


def test_mechanical_gate_matches_campaign_definition() -> None:
    candidate = {
        "rel_l2_raw": BASELINE_CURRENT80["rel_l2_raw"] * 1.004,
        "tke_raw": BASELINE_CURRENT80["tke_raw"] * 0.96,
        "mvpe_raw": BASELINE_CURRENT80["mvpe_raw"] * 1.002,
    }
    assert mechanical_gate(candidate)["pass"] is True

    candidate["rel_l2_raw"] = BASELINE_CURRENT80["rel_l2_raw"] * 1.006
    assert mechanical_gate(candidate)["pass"] is False


def test_selection_prefers_best_final_est_among_gate_passing_rows() -> None:
    rows = [
        {
            "beta": 0.50,
            "final_est": 82.0,
            "mechanical_gate": {"pass": True},
        },
        {
            "beta": 0.65,
            "final_est": 82.1,
            "mechanical_gate": {"pass": True},
        },
        {
            "beta": 0.80,
            "final_est": 82.3,
            "mechanical_gate": {"pass": False},
        },
    ]
    result = select_candidate(rows)
    assert result["selected_beta"] == 0.65
    assert result["gate_passing_count"] == 2
    assert result["selection_policy"] == "highest_final_est_among_gate_passing"
