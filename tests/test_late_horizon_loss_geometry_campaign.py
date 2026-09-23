from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import late_horizon_loss_geometry_campaign as campaign


def test_rel_dominant_loss_replaces_only_unit_mse(monkeypatch) -> None:
    pred = torch.tensor([1.0, 2.0], requires_grad=True)
    target = torch.zeros_like(pred)
    mse = (pred - target).square().mean()
    rel = (pred - target).abs().mean()
    tke = pred.sum() * 0.0 + 3.0
    control = mse + 0.25 * rel + 0.5 * tke

    def fake_historical_stage_b_loss(p, y):
        return control, {
            "mse": mse,
            "rel": rel,
            "tke": tke,
        }

    monkeypatch.setattr(
        campaign.late,
        "historical_stage_b_loss",
        fake_historical_stage_b_loss,
    )
    candidate, parts = campaign.rel_dominant_stage_b_loss(pred, target)

    assert torch.allclose(candidate - control, rel - mse)
    assert torch.allclose(parts["tke"], tke)
    assert torch.allclose(parts["candidate_total"], candidate)


def test_tail_rel_loss_uses_only_last_two_horizons() -> None:
    target = torch.zeros((1, 20, 2, 2, 3), dtype=torch.float32)
    target[..., :2] = 1.0
    pred = target.clone()
    pred[:, 18:20, ..., :2] = 2.0

    tail = campaign.tail_rel_loss(pred, target)
    first18 = campaign.first18_rel_loss(pred, target)

    assert float(tail) > 0.0
    assert float(first18) == pytest.approx(0.0, abs=1e-12)


def test_gradient_cosine_basic_geometry() -> None:
    a = (torch.tensor([1.0, 0.0]),)
    b = (torch.tensor([1.0, 0.0]),)
    c = (torch.tensor([0.0, 1.0]),)

    assert campaign.gradient_cosine(a, b) == pytest.approx(1.0)
    assert campaign.gradient_cosine(a, c) == pytest.approx(0.0)
    assert campaign.gradient_norm(a) == pytest.approx(1.0)


def test_select_probe_indices_uses_distinct_trajectories() -> None:
    refs = [
        SimpleNamespace(path=Path("a.h5"), start=0),
        SimpleNamespace(path=Path("a.h5"), start=20),
        SimpleNamespace(path=Path("b.h5"), start=0),
        SimpleNamespace(path=Path("c.h5"), start=0),
    ]
    assert campaign.select_probe_indices(refs, 3) == [0, 2, 3]


def test_diagnostic_summary_is_descriptive_not_auto_go() -> None:
    rows = [
        {
            "control_tail_cosine": 0.1,
            "candidate_tail_cosine": 0.3,
            "mse_tail_cosine": 0.0,
            "rel_tail_cosine": 0.4,
        },
        {
            "control_tail_cosine": 0.2,
            "candidate_tail_cosine": 0.25,
            "mse_tail_cosine": 0.1,
            "rel_tail_cosine": 0.2,
        },
    ]
    result = campaign.diagnostic_summary(rows)

    assert result["candidate_minus_control_tail_cosine"] > 0
    assert result["rel_minus_mse_tail_cosine"] > 0
    assert result["candidate_tail_alignment_wins"] == 2
    assert result["automatic_train_started"] is False
    assert result["status"] == "REVIEW_REQUIRED"


def test_compare_candidate_to_control_reports_negative_as_improvement(tmp_path) -> None:
    base = {
        "progress": [
            {
                "step": step,
                **{metric: 1.0 for metric in campaign.METRICS},
            }
            for step in campaign.EVAL_STEPS
        ]
    }
    candidate = [
        {
            "step": step,
            **{
                metric: (0.9 if step == campaign.UPDATES else 1.0)
                for metric in campaign.METRICS
            },
        }
        for step in campaign.EVAL_STEPS
    ]

    result = campaign.compare_candidate_to_control(base, candidate, tmp_path)
    assert result["automatic_go_no_go"] is False
    assert result["final_step"]["candidate_vs_control_f20_rel_pct"] == pytest.approx(-10.0)
    assert (tmp_path / "matched_comparison.csv").is_file()
