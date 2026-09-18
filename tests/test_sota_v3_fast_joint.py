from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import realpde_sota_v3_fast_joint as fast  # noqa: E402


def test_fast_joint_budget_and_discriminative_lrs_are_frozen():
    assert fast.UPDATES == 5_000
    assert fast.MILESTONES == (1_000, 2_000, 3_000, 4_000, 5_000)
    assert fast.EFFECTIVE_BATCH == 8
    assert fast.BACKBONE_LR == pytest.approx(1e-6)
    assert fast.CORRECTOR_LR == pytest.approx(1e-5)


def test_selection_compares_against_historical_projected_residual_anchor():
    assert fast.HISTORICAL_FINAL == pytest.approx({
        "rel_l2": 0.0889103040099144,
        "tke": 0.4692927300930023,
        "mvpe": 0.07112862169742584,
    })
    rows = [
        {"update": 0, **fast.HISTORICAL_FINAL},
        {"update": 1000, "rel_l2": 0.088, "tke": 0.468, "mvpe": 0.0705},
        {"update": 2000, "rel_l2": 0.087, "tke": 0.467, "mvpe": 0.0700},
    ]
    assert fast.select_candidate(rows)["update"] == 2000


def test_go_full_gate_requires_mean_gain_guardrails_and_two_metrics():
    good = {k: v * 0.985 for k, v in fast.HISTORICAL_FINAL.items()}
    assert fast.evaluate_gate(good)["status"] == "FAST_JOINT_GO_FULL"

    one_metric_only = dict(fast.HISTORICAL_FINAL)
    one_metric_only["rel_l2"] *= 0.95
    one_metric_only["tke"] *= 1.005
    one_metric_only["mvpe"] *= 1.005
    assert fast.evaluate_gate(one_metric_only)["status"] == "FAST_JOINT_NO_GO"

    bad_guard = dict(good)
    bad_guard["tke"] = fast.HISTORICAL_FINAL["tke"] * 1.021
    assert fast.evaluate_gate(bad_guard)["status"] == "FAST_JOINT_NO_GO"


def test_stage_b_loss_keeps_existing_extra_rel(monkeypatch):
    parts = {
        "mse": torch.tensor(1.0),
        "tke": torch.tensor(2.0),
        "rel": torch.tensor(3.0),
        "mvpe": torch.tensor(4.0),
    }
    monkeypatch.setattr(fast.base.core, "loss_parts", lambda p, y: dict(parts))
    monkeypatch.setattr(fast.base, "vorticity", lambda x, dx, dy: torch.zeros(1))
    total, returned = fast.stage_b_loss(torch.zeros(1), torch.zeros(1))
    expected = sum(fast.base.N2[k] * parts[k] for k in fast.base.N2)
    expected = expected + fast.base.EXTRA_REL * parts["rel"]
    assert total.item() == pytest.approx(expected.item())
    assert "vorticity" in returned


def test_runner_has_no_full_data_or_submission_path():
    source = inspect.getsource(fast.run)
    assert "released_paths" not in source
    assert "full_updates" not in source
