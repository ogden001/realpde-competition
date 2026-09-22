from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "colleague_80pt"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(ROOT / "tools"))

from pareto_tke import (  # noqa: E402
    project_tke_backward,
    split_residual_objective,
)
from realpde_h5_feature_adapter_train import physics_loss  # noqa: E402
from archive_v2_campaign import archive  # noqa: E402
from run_v2_campaign import (  # noqa: E402
    ARM_A_EVAL,
    ARM_A_UPDATES,
    ARM_B_EVAL,
    ARM_B_UPDATES,
    ARM_C_UPDATES,
    backbone_mechanical_gate,
    pareto_mechanical_gate,
)


def test_pareto_objective_sum_matches_historical_scalar_objective() -> None:
    torch.manual_seed(7)
    pred = torch.randn(2, 20, 4, 6, 3)
    target = torch.randn(2, 20, 4, 6, 3)
    base = torch.randn(2, 20, 4, 6, 3)
    delta = pred - base
    pred[..., 2] = 0.0
    target[..., 2] = 0.0
    base[..., 2] = 0.0
    delta[..., 2] = 0.0

    weights = {
        "point": 1.0,
        "mse": 0.05,
        "tke": 0.12,
        "temporal": 0.03,
        "grad": 0.015,
        "p_zero": 0.01,
    }
    scalar, _ = physics_loss(pred, target, weights)
    residual_mse = torch.mean(
        (delta[..., :2] - (target[..., :2] - base[..., :2])) ** 2
    )
    delta_penalty = torch.mean(delta[..., :2] ** 2)
    scalar = scalar + 0.25 * residual_mse + 0.02 * delta_penalty

    primary, energy, _ = split_residual_objective(
        pred,
        target,
        base,
        delta,
        weights=weights,
        residual_mse_weight=0.25,
        delta_penalty_weight=0.02,
    )
    torch.testing.assert_close(primary + energy, scalar, atol=1e-6, rtol=1e-6)


def test_pareto_projection_removes_only_conflicting_energy_component() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    primary = parameter[0] + parameter[1]
    energy = -parameter[0]

    diagnostics = project_tke_backward(primary, energy, [parameter])

    assert diagnostics["pareto_conflict"] == 1.0
    assert diagnostics["pareto_grad_dot_before"] < 0.0
    assert abs(diagnostics["pareto_grad_dot_after"]) < 1e-6
    torch.testing.assert_close(
        parameter.grad,
        torch.tensor([0.5, 1.5]),
        atol=1e-6,
        rtol=0.0,
    )


def test_pareto_projection_matches_sum_when_gradients_do_not_conflict() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    primary = parameter[0].square()
    energy = parameter[1].square()

    diagnostics = project_tke_backward(primary, energy, [parameter])

    assert diagnostics["pareto_conflict"] == 0.0
    torch.testing.assert_close(
        parameter.grad,
        torch.tensor([4.0, 6.0]),
        atol=1e-6,
        rtol=0.0,
    )


def test_campaign_budgets_are_frozen() -> None:
    assert (ARM_A_UPDATES, ARM_A_EVAL) == (12_000, 2_000)
    assert (ARM_B_UPDATES, ARM_B_EVAL) == (38_400, 4_800)
    assert ARM_C_UPDATES == 20_000


def test_pareto_mechanical_gate_requires_tke_gain_and_point_protection() -> None:
    candidate = {
        "rel_l2_raw": 0.0805,
        "tke_raw": 0.4300,
        "mvpe_raw": 0.0712,
    }
    assert pareto_mechanical_gate(candidate)["pass"] is True
    candidate["rel_l2_raw"] = 0.082
    assert pareto_mechanical_gate(candidate)["pass"] is False


def test_backbone_gate_allows_only_bounded_point_tradeoff() -> None:
    candidate = {
        "rel_l2_raw": 0.0808,
        "tke_raw": 0.4300,
        "mvpe_raw": 0.0715,
    }
    assert backbone_mechanical_gate(candidate)["pass"] is True
    candidate["mvpe_raw"] = 0.073
    assert backbone_mechanical_gate(candidate)["pass"] is False


def test_campaign_archiver_whitelists_lightweight_evidence(tmp_path: Path) -> None:
    source = tmp_path / "run"
    destination = tmp_path / "archive"
    (source / "A" / "checkpoints").mkdir(parents=True)
    (source / "A" / "metrics.json").write_text('{"x": 1}\n', encoding="utf-8")
    (source / "A" / "table.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (source / "A.train.review.log").write_text("review\n", encoding="utf-8")
    (source / "A.raw.log").write_text("raw log must stay remote\n", encoding="utf-8")
    (source / "A" / "checkpoints" / "model.pth").write_bytes(b"checkpoint")

    result = archive(source, destination)

    assert result["excluded_checkpoints"] is True
    assert result["excluded_raw_logs"] is True
    assert (destination / "A" / "metrics.json").is_file()
    assert (destination / "A" / "table.csv").is_file()
    assert (destination / "A.train.review.log").is_file()
    assert not (destination / "A.raw.log").exists()
    assert not (destination / "A" / "checkpoints" / "model.pth").exists()
    assert (destination / "ARCHIVE_MANIFEST.json").is_file()
