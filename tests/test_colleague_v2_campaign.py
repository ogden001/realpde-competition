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
from benchmark_v2_batch_profiles import parse_arms  # noqa: E402
from realpde_h5_feature_adapter_train import physics_loss  # noqa: E402
from archive_v2_campaign import archive  # noqa: E402
from batch_profiles import ARM_PROFILES, select_profile  # noqa: E402
from run_v2_campaign import (  # noqa: E402
    ARM_A_EVAL,
    ARM_A_UPDATES,
    ARM_B_EVAL,
    ARM_B_UPDATES,
    ARM_C_UPDATES,
    backbone_mechanical_gate,
    load_training_profiles,
    pareto_mechanical_gate,
)


def test_parse_arms_allows_independent_a_c_execution() -> None:
    assert parse_arms("A_pareto_tke,C_aoa_meanfield") == (
        "A_pareto_tke",
        "C_aoa_meanfield",
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
    assert ARM_PROFILES["A_pareto_tke"]["b16"].to_dict() == {
        "name": "b16",
        "batch_size": 16,
        "lr": 0.00028,
        "updates": 6000,
        "eval_interval": 1000,
        "matched_control_step": None,
        "samples": 96000,
    }
    assert ARM_PROFILES["B_strong_backbone"]["b16"].samples == (
        ARM_PROFILES["B_strong_backbone"]["b8"].samples
    )
    assert ARM_PROFILES["C_aoa_meanfield"]["b16"].samples == (
        ARM_PROFILES["C_aoa_meanfield"]["b8"].samples
    )
    assert ARM_PROFILES["C_aoa_meanfield"]["b16"].matched_control_step == 2500


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


def test_batch_profile_selects_b16_only_for_safe_clear_speedup() -> None:
    b8 = {
        "success": True,
        "samples_per_second": 100.0,
        "peak_allocated_fraction": 0.50,
        "peak_reserved_fraction": 0.55,
    }
    b16 = {
        "success": True,
        "samples_per_second": 125.0,
        "peak_allocated_fraction": 0.88,
        "peak_reserved_fraction": 0.93,
    }
    assert select_profile(b8, b16)["selected"] == "b16"

    b16["samples_per_second"] = 115.0
    assert select_profile(b8, b16)["selected"] == "b8"

    b16["samples_per_second"] = 130.0
    b16["peak_allocated_fraction"] = 0.93
    assert select_profile(b8, b16)["selected"] == "b8"


def test_batch_profile_falls_back_to_b8_if_b16_fails() -> None:
    b8 = {"success": True, "samples_per_second": 100.0}
    b16 = {"success": False}
    result = select_profile(b8, b16)
    assert result["selected"] == "b8"
    assert result["reason"] == "b16_failed"


def test_campaign_loads_and_validates_benchmark_profile(tmp_path: Path) -> None:
    profile_path = tmp_path / "selected_profiles.json"
    profile_path.write_text(
        """{
          "arms": {
            "A_pareto_tke": {"name": "b16", "batch_size": 16, "lr": 0.00028,
              "updates": 6000, "eval_interval": 1000, "matched_control_step": null,
              "samples": 96000},
            "B_strong_backbone": {"name": "b8"},
            "C_aoa_meanfield": {"name": "b16", "matched_control_step": 2500}
          }
        }""",
        encoding="utf-8",
    )
    profiles = load_training_profiles(profile_path)
    assert profiles["A_pareto_tke"].name == "b16"
    assert profiles["B_strong_backbone"].name == "b8"
    assert profiles["C_aoa_meanfield"].name == "b16"


def test_campaign_rejects_profile_semantic_drift(tmp_path: Path) -> None:
    profile_path = tmp_path / "bad_profiles.json"
    profile_path.write_text(
        """{
          "arms": {
            "A_pareto_tke": {"name": "b16", "lr": 0.0004},
            "B_strong_backbone": {"name": "b8"},
            "C_aoa_meanfield": {"name": "b8"}
          }
        }""",
        encoding="utf-8",
    )
    try:
        load_training_profiles(profile_path)
    except ValueError as error:
        assert "profile field lr" in str(error)
    else:
        raise AssertionError("expected profile drift to be rejected")
