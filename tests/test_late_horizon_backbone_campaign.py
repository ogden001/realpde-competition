from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "colleague_80pt"))

from late_horizon_backbone_campaign import (  # noqa: E402
    continuation_loss,
    horizon_weighted_velocity_mse,
    normalized_linear_horizon_weights,
    tail_fingerprint,
)
from realpde_h5_feature_adapter_train import load_cno_class  # noqa: E402


def test_horizon_ramp_is_positive_mean_one_and_two_to_one_endpoints() -> None:
    weights = normalized_linear_horizon_weights(20)
    assert torch.all(weights > 0)
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0))
    torch.testing.assert_close(weights[-1] / weights[0], torch.tensor(2.0))


def test_horizon_weighted_mse_equals_uniform_when_each_horizon_error_is_equal() -> None:
    pred = torch.zeros(2, 20, 3, 4, 3)
    target = torch.ones_like(pred)
    target[..., 2] = 0.0
    uniform = ((pred[..., :2] - target[..., :2]) ** 2).mean()
    ramped = horizon_weighted_velocity_mse(pred, target)
    torch.testing.assert_close(ramped, uniform)


def test_ramp_loss_only_reweights_historical_mse_term() -> None:
    torch.manual_seed(3)
    pred = torch.randn(2, 20, 4, 5, 3, requires_grad=True)
    target = torch.randn(2, 20, 4, 5, 3)
    target[..., 2] = 0.0

    control, cparts = continuation_loss(pred, target, mode="control")
    ramp, rparts = continuation_loss(pred, target, mode="ramp")
    expected = control + (
        rparts["ramped_velocity_mse"] - cparts["uniform_velocity_mse"]
    )
    torch.testing.assert_close(ramp, expected, atol=1e-7, rtol=1e-6)


def test_ramp_penalizes_late_only_error_more_than_uniform() -> None:
    pred = torch.zeros(1, 20, 2, 2, 3)
    target = torch.zeros_like(pred)
    target[:, -1, ..., :2] = 1.0
    uniform = ((pred[..., :2] - target[..., :2]) ** 2).mean()
    ramped = horizon_weighted_velocity_mse(pred, target)
    assert float(ramped) > float(uniform)


def test_tail_fingerprint_detects_f20_cliff() -> None:
    rows = [
        {"horizon": h, "frame_rel_l2": 0.05 + 0.001 * h}
        for h in range(1, 20)
    ]
    rows.append({"horizon": 20, "frame_rel_l2": 0.15})
    result = tail_fingerprint(rows)
    assert result["f20_rel"] == 0.15
    assert result["f20_over_f18"] > 2.0
    assert result["f20_over_mean_f1_f17"] > 2.0


def test_tail_fingerprint_requires_all_20_horizons() -> None:
    rows = [{"horizon": h, "frame_rel_l2": 0.1} for h in range(1, 20)]
    try:
        tail_fingerprint(rows)
    except ValueError as error:
        assert "1..20" in str(error)
    else:
        raise AssertionError("expected missing horizon to fail")


def test_load_cno_class_supports_official_kit_module_layout(
    tmp_path: Path, monkeypatch
) -> None:
    package = tmp_path / "rpde_baselines"
    model_package = package / "model"
    model_package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (model_package / "__init__.py").write_text("", encoding="utf-8")
    (model_package / "cno.py").write_text(
        "class CNO3d:\n    pass\n", encoding="utf-8"
    )

    monkeypatch.setattr(sys, "path", list(sys.path))
    for module_name in tuple(sys.modules):
        if module_name == "rpde_baselines" or module_name.startswith(
            "rpde_baselines."
        ):
            monkeypatch.delitem(sys.modules, module_name)

    cno_class = load_cno_class(tmp_path)

    assert cno_class.__module__ == "rpde_baselines.model.cno"
    assert isinstance(cno_class(), cno_class)
