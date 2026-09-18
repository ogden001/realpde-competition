from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import build_sota_v3_package as package  # noqa: E402
import realpde_sota_v3_backbone as backbone  # noqa: E402
import realpde_sota_v3_residual as residual  # noqa: E402
import realpde_sota_v3_sps as sps  # noqa: E402
import run_sota_v3_pipeline as pipeline  # noqa: E402


def test_aoa_rotation_preserves_speed_and_non_uv_channels():
    flow = torch.zeros(2, 3, 2, 2, 3)
    flow[..., 0] = 1.0
    flow[..., 2] = 7.0
    rotated = backbone.rotate_uv(flow, torch.tensor([90.0, -90.0]))
    assert torch.allclose(rotated[0, ..., 0], torch.zeros_like(rotated[0, ..., 0]), atol=1e-6)
    assert torch.allclose(rotated[0, ..., 1], torch.ones_like(rotated[0, ..., 1]), atol=1e-6)
    assert torch.allclose(rotated[1, ..., 1], -torch.ones_like(rotated[1, ..., 1]), atol=1e-6)
    assert torch.allclose(rotated[..., :2].square().sum(-1), flow[..., :2].square().sum(-1), atol=1e-6)
    assert torch.equal(rotated[..., 2], flow[..., 2])


def test_aoa_pair_uses_same_sample_angle(monkeypatch):
    fixed = torch.tensor([2.0, -1.0])
    monkeypatch.setattr(backbone, "sample_aoa_angles", lambda *a, **k: fixed)
    x = torch.randn(2, 20, 2, 3, 3)
    y = torch.randn(2, 20, 2, 3, 3)
    ax, ay, angles = backbone.augment_pair(x, y)
    assert torch.equal(angles, fixed)
    assert torch.allclose(ax, backbone.rotate_uv(x, fixed))
    assert torch.allclose(ay, backbone.rotate_uv(y, fixed))


def test_dev_checkpoint_selection_is_preregistered_and_post_30k_only():
    rows = [
        {"update": 25_000, "rel_l2": 0.01, "tke": 0.01, "mvpe": 0.01},
        {"update": 30_000, "rel_l2": 0.10, "tke": 0.47, "mvpe": 0.08},
        {"update": 32_500, "rel_l2": 0.09, "tke": 0.46, "mvpe": 0.07},
    ]
    selected = backbone.select_dev_checkpoint(rows)
    assert selected["update"] == 32_500


def test_full_residual_budget_maps_dev_dense_epoch_exposure():
    assert residual.full_updates() == 49_461


def test_sps_recipe_is_exact_budget_but_final_center_reconstruction():
    assert sps.SEED == 41
    assert sps.MAX_UPDATES == 2000
    assert sps.EVAL_INTERVAL == 200
    assert sps.HEAD_LR == pytest.approx(1e-3)
    assert sps.HEAD_WEIGHT_DECAY == pytest.approx(1e-5)


def test_package_runtime_feeds_head_base_and_centers_bounds_on_final():
    source = package.submission_source()
    assert "sigma_from_log_std(head(tensor, base))" in source
    assert "prediction = _spatial_tke_map_projection(base, corrected)" in source
    assert "lower, upper = prediction - half, prediction + half" in source


def test_pipeline_has_fixed_unattended_stage_order():
    assert pipeline.STAGES == (
        "01_dev_backbone", "02_dev_residual", "03_dev_sps",
        "04_full_backbone", "05_full_residual", "06_full_sps",
        "07_package", "08_smoke",
    )
