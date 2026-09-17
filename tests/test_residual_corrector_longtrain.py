from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import realpde_residual_corrector_longtrain as longtrain  # noqa: E402


def test_longtrain_recipe_is_single_frozen_campaign():
    assert longtrain.BACKBONE_UPDATE == 32_500
    assert longtrain.FINAL_UPDATE == 30_000
    assert longtrain.MILESTONES == (6_000, 12_000, 20_000, 30_000)
    assert longtrain.BATCH_SIZE == 8
    assert longtrain.CORRECTOR_LR == pytest.approx(1e-4)
    assert longtrain.CORRECTOR_WEIGHT_DECAY == pytest.approx(1e-5)
    assert longtrain.ALPHA == pytest.approx(1.0)


def test_full_campaign_exposes_nearly_six_dense_epochs():
    assert longtrain.effective_dense_epochs() == pytest.approx(
        30_000 * 8 / 40_488
    )
    assert 5.9 < longtrain.effective_dense_epochs() < 6.0


def test_validate_recipe_rejects_budget_or_batch_sweeps():
    valid = Namespace(updates=30_000, batch_size=8, milestones=(6_000, 12_000, 20_000, 30_000))
    longtrain.validate_recipe(valid)

    with pytest.raises(ValueError):
        longtrain.validate_recipe(Namespace(updates=29_999, batch_size=8, milestones=valid.milestones))
    with pytest.raises(ValueError):
        longtrain.validate_recipe(Namespace(updates=30_000, batch_size=4, milestones=valid.milestones))
    with pytest.raises(ValueError):
        longtrain.validate_recipe(Namespace(updates=30_000, batch_size=8, milestones=(6_000, 30_000)))


def test_optimizer_scheduler_uses_full_30k_cosine_horizon():
    module = torch.nn.Linear(2, 2)
    optimizer, scheduler = longtrain.build_optimizer_scheduler(module)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-5)
    assert scheduler.T_max == 30_000


def _toy_arrays() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    # Two trajectories, one canonical window each, Future20, official 32x64 grid.
    target = np.ones((2, 20, 32, 64, 3), dtype=np.float32)
    target[..., 2] = 0.0
    base = target.copy()
    base[..., :2] += 0.20
    better = target.copy()
    better[..., :2] += 0.10
    best = target.copy()
    best[..., :2] += 0.05
    names = np.asarray(["traj_a.h5", "traj_b.h5"])
    return {"base": base, "u6000": better, "u30000": best}, target, names


def test_analysis_tables_are_generic_over_training_milestones():
    predictions, target, names = _toy_arrays()
    tables = longtrain.build_analysis_tables(
        predictions=predictions,
        target=target,
        trajectory_names=names,
    )

    assert len(tables["by_horizon"]) == 20 * 3
    assert len(tables["by_trajectory_horizon"]) == 2 * 20 * 3
    assert len(tables["horizon_trajectory_stability"]) == 20 * 2

    u30000 = [
        row for row in tables["by_horizon"]
        if row["variant"] == "u30000" and row["horizon"] == 1
    ][0]
    assert u30000["frame_rel_l2_improvement_vs_base"] > 0.0
    assert u30000["frame_rmse_improvement_vs_base"] > 0.0


def test_analysis_requires_base_and_matched_windows():
    predictions, target, names = _toy_arrays()
    with pytest.raises(ValueError):
        longtrain.build_analysis_tables(
            predictions={"u6000": predictions["u6000"]},
            target=target,
            trajectory_names=names,
        )
    with pytest.raises(ValueError):
        longtrain.build_analysis_tables(
            predictions=predictions,
            target=target,
            trajectory_names=np.asarray(["only_one.h5"]),
        )
