from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "colleague_80pt"))


def load_module():
    path = ROOT / "tools" / "train_clean_direct_cno_7500_control.py"
    spec = importlib.util.spec_from_file_location("direct_control", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_module()


def test_protocol_is_exact_7500_stage_a_control():
    assert M.SEED == 41
    assert M.FINAL_UPDATE == 7_500
    assert M.BATCH_SIZE == 8
    assert M.LR == 1e-5
    assert M.MILESTONES == (0, 2_500, 5_000, 7_500)


def test_memory_cap_is_exactly_12_gib():
    assert M.MAX_PEAK_RESERVED_BYTES == 12 * 1024**3


def test_official_init_sha_is_frozen():
    assert M.SIM_PRETRAIN_SHA256 == "82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61"


def test_source_uses_direct_not_mf_output():
    text = (ROOT / "tools" / "train_clean_direct_cno_7500_control.py").read_text(encoding="utf-8")
    assert "sota.build_direct" in text
    assert "sota.forward_direct" in text
    assert "MF01CNO" not in text
    assert "forward_mf(" not in text


def test_source_keeps_dense_all_p0a_and_integrated_loss():
    text = (ROOT / "tools" / "train_clean_direct_cno_7500_control.py").read_text(encoding="utf-8")
    assert "sota.build_features" in text
    assert "sota.dense_loader" in text
    assert "sota.integrated_loss" in text
    assert "torch.optim.AdamW" in text


def test_source_has_no_stage_b_or_holdout_execution():
    text = (ROOT / "tools" / "train_clean_direct_cno_7500_control.py").read_text(encoding="utf-8").lower()
    assert '"stage_b_used": false' in text
    assert "holdout_paths" not in text
    assert "codabench" in text
    assert "package_submission" not in text


def test_trajectory_horizon_aggregation_preserves_groups():
    rows = [
        {
            "trajectory": "a.h5",
            "window_start": 0,
            "horizon": 1,
            "frame_rel_l2": 1.0,
            "frame_rmse": 2.0,
            "tke_contrib_rel_l2": 3.0,
            "tke_contrib_ratio": 4.0,
            "mvpe_probe_rel_l2": 5.0,
        },
        {
            "trajectory": "a.h5",
            "window_start": 20,
            "horizon": 1,
            "frame_rel_l2": 3.0,
            "frame_rmse": 4.0,
            "tke_contrib_rel_l2": 5.0,
            "tke_contrib_ratio": 6.0,
            "mvpe_probe_rel_l2": 7.0,
        },
        {
            "trajectory": "b.h5",
            "window_start": 0,
            "horizon": 2,
            "frame_rel_l2": 10.0,
            "frame_rmse": 11.0,
            "tke_contrib_rel_l2": 12.0,
            "tke_contrib_ratio": 13.0,
            "mvpe_probe_rel_l2": 14.0,
        },
    ]
    out = M.aggregate_by_trajectory_horizon(rows)
    assert len(out) == 2
    a = next(row for row in out if row["trajectory"] == "a.h5")
    assert a["horizon"] == 1
    assert a["windows"] == 2
    assert a["frame_rel_l2"] == pytest.approx(2.0)


def test_point_score_rewards_lower_errors():
    worse = {"rel_l2": 0.12, "tke": 0.50, "mvpe": 0.10}
    better = {"rel_l2": 0.10, "tke": 0.45, "mvpe": 0.08}
    assert M.point_score(better) > M.point_score(worse)
