from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "colleague_80pt"))


def load_module():
    path = ROOT / "tools" / "diagnose_strong_backbone_tail_origin.py"
    spec = importlib.util.spec_from_file_location("tail_origin", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_module()


def synthetic_raw(tail_scale: float = 1.0, fluct_bias: float = 0.0):
    raw = np.zeros((2, 20, 2, 2, 5), dtype=np.float32)
    target = np.zeros((2, 20, 2, 2, 3), dtype=np.float32)
    mode = np.asarray([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32)
    phases = np.sin(np.linspace(0, 2 * np.pi, 20, endpoint=False)).astype(np.float32)
    for h in range(20):
        amp = tail_scale if h == 19 else 1.0
        field = amp * phases[h] * mode
        raw[:, h, ..., 0] = 0.5
        raw[:, h, ..., 1] = -0.25
        raw[:, h, ..., 2] = field + fluct_bias
        raw[:, h, ..., 3] = 0.5 * field + fluct_bias
        target[:, h, ..., 0] = 0.5 + phases[h] * mode
        target[:, h, ..., 1] = -0.25 + 0.5 * phases[h] * mode
    return raw, target


def test_raw_reconstruct_centers_fluctuation_only():
    raw, _ = synthetic_raw(fluct_bias=0.3)
    mean_raw, raw_fluct, centered, reconstructed = M.raw_reconstruct(raw)
    assert mean_raw.shape[-1] == 2
    assert raw_fluct.shape[-1] == 2
    assert np.max(np.abs(centered.mean(axis=1))) < 1e-10
    assert np.allclose(reconstructed.mean(axis=1), mean_raw.mean(axis=1), atol=1e-10)


def test_component_rows_detect_raw_tail_amplitude_growth():
    raw, target = synthetic_raw(tail_scale=3.0)
    rows = M.component_horizon_rows(raw, target)
    f18, f20 = rows[17], rows[19]
    assert f20["raw_fluct_rms"] > f18["raw_fluct_rms"]
    assert f20["raw_fluct_amp_ratio"] > 2.0
    assert f20["raw_energy_share"] > f18["raw_energy_share"]


def test_centering_can_be_distinguished_from_raw_head():
    raw, target = synthetic_raw(tail_scale=1.0, fluct_bias=2.0)
    rows = M.component_horizon_rows(raw, target)
    # Large constant bias makes raw fluctuation bad, while zero-mean centering removes it.
    improvements = [
        row["raw_fluct_rel_l2"] - row["centered_fluct_rel_l2"]
        for row in rows
        if row["target_fluct_rms"] > 1e-6
    ]
    assert np.mean(improvements) > 0.5


def test_cliff_summary_reports_tail_energy_share():
    horizon_rows = []
    component_rows = []
    for h in range(1, 21):
        rel = 0.1
        if h == 19:
            rel = 0.12
        if h == 20:
            rel = 0.18
        horizon_rows.append({"horizon": h, "frame_rel_l2": rel})
        component_rows.append({
            "horizon": h,
            "raw_fluct_amp_ratio": 1.0 if h < 20 else 1.8,
            "centered_fluct_amp_ratio": 1.0 if h < 20 else 1.6,
            "centered_fluct_cosine": 0.8 if h < 20 else 0.2,
            "raw_energy_share": 0.05,
            "centered_energy_share": 0.05,
            "target_energy_share": 0.05,
            "precenter_frame_rel_l2": rel,
            "reconstructed_frame_rel_l2": rel,
        })
    summary = M.cliff_summary(horizon_rows, component_rows)
    assert abs(summary["rel_growth_f18_f20_pct"] - 80.0) < 1e-12
    assert abs(summary["raw_tail_energy_share_f19_f20"] - 0.10) < 1e-12
    assert summary["raw_fluct_amp_ratio_f20"] == 1.8


def test_classify_evolution_distinguishes_training_growth():
    curve = [
        {
            "update": 0,
            "rel_growth_f18_f20_pct": 5.0,
            "raw_fluct_amp_ratio_f20": 1.0,
            "centered_fluct_amp_ratio_f20": 1.0,
            "centering_f20_rel_delta_pct": 0.0,
        },
        {
            "update": 7500,
            "rel_growth_f18_f20_pct": 12.0,
            "raw_fluct_amp_ratio_f20": 1.1,
            "centered_fluct_amp_ratio_f20": 1.1,
            "centering_f20_rel_delta_pct": 2.0,
        },
        {
            "update": 35000,
            "rel_growth_f18_f20_pct": 55.0,
            "raw_fluct_amp_ratio_f20": 1.7,
            "centered_fluct_amp_ratio_f20": 1.6,
            "centering_f20_rel_delta_pct": 3.0,
        },
    ]
    result = M.classify_evolution(curve)
    flags = result["diagnostic_flags"]
    assert flags["cliff_present_at_init"] is False
    assert flags["cliff_grows_materially_during_training"] is True
    assert flags["raw_head_f20_amplitude_grows_materially"] is True
    assert flags["mf_centering_materially_worsens_f20"] is False


def test_classify_evolution_detects_init_cliff_and_centering():
    curve = [
        {
            "update": 0,
            "rel_growth_f18_f20_pct": 20.0,
            "raw_fluct_amp_ratio_f20": 1.0,
            "centered_fluct_amp_ratio_f20": 1.2,
            "centering_f20_rel_delta_pct": 15.0,
        },
        {
            "update": 7500,
            "rel_growth_f18_f20_pct": 22.0,
            "raw_fluct_amp_ratio_f20": 1.05,
            "centered_fluct_amp_ratio_f20": 1.25,
            "centering_f20_rel_delta_pct": 16.0,
        },
        {
            "update": 35000,
            "rel_growth_f18_f20_pct": 24.0,
            "raw_fluct_amp_ratio_f20": 1.1,
            "centered_fluct_amp_ratio_f20": 1.3,
            "centering_f20_rel_delta_pct": 18.0,
        },
    ]
    result = M.classify_evolution(curve)
    flags = result["diagnostic_flags"]
    assert flags["cliff_present_at_init"] is True
    assert flags["cliff_grows_materially_during_training"] is False
    assert flags["mf_centering_materially_worsens_f20"] is True


def test_frozen_protocol_constants():
    assert M.EXPECTED_MILESTONES == (7500, 15000, 20000, 25000, 30000, 31000, 32500, 35000)
    assert M.SIM_PRETRAIN_SHA256 == "82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61"
    assert M.EXPECTED_DEV_WINDOWS == 491
