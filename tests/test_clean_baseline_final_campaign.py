from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "colleague_80pt"))
CAMPAIGN = load_module("cbfc", str(ROOT / "tools/clean_baseline_final_campaign.py"))
STRONG = load_module("cbfc_strong", str(ROOT / "tools/train_clean_strong_backbone.py"))


def test_strong_gate_requires_real_multimetric_gain():
    baseline = {"rel_l2_raw": 1.0, "tke_raw": 1.0, "mvpe_raw": 1.0}
    good = {"rel_l2_raw": 0.98, "tke_raw": 0.97, "mvpe_raw": 0.99}
    bad = {"rel_l2_raw": 0.97, "tke_raw": 1.001, "mvpe_raw": 0.97}
    assert CAMPAIGN.relative_gate(good, baseline, kind="strong")["status"] == "GO"
    assert CAMPAIGN.relative_gate(bad, baseline, kind="strong")["status"] == "NO_GO"


def test_pareto_gate_protects_rel_and_mvpe():
    baseline = {"rel_l2_raw": 1.0, "tke_raw": 1.0, "mvpe_raw": 1.0}
    good = {"rel_l2_raw": 1.004, "tke_raw": 0.96, "mvpe_raw": 0.998}
    bad = {"rel_l2_raw": 1.006, "tke_raw": 0.95, "mvpe_raw": 0.99}
    assert CAMPAIGN.relative_gate(good, baseline, kind="pareto")["status"] == "GO"
    assert CAMPAIGN.relative_gate(bad, baseline, kind="pareto")["status"] == "NO_GO"


def test_point_score_rewards_lower_error():
    a = {"rel_l2_raw": 0.1, "tke_raw": 0.5, "mvpe_raw": 0.08}
    b = {"rel_l2_raw": 0.09, "tke_raw": 0.45, "mvpe_raw": 0.07}
    assert CAMPAIGN.point_score(b) > CAMPAIGN.point_score(a)


def test_campaign_beta_grid_is_frozen_small():
    assert CAMPAIGN.BETAS == (0.5, 0.65, 0.8, 1.0)


def test_strong_recipe_uses_full_historical_schedule():
    assert STRONG.FINAL_UPDATE == 35_000
    assert STRONG.STAGE_A_END == 30_000
    assert STRONG.BATCH_SIZE == 8
    assert STRONG.SEED == 41


def test_no_submission_or_locked_final_execution_codepaths_in_campaign_source():
    text = (ROOT / "tools/clean_baseline_final_campaign.py").read_text(encoding="utf-8").lower()
    assert "codabench_accessed" in text
    assert "locked_final_accessed" in text
    assert "package_submission" not in text
    assert "submission.py" not in text


def test_sps_holdout_is_gated_by_seen_dev():
    text = (ROOT / "tools/train_clean_residual_aware_sps.py").read_text(encoding="utf-8")
    gate_pos = text.index('if gate["status"] == "GO":')
    holdout_collect_pos = text.index("collect(model, head, holdout_paths")
    assert gate_pos < holdout_collect_pos
    assert '"holdout_accessed": False' in text
