from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "train_sota_merge_sps.py"
LAUNCHER = ROOT / "tools" / "run_sota_merge_sps.sh"
MANIFEST = ROOT / "configs" / "clean_baseline_v1_split.json"


def test_sps_runner_is_hard_bound_to_reviewed_joint6k_pair():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "TARGET_JOINT_UPDATE = 6_000" in text
    assert (
        'EXPECTED_BACKBONE_SHA256 = '
        '"ca9d3efbcbe10a5b0190875c6bc749386002678e33b86d7ecda0fba55c306d86"'
    ) in text
    assert (
        'EXPECTED_CORRECTOR_SHA256 = '
        '"a98b4eb048e193a6337d267fe54ead8645d77ddcfaad18321f0f6c07c76ebec8"'
    ) in text


def test_sps_runner_preserves_frozen_exp3_recipe():
    text = SCRIPT.read_text(encoding="utf-8")
    for expected in (
        "UPDATES = 5_000",
        "EVAL_EVERY = 500",
        "BATCH = 16",
        "LR = 1e-3",
        "WARMUP = 200",
        "hidden=64",
        "blocks=2",
        "dropout=0.0",
        "include_delta=False",
        "fixed_dataset(train_paths, 5)",
        "exp3.calibrate_static",
        "exp3.calibrate_adaptive",
        "exp3.select_adaptive_under_width_cap",
    ):
        assert expected in text


def test_point_predictor_is_frozen_and_not_in_optimizer():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "parameter.requires_grad_(False)" in text
    assert "torch.optim.AdamW(\n        head.parameters()" in text
    assert "optimizer_updates_point_model" in text
    assert '"optimizer_updates_point_model": 0' in text


def test_aoa10_is_gated_after_seen_dev_go_and_not_used_for_selection():
    text = SCRIPT.read_text(encoding="utf-8")
    gate_pos = text.index('if gate == "GO":')
    aoa_collect_pos = text.index("aoa10_paths", gate_pos)
    assert gate_pos < aoa_collect_pos
    assert '"selection_split": "seen_dev"' in text
    assert '"aoa10_used_for_selection": False' in text
    assert '"aoa10_recalibrated": False' in text


def test_canonical_manifest_is_51_12_18_with_aoa10_holdout_only():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert len(payload["train"]) == 51
    assert len(payload["dev"]) == 12
    assert len(payload["holdout"]) == 18
    assert all(not name.endswith("_10.h5") for name in payload["train"])
    assert all(not name.endswith("_10.h5") for name in payload["dev"])
    assert all(name.endswith("_10.h5") for name in payload["holdout"])


def test_launcher_accepts_one_canonical_manifest_only():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "CLEAN_51_12_18_MANIFEST" in text
    assert "AOA10_MANIFEST" not in text
    assert "--aoa10-manifest" not in text
    assert "--aoa10-split" not in text
