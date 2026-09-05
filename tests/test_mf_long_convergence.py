import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import realpde_mf_long_convergence_lowmem as runner


def test_absolute_milestones_are_frozen_for_3k_to_15k_campaign():
    assert runner.absolute_milestones(3000, 12000, 3000) == [6000, 9000, 12000, 15000]
    with pytest.raises(ValueError):
        runner.absolute_milestones(3000, 12000, 2500)


def test_low_memory_microbatch_plan_preserves_effective_batch_weight():
    plan = runner.microbatch_plan(8, 2)
    assert plan == [(0, 2, 0.25), (2, 4, 0.25), (4, 6, 0.25), (6, 8, 0.25)]
    assert sum(weight for _, _, weight in plan) == pytest.approx(1.0)

    partial = runner.microbatch_plan(4, 2)
    assert partial == [(0, 2, 0.5), (2, 4, 0.5)]
    assert sum(weight for _, _, weight in partial) == pytest.approx(1.0)


def test_mf_checkpoint_metadata_requires_exact_c0_parent_contract():
    good = {
        "iteration": 1500,
        "metadata": {
            "experiment_id": "T1-ID-MF-C02-CONT-S20260901",
            "mode": "c0",
            "seed": 20260901,
            "updates": 1500,
            "lr": 1e-5,
            "batch_size": 8,
            "workers": 2,
            "manifest_sha256": runner.EXPECTED_MANIFEST_SHA256,
            "checkpoint_sha256": runner.EXPECTED_MF1500_SHA256,
            "scorer_sha256": runner.EXPECTED_SCORER_SHA256,
        },
    }
    runner.validate_mf3000_metadata(good)

    bad = json.loads(json.dumps(good))
    bad["metadata"]["mode"] = "e5"
    with pytest.raises(RuntimeError, match="MF@3000 metadata mismatch"):
        runner.validate_mf3000_metadata(bad)


def test_matched_improvement_is_positive_when_mf_has_lower_error():
    direct = {"rel_l2": 0.20, "tke": 0.50, "mvpe": 0.10}
    mf = {"rel_l2": 0.18, "tke": 0.49, "mvpe": 0.08}
    assert runner.mf_improvement_pct(direct, mf) == pytest.approx(
        {"rel_l2": 10.0, "tke": 2.0, "mvpe": 20.0}
    )


def test_optimizer_state_roundtrip_preserves_adamw_moments():
    source = torch.nn.Linear(3, 2)
    source_opt = torch.optim.AdamW(source.parameters(), lr=1e-5)
    x = torch.randn(4, 3)
    source(x).square().mean().backward()
    source_opt.step()
    checkpoint = {"optimizer_state_dict": source_opt.state_dict()}

    target = torch.nn.Linear(3, 2)
    target.load_state_dict(source.state_dict())
    target_opt = torch.optim.AdamW(target.parameters(), lr=1e-5)
    runner.restore_optimizer_state(target_opt, checkpoint)

    source_state = source_opt.state_dict()
    target_state = target_opt.state_dict()
    assert source_state["param_groups"] == target_state["param_groups"]
    assert source_state["state"].keys() == target_state["state"].keys()
    for key in source_state["state"]:
        for name, value in source_state["state"][key].items():
            other = target_state["state"][key][name]
            if torch.is_tensor(value):
                assert torch.equal(value, other)
            else:
                assert value == other
