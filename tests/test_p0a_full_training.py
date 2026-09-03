from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from realpde_p0_features import P0FeatureConfig  # noqa: E402
from realpde_p0a_n2_full import (  # noqa: E402
    N2_WEIGHTS,
    historical_p0a_spacing,
    load_resume_checkpoint,
    manifest_paths,
    milestone_checkpoint_path,
    save_checkpoint,
    terminal_state,
    validate_milestone_updates,
)


EXPECTED_N2 = {"mse": 1.0, "tke": 0.05, "rel": 0.027514, "mvpe": 0.009757}


def test_continuation_uses_historical_four_key_n2_schema():
    assert N2_WEIGHTS == EXPECTED_N2


def test_historical_p0a_spacing_matches_full15300_runner_semantics(tmp_path):
    path = tmp_path / "grid.h5"
    x = np.tile(np.arange(128, dtype=np.float32) * 0.1, (64, 1))
    y = np.tile((np.arange(64, dtype=np.float32) * -0.2)[:, None], (1, 128))
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x", data=x)
        handle.create_dataset("y", data=y)

    dx, dy = historical_p0a_spacing([path])

    assert dx == pytest.approx(0.1)
    assert dy == pytest.approx(-0.2)
    # A naive adjacent difference after ::2 downsampling would be 0.2/-0.4.
    # The historical 15,300 checkpoint was trained with the original-grid spacing.
    assert dx != pytest.approx(0.2)
    assert dy != pytest.approx(-0.4)


def test_historical_p0a_spacing_rejects_mixed_grids(tmp_path):
    paths = []
    for index, dx in enumerate((0.1, 0.11)):
        path = tmp_path / f"grid_{index}.h5"
        x = np.tile(np.arange(128, dtype=np.float32) * dx, (64, 1))
        y = np.tile((np.arange(64, dtype=np.float32) * -0.2)[:, None], (1, 128))
        with h5py.File(path, "w") as handle:
            handle.create_dataset("x", data=x)
            handle.create_dataset("y", data=y)
        paths.append(path)

    with pytest.raises(ValueError, match="spacing differs"):
        historical_p0a_spacing(paths)


def test_resume_accepts_historical_four_key_n2_checkpoint(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    checkpoint = tmp_path / "historical_resume.pth"
    config = P0FeatureConfig(include_p0_a=True, include_p0_b=False, dx=0.1, dy=-0.1)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": 18860,
            "feature_set": "P0-A",
            "feature_config": {"dx": 0.1, "dy": -0.1},
            "loss_weights": EXPECTED_N2,
        },
        checkpoint,
    )

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=5e-6)
    iteration = load_resume_checkpoint(
        checkpoint,
        model=restored_model,
        optimizer=restored_optimizer,
        config=config,
        lr_override=5e-6,
    )

    assert iteration == 18860
    assert restored_optimizer.state_dict()["state"]
    assert {group["lr"] for group in restored_optimizer.param_groups} == {5e-6}


def test_resume_can_override_lr_without_losing_optimizer_state(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    checkpoint = tmp_path / "resume.pth"
    config = P0FeatureConfig(include_p0_a=True, include_p0_b=False, dx=0.1, dy=-0.1)
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        update=18860,
        config=config,
        metadata={},
    )

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=5e-6)
    iteration = load_resume_checkpoint(
        checkpoint,
        model=restored_model,
        optimizer=restored_optimizer,
        config=config,
        lr_override=5e-6,
    )

    assert iteration == 18860
    assert restored_optimizer.state_dict()["state"]
    assert {group["lr"] for group in restored_optimizer.param_groups} == {5e-6}
    assert all(torch.equal(left, right) for left, right in zip(model.parameters(), restored_model.parameters()))


def test_resume_rejects_nonpositive_lr_override(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    checkpoint = tmp_path / "resume.pth"
    config = P0FeatureConfig(include_p0_a=True, include_p0_b=False, dx=0.1, dy=-0.1)
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        update=18860,
        config=config,
        metadata={},
    )
    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-5)

    with pytest.raises(ValueError, match="lr_override"):
        load_resume_checkpoint(
            checkpoint,
            model=restored_model,
            optimizer=restored_optimizer,
            config=config,
            lr_override=0.0,
        )


def test_manifest_paths_uses_data_root_override(tmp_path):
    manifest = tmp_path / "split.json"
    manifest.write_text('{"train": [{"file": "a.h5"}], "dev": [{"file": "b.h5"}]}', encoding="utf-8")
    (tmp_path / "b.h5").touch()

    assert manifest_paths(manifest, tmp_path, "dev") == [tmp_path / "b.h5"]


def test_milestones_must_be_after_resume_and_at_or_before_target(tmp_path):
    values = validate_milestone_updates([31100, 36500, 37850, 40560, 43260], start=15300, target=43260)
    assert values == (31100, 36500, 37850, 40560, 43260)
    assert milestone_checkpoint_path(tmp_path, 31100) == tmp_path / "model_update_31100.pth"

    with pytest.raises(ValueError, match="milestone"):
        validate_milestone_updates([15300], start=15300, target=43260)
    with pytest.raises(ValueError, match="milestone"):
        validate_milestone_updates([43261], start=15300, target=43260)


def test_time_cap_is_valid_terminal_only_when_explicitly_allowed():
    assert terminal_state(
        stop_reason="time_cap",
        completed_update=40123,
        target_update=43260,
        initial_update=15300,
        allow_time_cap=True,
    ) == "TIME_CAPPED"
    assert terminal_state(
        stop_reason="time_cap",
        completed_update=40123,
        target_update=43260,
        initial_update=15300,
        allow_time_cap=False,
    ) == "STOPPED"
    assert terminal_state(
        stop_reason="updates_complete",
        completed_update=43260,
        target_update=43260,
        initial_update=15300,
        allow_time_cap=True,
    ) == "DONE"
