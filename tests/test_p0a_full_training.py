from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from realpde_p0_features import P0FeatureConfig  # noqa: E402
from realpde_p0a_n2_full import (  # noqa: E402
    load_resume_checkpoint,
    manifest_paths,
    save_checkpoint,
)


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
