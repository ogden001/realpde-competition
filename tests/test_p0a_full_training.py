from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from realpde_p0a_n2_full import (  # noqa: E402
    N2_WEIGHTS,
    LIFT_WEIGHT_KEY,
    P0AConfig,
    build_p0a_features,
    expand_cno_input_state,
    n2_loss,
    manifest_paths,
    load_resume_checkpoint,
    milestone_checkpoint_path,
    _save_checkpoint,
    validate_training_protocol,
    validate_limits,
)


def test_expand_cno_input_preserves_raw_channels_and_zeros_features():
    source = {LIFT_WEIGHT_KEY: torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2, 1)}
    target = {LIFT_WEIGHT_KEY: torch.full((2, 20, 2, 2, 1), -1.0)}

    expanded = expand_cno_input_state(source, target, LIFT_WEIGHT_KEY)

    assert torch.equal(expanded[LIFT_WEIGHT_KEY][:, :3], source[LIFT_WEIGHT_KEY])
    assert torch.count_nonzero(expanded[LIFT_WEIGHT_KEY][:, 3:]) == 0


def test_p0a_requires_only_input_history_and_has_20_channels():
    inputs = torch.zeros(2, 20, 32, 64, 3)

    features = build_p0a_features(inputs, P0AConfig(dx=0.1, dy=0.1))

    assert features.shape == (2, 20, 32, 64, 20)
    assert torch.isfinite(features).all()


@pytest.mark.parametrize(
    ("max_gpu_gib", "max_train_seconds"),
    [(23.51, 20_000), (23.5, 20_401)],
)
def test_training_config_rejects_memory_or_time_over_limit(max_gpu_gib, max_train_seconds):
    with pytest.raises(ValueError):
        validate_limits(max_gpu_gib=max_gpu_gib, max_train_seconds=max_train_seconds)


def test_n2_loss_is_zero_for_identical_velocity_fields():
    fields = torch.randn(2, 20, 4, 5, 3)

    loss, parts = n2_loss(fields, fields)

    assert loss.item() == pytest.approx(0.0, abs=1e-8)
    assert set(parts) == set(N2_WEIGHTS)


def test_manifest_paths_uses_only_requested_split(tmp_path):
    manifest = tmp_path / "split.json"
    manifest.write_text('{"train": [{"file": "a.h5"}], "dev": [{"file": "b.h5"}]}')
    (tmp_path / "b.h5").touch()

    paths = manifest_paths(manifest, tmp_path, "dev")

    assert paths == [tmp_path / "b.h5"]


def test_resume_checkpoint_restores_model_optimizer_and_iteration(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    path = tmp_path / "resume.pth"
    config = P0AConfig(dx=0.1, dy=-0.1)
    _save_checkpoint(path, model=model, optimizer=optimizer, update=17, config=config, metadata={})

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    iteration = load_resume_checkpoint(path, model=restored_model, optimizer=restored_optimizer, config=config)

    assert iteration == 17
    assert all(torch.equal(left, right) for left, right in zip(model.parameters(), restored_model.parameters()))
    assert restored_optimizer.state_dict()["state"]


def test_milestone_checkpoint_path_is_unique_and_ordered(tmp_path):
    assert milestone_checkpoint_path(tmp_path, 10_300) == tmp_path / "model_update_10300.pth"
    with pytest.raises(ValueError):
        milestone_checkpoint_path(tmp_path, 0)


def test_training_protocol_allows_explicit_high_throughput_batching():
    validate_training_protocol(micro_batch=16, accumulate=1)
    with pytest.raises(ValueError):
        validate_training_protocol(micro_batch=0, accumulate=1)
