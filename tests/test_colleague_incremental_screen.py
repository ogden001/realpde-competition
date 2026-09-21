from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch


TOOLS = Path(__file__).resolve().parents[1] / "tools" / "colleague_80pt"
sys.path.insert(0, str(TOOLS))

from realpde_h5_feature_adapter_train import (  # noqa: E402
    H5WindowDataset,
    RandomPhaseWindowSampler,
    paths_from_split_manifest,
)
from residual_multi import load_residual_checkpoint  # noqa: E402
from train_head_fast import Head3D, HeadConfig, build_head_features, initialize_coupled  # noqa: E402
from incremental_screen import (  # noqa: E402
    REGISTERED_ARMS,
    random_phase_gate,
    tke_gate,
    validate_registered_arms,
)


def _write_h5(path: Path, length: int) -> None:
    values = np.arange(length * 4 * 6, dtype=np.float32).reshape(length, 4, 6)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("u", data=values)
        handle.create_dataset("v", data=values + 1)


def _paths(tmp_path: Path) -> list[Path]:
    paths = [tmp_path / "a.h5", tmp_path / "b.h5"]
    _write_h5(paths[0], 100)
    _write_h5(paths[1], 121)
    return paths


def _selected(dataset: H5WindowDataset, indices: list[int]) -> list[tuple[str, int]]:
    return [(dataset.refs[index].path.name, dataset.refs[index].start) for index in indices]


def test_forced_zero_random_phase_matches_fixed_windows_and_global_shuffle(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    fixed = H5WindowDataset(paths, stride=20, sub_sample=2, window_mode="fixed")
    phased = H5WindowDataset(paths, stride=20, sub_sample=2, window_mode="random_phase")
    sampler = RandomPhaseWindowSampler(phased, seed=41)
    sampler.set_epoch(3, phases={path.name: 0 for path in paths})

    fixed_refs = _selected(fixed, list(range(len(fixed))))
    phased_refs = _selected(phased, list(iter(sampler)))
    assert sorted(phased_refs) == fixed_refs
    assert phased_refs != fixed_refs


def test_random_phase_is_deterministic_by_seed_and_changes_by_epoch(tmp_path: Path) -> None:
    dataset = H5WindowDataset(_paths(tmp_path), stride=20, sub_sample=2, window_mode="random_phase")
    first = RandomPhaseWindowSampler(dataset, seed=41)
    second = RandomPhaseWindowSampler(dataset, seed=41)
    first.set_epoch(2)
    second.set_epoch(2)

    assert first.phases == second.phases
    assert list(first) == list(second)
    old_phases = first.phases.copy()
    first.set_epoch(3)
    assert first.phases != old_phases
    assert all(0 <= phase < 20 for phase in first.phases.values())


def test_equalized_phase_sampler_keeps_per_trajectory_quota_constant(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    dataset = H5WindowDataset(paths, stride=20, sub_sample=2, window_mode="random_phase")
    sampler = RandomPhaseWindowSampler(dataset, seed=41, equalize_phase_counts=True)
    counts_by_epoch: list[dict[str, int]] = []
    for epoch in range(4):
        sampler.set_epoch(epoch)
        counts = {path.name: 0 for path in paths}
        for name, _ in _selected(dataset, list(sampler)):
            counts[name] += 1
        counts_by_epoch.append(counts)

    assert all(counts == counts_by_epoch[0] for counts in counts_by_epoch[1:])


def test_residual_checkpoint_restores_full_model_state_strictly(tmp_path: Path) -> None:
    source = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    target = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    checkpoint = tmp_path / "model.pth"
    torch.save({"model_state_dict": source.state_dict(), "iteration": 73500}, checkpoint)

    metadata = load_residual_checkpoint(target, checkpoint, torch.device("cpu"))

    assert metadata["iteration"] == 73500
    for expected, actual in zip(source.parameters(), target.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_head_delta_features_append_exact_uv_correction() -> None:
    x = torch.randn(1, 20, 2, 3, 3)
    base = torch.randn(1, 20, 2, 3, 3)
    corrected = base.clone()
    corrected[..., 0] += 0.25
    corrected[..., 1] -= 0.5

    original = build_head_features(x, base, corrected, include_pressure=True, include_delta=False)
    residual_aware = build_head_features(x, base, corrected, include_pressure=True, include_delta=True)

    assert residual_aware.shape[-1] == original.shape[-1] + 2
    torch.testing.assert_close(residual_aware[..., : original.shape[-1]], original)
    torch.testing.assert_close(residual_aware[..., -2], torch.full_like(residual_aware[..., -2], 0.25))
    torch.testing.assert_close(residual_aware[..., -1], torch.full_like(residual_aware[..., -1], -0.5))


def test_head_delta_features_require_corrected_prediction() -> None:
    x = torch.randn(1, 20, 2, 3, 3)
    base = torch.randn(1, 20, 2, 3, 3)
    with pytest.raises(ValueError, match="corrected prediction"):
        build_head_features(x, base, None, include_pressure=True, include_delta=True)


def test_head_config_adds_exactly_two_channels_for_delta() -> None:
    plain = HeadConfig(include_delta=False)
    aware = HeadConfig(include_delta=True)
    assert aware.include_delta is True
    assert plain.include_delta is False


def test_registered_residual_arms_match_frozen_protocol() -> None:
    assert REGISTERED_ARMS == {
        "R0": {"tke": 0.06, "window_mode": "fixed"},
        "R1": {"tke": 0.09, "window_mode": "fixed"},
        "R2": {"tke": 0.12, "window_mode": "fixed"},
        "R3": {"tke": 0.06, "window_mode": "random_phase"},
    }
    validate_registered_arms(REGISTERED_ARMS)


def test_registered_arm_validation_rejects_extra_sweep() -> None:
    arms = dict(REGISTERED_ARMS)
    arms["R4"] = {"tke": 0.2, "window_mode": "fixed"}
    with pytest.raises(ValueError, match="exactly R0-R3"):
        validate_registered_arms(arms)


def test_tke_gate_uses_raw_errors_and_fixed_time_composite() -> None:
    control = {"rel_l2_raw": 1.0, "tke_raw": 2.0, "mvpe_raw": 4.0, "final_est": 81.0}
    candidate = {"rel_l2_raw": 1.005, "tke_raw": 1.92, "mvpe_raw": 4.02, "final_est": 81.01}
    assert tke_gate(control, candidate)["pass"] is True
    candidate["final_est"] = 80.99
    assert tke_gate(control, candidate)["pass"] is False


def test_random_phase_gate_requires_two_metrics_and_point_one_five_composite() -> None:
    control = {"rel_l2_raw": 1.0, "tke_raw": 2.0, "mvpe_raw": 4.0, "final_est": 81.0}
    candidate = {"rel_l2_raw": 0.99, "tke_raw": 1.99, "mvpe_raw": 4.02, "final_est": 81.15}
    assert random_phase_gate(control, candidate)["pass"] is True
    candidate["final_est"] = 81.149
    assert random_phase_gate(control, candidate)["pass"] is False


def test_split_manifest_never_materializes_locked_final(tmp_path: Path) -> None:
    for name in ("train.h5", "dev.h5", "final.h5"):
        _write_h5(tmp_path / name, 80)
    manifest = tmp_path / "split.json"
    manifest.write_text(
        '{"train":[{"file":"train.h5"}],"dev":[{"file":"dev.h5"}],'
        '"final":[{"file":"final.h5"}]}',
        encoding="utf-8",
    )
    train, dev = paths_from_split_manifest(tmp_path, manifest)
    assert [path.name for path in train] == ["train.h5"]
    assert [path.name for path in dev] == ["dev.h5"]
    assert "final.h5" not in {path.name for path in train + dev}


def test_colleague_manifest_can_explicitly_allow_train_dev_overlap(tmp_path: Path) -> None:
    for name in ("a.h5", "b.h5"):
        _write_h5(tmp_path / name, 80)
    manifest = tmp_path / "colleague.json"
    manifest.write_text(
        '{"protocol":"colleague_dev16_seed41_all81","train":["a.h5","b.h5"],"dev":["b.h5"]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlap"):
        paths_from_split_manifest(tmp_path, manifest)
    train, dev = paths_from_split_manifest(tmp_path, manifest, allow_train_dev_overlap=True)
    assert [path.name for path in train] == ["a.h5", "b.h5"]
    assert [path.name for path in dev] == ["b.h5"]


def test_delta_head_is_identical_to_plain_head_when_delta_is_zero() -> None:
    plain = Head3D(HeadConfig(hidden=8, blocks=1, include_delta=False))
    aware = Head3D(HeadConfig(hidden=8, blocks=1, include_delta=True))
    initialize_coupled(plain, seed=41)
    initialize_coupled(aware, seed=41)
    x = torch.randn(2, 20, 2, 3, 3)
    base = torch.randn(2, 20, 2, 3, 3)
    torch.testing.assert_close(plain(x, base, base), aware(x, base, base))
