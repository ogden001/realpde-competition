from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from realpde_runtime_context import (  # noqa: E402
    generate_artifact_manifest,
    inspect_checkpoint,
    inventory_data,
    resolve_checkpoint,
)


def _write_h5(path: Path, frames: int) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("u", data=np.zeros((frames, 64, 128), dtype=np.float32))
        handle.create_dataset("v", data=np.zeros((frames, 64, 128), dtype=np.float32))


def _write_checkpoint(path: Path, update: int, *, optimizer: bool = True) -> None:
    payload = {
        "iteration": update,
        "feature_set": "P0-A",
        "feature_config": {"dx": 0.1, "dy": -0.1},
        "loss_weights": {"mse": 1.0, "tke": 0.05},
        "metadata": {"run": "T1-COMP-P0A-N2-FULL"},
        "model_state_dict": {"weight": torch.ones(1)},
    }
    if optimizer:
        payload["optimizer_state_dict"] = {"state": {1: {}}, "param_groups": [{"lr": 1e-5}]}
    torch.save(payload, path)


def test_inventory_data_counts_trajectories_and_windows(tmp_path):
    _write_h5(tmp_path / "a.h5", 60)
    _write_h5(tmp_path / "b.h5", 40)

    result = inventory_data(tmp_path)

    assert result["trajectories"] == 2
    assert result["windows"] == 3


def test_inspect_checkpoint_reports_resume_semantics(tmp_path):
    checkpoint = tmp_path / "model_last.pth"
    _write_checkpoint(checkpoint, 15300, optimizer=True)

    result = inspect_checkpoint(checkpoint)

    assert result["iteration"] == 15300
    assert result["feature_set"] == "P0-A"
    assert result["optimizer_state"] is True
    assert result["feature_config"] == {"dx": 0.1, "dy": -0.1}
    assert len(result["sha256"]) == 64


def test_generate_artifact_manifest_distinguishes_resumable_checkpoints(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"run": "demo", "initial_update": 15300}), encoding="utf-8"
    )
    (run_dir / "status.json").write_text(
        json.dumps({"state": "DONE", "update": 31100}), encoding="utf-8"
    )
    _write_checkpoint(run_dir / "model_update_31100.pth", 31100, optimizer=True)
    _write_checkpoint(run_dir / "inference_only.pth", 31100, optimizer=False)

    manifest = generate_artifact_manifest(run_dir)

    assert manifest["status"]["state"] == "DONE"
    assert [row["name"] for row in manifest["resumable_checkpoints"]] == ["model_update_31100.pth"]
    assert {row["name"] for row in manifest["checkpoints"]} == {
        "model_update_31100.pth",
        "inference_only.pth",
    }


def test_resolve_checkpoint_requires_unique_semantic_match(tmp_path):
    first = tmp_path / "a.pth"
    second = tmp_path / "b.pth"
    _write_checkpoint(first, 15300, optimizer=True)
    _write_checkpoint(second, 30900, optimizer=True)
    snapshot = {"artifacts": [inspect_checkpoint(first), inspect_checkpoint(second)]}

    resolved = resolve_checkpoint(
        snapshot,
        iteration=15300,
        feature_set="P0-A",
        require_optimizer_state=True,
    )
    assert resolved["path"] == str(first.resolve())

    with pytest.raises(ValueError, match="no checkpoint"):
        resolve_checkpoint(
            snapshot,
            iteration=18860,
            feature_set="P0-A",
            require_optimizer_state=True,
        )
