from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from build_p0a_n2_submission import (  # noqa: E402
    generate_submission_module,
    package_file_inventory,
    validate_checkpoint_for_submission,
)


def test_validate_checkpoint_rejects_non_15300_iteration(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"iteration": 1}, checkpoint)

    with pytest.raises(ValueError, match="15300"):
        validate_checkpoint_for_submission(checkpoint, required_iteration=15_300)


def test_package_inventory_rejects_training_artifacts(tmp_path):
    (tmp_path / "optimizer.pth").write_bytes(b"x")

    with pytest.raises(ValueError, match="optimizer"):
        package_file_inventory(tmp_path)


def test_generated_submission_uses_file_relative_singleton_inference(tmp_path):
    generate_submission_module(tmp_path)
    source = (tmp_path / "submission.py").read_text(encoding="utf-8")

    assert "Path(__file__).resolve().parent" in source
    assert "torch.inference_mode" in source
    assert "model.eval()" in source
    assert "metadata=None" in source
    assert "h5py" not in source
    assert "http" not in source
    assert "_MODEL is None" in source
