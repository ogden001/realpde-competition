from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from build_p0a_n2_submission import (  # noqa: E402
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
