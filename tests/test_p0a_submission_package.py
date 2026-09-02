from __future__ import annotations

import sys
from pathlib import Path

import pytest
import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from build_p0a_n2_submission import (  # noqa: E402
    copy_vendored_sources,
    generate_submission_module,
    package_file_inventory,
    validate_checkpoint_for_submission,
)
from realpde_p0a_submission_smoke import prediction_error_summary  # noqa: E402


def test_validate_checkpoint_rejects_non_15300_iteration(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"iteration": 1}, checkpoint)

    with pytest.raises(ValueError, match="15300"):
        validate_checkpoint_for_submission(checkpoint, required_iteration=15_300)


def test_package_inventory_rejects_training_artifacts(tmp_path):
    (tmp_path / "optimizer.pth").write_bytes(b"x")

    with pytest.raises(ValueError, match="optimizer"):
        package_file_inventory(tmp_path)


def test_package_inventory_allows_explicit_license_names_only(tmp_path):
    (tmp_path / "submission.py").write_text("", encoding="utf-8")
    (tmp_path / "model.pth").write_bytes(b"model")
    license_dir = tmp_path / "_vendor" / "einops"
    license_dir.mkdir(parents=True)
    (license_dir / "LICENSE").write_text("license", encoding="utf-8")
    assert any(item["path"] == "_vendor/einops/LICENSE" for item in package_file_inventory(tmp_path))
    (license_dir / "README.txt").write_text("readme", encoding="utf-8")
    with pytest.raises(ValueError, match="README.txt"):
        package_file_inventory(tmp_path)


def test_copy_vendored_sources_keeps_license_and_excludes_nonruntime_markers(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    (source / "module.py").write_text("x = 1\n")
    (source / "LICENSE").write_text("license")
    (source / "README.txt").write_text("readme")
    (source / "py.typed").write_text("")

    copy_vendored_sources(source, destination)

    assert (destination / "module.py").is_file()
    assert (destination / "LICENSE").is_file()
    assert (destination / "README.txt").is_file()
    assert not (destination / "py.typed").exists()


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


def test_error_summary_is_zero_for_equal_predictions():
    result = prediction_error_summary(np.ones((1, 2), np.float32), np.ones((1, 2), np.float32))

    assert result == {"max_abs_error": 0.0, "max_rel_error": 0.0}


def test_error_summary_uses_safe_relative_denominator():
    result = prediction_error_summary(np.zeros(1, np.float32), np.ones(1, np.float32))

    assert result["max_rel_error"] > 0.0
