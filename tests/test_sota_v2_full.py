from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import realpde_sota_v2_full as full  # noqa: E402


def test_full_data_schedule_is_dense_epoch_mapped_from_dev_recipe():
    assert full.REFERENCE_DENSE_WINDOWS == 40_488
    assert full.FULL_DENSE_WINDOWS == 66_755
    assert full.REFERENCE_UPDATES_PER_EPOCH == 5_061
    assert full.FULL_UPDATES_PER_EPOCH == 8_344
    assert full.map_reference_update(30_000) == 49_461
    assert full.map_reference_update(31_000) == 51_109
    assert full.map_reference_update(32_500) == 53_582
    assert full.map_reference_update(35_000) == 57_704
    assert full.STAGE_A_END == 49_461
    assert full.FINAL_UPDATE == 53_582
    assert full.MILESTONES == (12_365, 24_730, 32_974, 41_217, 49_461, 51_109, 53_582)


def test_stage_b_begins_only_after_epoch_mapped_stage_a_boundary():
    assert full.stage_config(49_461) == {"stage": "A", "lr": 1e-5, "extra_rel": 0.0}
    assert full.stage_config(49_462) == {"stage": "B", "lr": 3e-6, "extra_rel": 0.027514}
    assert full.stage_config(53_582)["stage"] == "B"
    with pytest.raises(ValueError):
        full.stage_config(53_583)


def test_full_protocol_requires_effective_batch_8_and_frozen_schedule():
    base = dict(seed=20260901, final_update=53_582, milestones=list(full.MILESTONES), workers=2)
    full.validate_protocol(Namespace(**base, micro_batch=8, accumulation_steps=1))
    full.validate_protocol(Namespace(**base, micro_batch=4, accumulation_steps=2))
    with pytest.raises(ValueError):
        full.validate_protocol(Namespace(**base, micro_batch=4, accumulation_steps=1))
    with pytest.raises(ValueError):
        full.validate_protocol(
            Namespace(**(base | {"final_update": 57_704}), micro_batch=8, accumulation_steps=1)
        )


def test_released_paths_requires_exactly_82_h5_files(tmp_path: Path):
    for index in range(82):
        (tmp_path / f"{index:03d}.h5").touch()
    paths = full.released_paths(tmp_path)
    assert len(paths) == 82
    assert paths == sorted(paths)

    (tmp_path / "extra.h5").touch()
    with pytest.raises(ValueError, match="exactly 82"):
        full.released_paths(tmp_path)
