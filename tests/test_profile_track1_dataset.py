import io

import h5py
import numpy as np

from tools.profile_track1_dataset import INPUT_NAMES, TARGET_NAMES, load_manifest, trajectory_row
from tools.audit_track1_duplicates import window_starts


def make_h5(frames=60, height=4, width=5):
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as h:
        h.create_dataset("u", data=np.ones((frames, height, width), dtype=np.float32))
        h.create_dataset("v", data=np.zeros((frames, height, width), dtype=np.float32))
        h.create_dataset("x", data=np.arange(width, dtype=np.float32)[None, :].repeat(height, axis=0))
        h.create_dataset("y", data=np.arange(height, dtype=np.float32)[:, None].repeat(width, axis=1))
        h.create_dataset("aoa", data=5.0)
        h.create_dataset("re", data=7500.0)
    return buffer.getvalue()


def test_final_trajectory_row_is_input_only():
    row, vector = trajectory_row("7575_0.h5", "final", make_h5(), 20, 20, 20)

    assert len(vector) == len(INPUT_NAMES) == 17
    assert row["split"] == "final"
    assert row["frames"] == 60
    assert row["windows"] == 2
    assert set(TARGET_NAMES).isdisjoint(row)
    assert not any("future" in key.lower() or "target" in key.lower() for key in row)


def test_manifest_loader_requires_all_three_audit_splits(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"train": [], "dev": [], "final": []}')

    loaded = load_manifest(manifest)

    assert set(("train", "dev", "final")) <= loaded.keys()


def test_duplicate_audit_uses_formal_window_protocol():
    assert list(window_starts(868, 20, 20, 20)) == list(range(0, 840, 20))
    assert list(window_starts(607, 20, 20, 20)) == list(range(0, 580, 20))
