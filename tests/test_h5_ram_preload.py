from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "colleague_80pt"
sys.path.insert(0, str(TOOLS))

from realpde_h5_feature_adapter_train import H5WindowDataset  # noqa: E402


def test_ram_preload_is_exactly_equivalent_to_direct_h5(tmp_path: Path) -> None:
    path = tmp_path / "traj.h5"
    u = np.arange(70 * 8 * 12, dtype=np.float32).reshape(70, 8, 12)
    v = u * 0.5 - 3.0
    with h5py.File(path, "w") as handle:
        handle.create_dataset("u", data=u)
        handle.create_dataset("v", data=v)

    direct = H5WindowDataset(
        [path], in_steps=20, out_steps=20, stride=1, sub_sample=2,
        include_pressure=False, window_mode="fixed", preload_to_ram=False,
    )
    cached = H5WindowDataset(
        [path], in_steps=20, out_steps=20, stride=1, sub_sample=2,
        include_pressure=False, window_mode="fixed", preload_to_ram=True,
    )

    assert direct.refs == cached.refs
    assert cached.cache_summary()["enabled"] is True
    assert cached.cache_bytes > 0

    for index in (0, len(direct) // 2, len(direct) - 1):
        dx, dy = direct[index]
        cx, cy = cached[index]
        torch.testing.assert_close(cx, dx, rtol=0, atol=0)
        torch.testing.assert_close(cy, dy, rtol=0, atol=0)
