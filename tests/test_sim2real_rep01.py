import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.realpde_sim2real_rep01 import TinyLinearProbe, coarse_target


def test_coarse_target_preserves_shape_and_is_temporal_mean():
    x = np.arange(2 * 4 * 3 * 5 * 2, dtype=np.float32).reshape(2, 4, 3, 5, 2)
    got = coarse_target(x)
    assert got.shape == (2, 3, 5, 2)
    np.testing.assert_allclose(got, x.mean(axis=1))


def test_tiny_probe_maps_frozen_cno_features_to_future_uv():
    probe = TinyLinearProbe(in_dim=6, out_dim=4)
    features = torch.zeros(2, 3, 6)
    output = probe(features)
    assert output.shape == (2, 3, 4)
    assert all(not p.requires_grad is False for p in probe.parameters())
