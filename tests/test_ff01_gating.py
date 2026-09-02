import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from realpde_ff01_gating import (  # noqa: E402
    CNOFiLM,
    build_prior_features,
    count_parameters,
)


def test_frozen_feature_packages_have_fixed_ten_channel_contract():
    x = torch.arange(2 * 20 * 4 * 6 * 3, dtype=torch.float32).reshape(2, 20, 4, 6, 3)

    g0 = build_prior_features(x, "G0")
    g1 = build_prior_features(x, "G1")
    g2 = build_prior_features(x, "G2")

    assert g0.shape == g1.shape == g2.shape == (2, 10, 4, 6)
    torch.testing.assert_close(g0[:, 0], x[:, -1, :, :, 0])
    torch.testing.assert_close(g0[:, 1], x[:, -1, :, :, 1])
    assert torch.count_nonzero(g0[:, 2:]) == 0
    torch.testing.assert_close(g1[:, 0], x[..., 0].mean(dim=1))
    torch.testing.assert_close(g1[:, 1], x[..., 1].mean(dim=1))
    torch.testing.assert_close(g1[:, 2], x[..., 0].std(dim=1, unbiased=False))
    torch.testing.assert_close(g1[:, 3], x[..., 1].std(dim=1, unbiased=False))
    torch.testing.assert_close(g1[:, 4], x[:, -1, :, :, 0] - x[:, -2, :, :, 0])
    torch.testing.assert_close(g1[:, 5], x[:, -1, :, :, 1] - x[:, -2, :, :, 1])
    assert torch.count_nonzero(g1[:, 6:]) == 0
    torch.testing.assert_close(g2[:, :6], g1[:, :6])
    assert torch.count_nonzero(g2[:, 6:]) > 0


def test_film_projection_is_an_exact_initial_noop():
    model = CNOFiLM(_ToyCNO(channels=8), feature_group="G1")
    z = torch.randn(2, 8, 20, 2, 3)
    x = torch.randn(2, 20, 4, 6, 3)

    out = model.apply_conditioning(z, build_prior_features(x, "G1"))

    torch.testing.assert_close(out, z, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(model.conditioner.projection.weight) == 0
    assert torch.count_nonzero(model.conditioner.projection.bias) == 0


def test_all_candidates_share_conditioner_parameter_count_and_latent_time_broadcast():
    models = [CNOFiLM(_ToyCNO(channels=8), feature_group=g) for g in ("G0", "G1", "G2")]
    assert len({count_parameters(m) for m in models}) == 1
    assert len({count_parameters(m.conditioner) for m in models}) == 1

    x = torch.randn(2, 20, 4, 6, 3)
    z = torch.randn(2, 8, 20, 2, 3)
    conditioned = models[0].apply_conditioning(z, build_prior_features(x, "G0"))
    assert conditioned.shape == z.shape


class _ToyCNO(nn.Module):
    """Minimal structural stand-in used to test the fixed FiLM contract."""

    def __init__(self, channels: int):
        super().__init__()
        self.N_layers = 1
        self.encoder_features = [4, channels]
        self.encoder_sizes = [4, 2]
        self.decoder_sizes = [2, 4]
        self.decoder_features_in = [channels]
        self.decoder_features_out = [4]
        self.inv_features = [channels, 8]
        self.N_res = 1
        self.N_res_neck = 1
        self.lift = nn.Conv3d(3, 4, 1)
        self.encoder = nn.ModuleList([nn.Conv3d(4, channels, 1)])
        self.res_nets = nn.Sequential(nn.Identity(), nn.Identity())
        self.ED_expansion = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.decoder_inv = nn.ModuleList([nn.Identity()])
        self.decoder = nn.ModuleList([nn.Conv3d(channels, 4, 1)])
        self.project = nn.Conv3d(8, 3, 1)
        self.add_inv = False

