from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import realpde_residual_corrector_projection as proj  # noqa: E402


def _toy_pair() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    base = torch.randn(3, 20, 5, 7, 3)
    corrected = base.clone()
    corrected[..., :2] = 0.8 * base[..., :2] + 0.15
    base[..., 2] = 0.0
    corrected[..., 2] = 0.0
    return base, corrected


def _window_fluct_energy(x: torch.Tensor) -> torch.Tensor:
    uv = x[..., :2]
    fluct = uv - uv.mean(dim=1, keepdim=True)
    return fluct.square().sum(dim=(1, 2, 3, 4))


def _spatial_tke_map(x: torch.Tensor) -> torch.Tensor:
    uv = x[..., :2]
    fluct = uv - uv.mean(dim=1, keepdim=True)
    return 0.5 * fluct.square().mean(dim=1).sum(dim=-1)


def test_projection_variants_are_frozen():
    assert proj.VARIANTS == ("base", "corrected", "window_energy", "spatial_tke_map")
    assert proj.BACKBONE_UPDATE == 32_500
    assert proj.CORRECTOR_UPDATE == 30_000


def test_window_energy_projection_preserves_corrected_mean_and_base_window_energy():
    base, corrected = _toy_pair()
    out = proj.window_energy_projection(base, corrected)

    assert torch.allclose(
        out[..., :2].mean(dim=1),
        corrected[..., :2].mean(dim=1),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        _window_fluct_energy(out),
        _window_fluct_energy(base),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.count_nonzero(out[..., 2]) == 0


def test_spatial_tke_map_projection_preserves_corrected_mean_and_base_tke_map():
    base, corrected = _toy_pair()
    out = proj.spatial_tke_map_projection(base, corrected)

    assert torch.allclose(
        out[..., :2].mean(dim=1),
        corrected[..., :2].mean(dim=1),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        _spatial_tke_map(out),
        _spatial_tke_map(base),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.count_nonzero(out[..., 2]) == 0


def test_both_projections_are_identity_when_corrected_equals_base():
    base, _ = _toy_pair()
    for fn in (proj.window_energy_projection, proj.spatial_tke_map_projection):
        out = fn(base, base)
        assert torch.allclose(out, base, atol=1e-6, rtol=1e-6)


def test_projection_rejects_shape_mismatch():
    base, corrected = _toy_pair()
    bad = corrected[:, :-1]
    for fn in (proj.window_energy_projection, proj.spatial_tke_map_projection):
        try:
            fn(base, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("shape mismatch must be rejected")
