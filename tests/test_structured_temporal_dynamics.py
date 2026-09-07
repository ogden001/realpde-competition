import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))


def test_future_deltas_anchor_first_future_frame_to_past_last_frame():
    from realpde_structured_temporal_dynamics import future_deltas

    past = torch.zeros(1, 20, 1, 1, 3)
    future = torch.zeros(1, 20, 1, 1, 3)
    future[:, :, ..., 0] = torch.arange(1, 21).view(1, 20, 1, 1)
    delta = future_deltas(past, future)

    assert delta.shape == future[..., :2].shape
    assert torch.equal(delta[..., 0], torch.ones_like(delta[..., 0]))
    assert torch.equal(delta[..., 1], torch.zeros_like(delta[..., 1]))


def test_vorticity_uses_dv_dx_minus_du_dy_on_interior():
    from realpde_structured_temporal_dynamics import vorticity

    yy, xx = torch.meshgrid(torch.arange(5.0), torch.arange(6.0), indexing="ij")
    field = torch.zeros(1, 1, 5, 6, 3)
    field[..., 0] = 3 * yy
    field[..., 1] = 2 * xx
    omega = vorticity(field, dx=1.0, dy=1.0)

    assert omega.shape == (1, 1, 3, 4)
    assert torch.allclose(omega, torch.full_like(omega, -1.0))


def test_zero_initialized_temporal_mixer_has_exact_prediction_parity_and_trainable_output():
    from realpde_structured_temporal_dynamics import TemporalMixer, apply_temporal_mixer

    torch.manual_seed(7)
    base = torch.randn(2, 20, 4, 5, 3, requires_grad=True)
    mixer = TemporalMixer()
    mixed = apply_temporal_mixer(base, mixer)

    assert torch.equal(mixed, base)
    assert mixer.output.weight.requires_grad
    mixed[..., :2].square().mean().backward()
    assert mixer.output.weight.grad is not None
    assert torch.count_nonzero(mixer.output.weight.grad).item() > 0
    assert torch.equal(mixed[..., 2], base[..., 2])
