import sys
from pathlib import Path

import csv

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


def test_memory_safe_batch_configuration_preserves_effective_batch():
    from realpde_structured_temporal_dynamics import validate_batch_configuration

    config = validate_batch_configuration(micro_batch_size=2, accumulation_steps=4, effective_batch_size=8, max_gpu_memory_gib=12.0)

    assert config["effective_batch_size"] == 8
    assert config["micro_batch_size"] == 2
    assert config["accumulation_steps"] == 4
    assert config["max_gpu_memory_gib"] == 12.0


def test_write_rows_accepts_metrics_with_late_added_columns(tmp_path):
    from realpde_structured_temporal_dynamics import write_rows

    output = tmp_path / "metrics.csv"
    write_rows(output, [{"update": 1500, "rel_l2": 0.1}, {"update": 2000, "rel_l2": 0.09, "inference_time": 0.01}])

    rows = list(csv.DictReader(output.open()))
    assert rows[0]["inference_time"] == ""
    assert rows[1]["inference_time"] == "0.01"
