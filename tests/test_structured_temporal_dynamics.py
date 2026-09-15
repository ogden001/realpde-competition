import sys
from pathlib import Path

import csv

import pytest
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


def test_latent_temporal_conv_is_time_only_and_zero_init_preserves_prediction():
    from realpde_structured_temporal_dynamics_r2 import LatentTemporalConv

    torch.manual_seed(3)
    z = torch.randn(2, 12, 20, 4, 5)
    module = LatentTemporalConv(12, 8)
    assert module.input.kernel_size == (3, 1, 1)
    assert module.output.kernel_size == (3, 1, 1)
    assert torch.equal(module(z), z)
    module(z).square().mean().backward()
    assert torch.count_nonzero(module.output.weight.grad).item() > 0


def test_latent_attention_is_future20_only_and_zero_init_preserves_prediction():
    from realpde_structured_temporal_dynamics_r2 import LatentTemporalAttention

    torch.manual_seed(4)
    z = torch.randn(1, 8, 20, 2, 3)
    module = LatentTemporalAttention(8, 4)
    assert module.attention.num_heads == 4
    assert torch.equal(module(z), z)
    module(z).square().mean().backward()
    assert torch.count_nonzero(module.output.weight.grad).item() > 0


def test_r2_rejects_accumulation_and_nonphysical_batch():
    from realpde_structured_temporal_dynamics_r2 import validate_r2_protocol
    import pytest

    validate_r2_protocol(8, 1)
    with pytest.raises(ValueError): validate_r2_protocol(2, 4)


def test_long_final_protocol_is_frozen_and_v1_lambda_is_exact():
    from realpde_vorticity_long_final import FIXED_LAMBDA_VORT, validate_long_protocol

    validate_long_protocol(arm="V1", batch_size=8, accumulation_steps=1, seed=20260901,
                           lr=1e-5, start_update=3000, final_update=15000,
                           lambda_vort=FIXED_LAMBDA_VORT)
    with pytest.raises(ValueError):
        validate_long_protocol(arm="V1", batch_size=2, accumulation_steps=4, seed=20260901,
                               lr=1e-5, start_update=3000, final_update=15000,
                               lambda_vort=FIXED_LAMBDA_VORT)
    with pytest.raises(ValueError):
        validate_long_protocol(arm="V1", batch_size=8, accumulation_steps=1, seed=20260901,
                               lr=1e-5, start_update=3000, final_update=15000,
                               lambda_vort=15.5)


def test_long_final_gate_uses_registered_late_thresholds_only():
    from realpde_vorticity_long_final_summary import compute_gate

    rows = [
        {"update": str(update), "rel_l2_improvement_pct": str(value),
         "tke_improvement_pct": str(tke), "mvpe_improvement_pct": str(mvpe)}
        for update, value, tke, mvpe in ((9000, 2.0, -2.0, 4.0), (12000, 3.0, 0.0, 6.0), (15000, 1.0, 0.0, -2.0))
    ]
    gate = compute_gate(rows)
    assert gate["FINAL_GATE"] == "PARK"
    assert gate["median_checks"]["median_rel_improvement_ge_2pct"]
    assert gate["median_checks"]["median_mvpe_improvement_ge_4pct"]
    assert not gate["at_least_two_late_checkpoints_pass"]
