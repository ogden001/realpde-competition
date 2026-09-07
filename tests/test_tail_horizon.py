import torch

from tools.realpde_tail_horizon import blend_tail, normalized_tail_weights, weighted_mse


def test_normalized_tail_weights_preserve_mean_one():
    weights = normalized_tail_weights(2.0, horizons=20, tail_count=2)
    assert weights.shape == (20,)
    assert torch.isclose(weights.mean(), torch.tensor(1.0))
    assert weights[-1] == weights[-2]
    assert weights[-1] > weights[0]


def test_weighted_mse_factor_one_matches_official_uv_mse_and_ignores_pressure():
    pred = torch.arange(60.0).reshape(1, 20, 1, 1, 3)
    target = torch.zeros_like(pred)
    pred[..., 2] = 999.0
    got = weighted_mse(pred, target, tail_factor=1.0)
    expected = torch.mean((pred[..., :2] - target[..., :2]) ** 2)
    assert torch.allclose(got, expected)


def test_blend_tail_changes_only_last_two_horizons():
    cno = torch.ones(1, 20, 1, 1, 3)
    cno[..., 2] = 7.0
    persist = torch.zeros_like(cno)
    blended = blend_tail(cno, persist, alpha19=0.25, alpha20=0.75)
    assert torch.allclose(blended[:, :18], cno[:, :18])
    assert torch.allclose(blended[:, 18, ..., :2], torch.full_like(blended[:, 18, ..., :2], 0.25))
    assert torch.allclose(blended[:, 19, ..., :2], torch.full_like(blended[:, 19, ..., :2], 0.75))
    assert torch.allclose(blended[..., 2], cno[..., 2])
