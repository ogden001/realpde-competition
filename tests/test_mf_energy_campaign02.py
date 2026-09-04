import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from realpde_mf_energy_campaign02 import (  # noqa: E402
    scorer_tke_map,
    relative_tke_map_loss,
    relative_rms_loss,
    high_energy_fluctuation_loss,
    set_frozen_gain_trainable,
)


class EnergyLossTests(unittest.TestCase):
    def test_tke_map_matches_definition_and_is_zero_for_constant_time(self):
        x = torch.ones(2, 20, 3, 4, 2)
        self.assertTrue(torch.allclose(scorer_tke_map(x), torch.zeros(2, 3, 4)))
        y = torch.randn(2, 20, 3, 4, 2)
        expected = 0.5 * (y[..., 0].var(dim=1, unbiased=False) + y[..., 1].var(dim=1, unbiased=False))
        self.assertTrue(torch.allclose(scorer_tke_map(y), expected))

    def test_new_losses_are_zero_for_identical_finite_prediction(self):
        y = torch.randn(2, 20, 3, 4, 2)
        self.assertLess(float(relative_tke_map_loss(y, y)), 1e-7)
        self.assertLess(float(relative_rms_loss(y, y)), 1e-7)
        self.assertLess(float(high_energy_fluctuation_loss(y, y)), 1e-7)

    def test_new_losses_are_finite_on_zero_energy_targets(self):
        p = torch.randn(2, 20, 3, 4, 2)
        y = torch.zeros_like(p)
        for value in (relative_tke_map_loss(p, y), relative_rms_loss(p, y), high_energy_fluctuation_loss(p, y)):
            self.assertTrue(torch.isfinite(value))


class FrozenGainTests(unittest.TestCase):
    def test_only_gain_head_remains_trainable(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Linear(2, 2)
                self.gain_head = torch.nn.Linear(2, 1)
        model = M()
        set_frozen_gain_trainable(model)
        self.assertFalse(any(p.requires_grad for p in model.base.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.gain_head.parameters()))


if __name__ == "__main__":
    unittest.main()
