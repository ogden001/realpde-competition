import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from realpde_mf_energy_campaign01 import reconstruct_with_gain  # noqa: E402


class GainInvariantTests(unittest.TestCase):
    def test_zero_gain_is_mf_reconstruction_and_fluctuation_is_zero_mean(self):
        torch.manual_seed(20260901)
        mean_raw = torch.randn(2, 20, 3, 4, 2)
        fluct_raw = torch.randn(2, 20, 3, 4, 2)
        pressure = torch.randn(2, 20, 3, 4, 1)
        prediction, fluctuation, alpha = reconstruct_with_gain(mean_raw, fluct_raw, pressure, torch.zeros(2, 1, 3, 4))
        expected_fluct = fluct_raw - fluct_raw.mean(dim=1, keepdim=True)
        expected_mean = mean_raw.mean(dim=1, keepdim=True).expand_as(mean_raw)
        self.assertLessEqual(float((alpha - 1).abs().max()), 1e-7)
        self.assertLessEqual(float((fluctuation - expected_fluct).abs().max()), 1e-7)
        self.assertLessEqual(float((prediction[..., :2] - (expected_mean + expected_fluct)).abs().max()), 1e-7)
        self.assertLessEqual(float((prediction[..., 2:3] - pressure).abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
