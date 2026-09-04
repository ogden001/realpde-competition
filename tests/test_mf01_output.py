import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from realpde_mf01 import factorized_reconstruct


class MF01OutputTests(unittest.TestCase):
    def test_zero_mean_fluctuation_reconstructs_direct_projection(self):
        torch.manual_seed(20260901)
        direct = torch.randn(2, 20, 4, 5, 2)
        mean_raw = direct.mean(dim=1, keepdim=True).expand_as(direct)
        fluct_raw = direct

        reconstructed, fluctuation = factorized_reconstruct(mean_raw, fluct_raw)

        self.assertLessEqual(float(fluctuation.mean(dim=1).abs().max()), 1e-7)
        self.assertLessEqual(float((reconstructed - direct).abs().max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
