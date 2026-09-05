import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import realpde_checkpoint_soup as soup


def test_normalize_weights_and_average_state_dict():
    weights = soup.normalize_weights([2.0, 1.0])
    assert weights == pytest.approx([2.0 / 3.0, 1.0 / 3.0])
    a = {"w": torch.tensor([1.0, 3.0]), "n": torch.tensor(4, dtype=torch.int64)}
    b = {"w": torch.tensor([4.0, 0.0]), "n": torch.tensor(4, dtype=torch.int64)}
    out = soup.average_state_dicts([a, b], [0.25, 0.75])
    assert torch.allclose(out["w"], torch.tensor([3.25, 0.75]))
    assert out["n"].item() == 4


def test_average_state_dict_rejects_nonfloating_mismatch():
    a = {"w": torch.tensor([1.0]), "n": torch.tensor(4, dtype=torch.int64)}
    b = {"w": torch.tensor([2.0]), "n": torch.tensor(5, dtype=torch.int64)}
    with pytest.raises(ValueError, match="non-floating state differs"):
        soup.average_state_dicts([a, b], [0.5, 0.5])


def test_prediction_blend_is_weighted_average():
    a = torch.tensor([1.0, 3.0]).numpy()
    b = torch.tensor([5.0, 1.0]).numpy()
    out = soup.blend_predictions([a, b], [0.25, 0.75])
    assert out.tolist() == pytest.approx([4.0, 1.5])


def test_candidate_set_is_small_and_preregistered():
    assert set(soup.SOUPS) == {
        "soup_26240_27880",
        "soup_26240_30340",
        "soup_27880_30340",
        "soup_equal3",
        "soup_balanced3",
    }
    assert set(soup.ENSEMBLES) == {
        "ens_equal3",
        "ens_balanced3",
    }
    for weights in list(soup.SOUPS.values()) + list(soup.ENSEMBLES.values()):
        assert sum(weights) == pytest.approx(1.0)
        assert len(weights) == 3
