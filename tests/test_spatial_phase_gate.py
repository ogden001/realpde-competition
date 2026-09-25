from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
COLLEAGUE = ROOT / "tools" / "colleague_80pt"
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(COLLEAGUE))

from realpde_h5_feature_adapter_train import H5WindowDataset  # noqa: E402


def load_gate():
    path = ROOT / "tools" / "run_spatial_phase_gate.py"
    spec = importlib.util.spec_from_file_location("spatial_phase_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


GATE = load_gate()


def make_h5(path: Path, *, frames: int = 12, height: int = 6, width: int = 8) -> np.ndarray:
    yy = np.arange(height, dtype=np.float32).reshape(1, height, 1)
    xx = np.arange(width, dtype=np.float32).reshape(1, 1, width)
    tt = np.arange(frames, dtype=np.float32).reshape(frames, 1, 1)
    u = 10000.0 * tt + 100.0 * yy + xx
    v = -u
    with h5py.File(path, "w") as handle:
        handle["u"] = u
        handle["v"] = v
        handle["aoa"] = np.asarray(0.0, dtype=np.float32)
        handle["re"] = np.asarray(1000.0, dtype=np.float32)
    return u


def test_default_spatial_phase_is_exact_historical_p00(tmp_path: Path) -> None:
    path = tmp_path / "case.h5"
    u = make_h5(path)
    ds = H5WindowDataset(
        [path], in_steps=2, out_steps=2, stride=4, sub_sample=2,
        window_mode="fixed", preload_to_ram=True,
    )
    x, y = ds[0]
    expected = torch.from_numpy(u[:4, ::2, ::2])
    assert ds.spatial_phase(0) == (0, 0)
    assert ds.spatial_phase_counts() == {"P00": len(ds), "P01": 0, "P10": 0, "P11": 0}
    assert torch.equal(x[..., 0], expected[:2])
    assert torch.equal(y[..., 0], expected[2:])
    assert ds.cache_summary()["full_resolution_spatial_cache"] is False


def test_spatial_phase_uses_same_offset_for_past_and_future(tmp_path: Path) -> None:
    path = tmp_path / "case.h5"
    u = make_h5(path)
    ds = H5WindowDataset(
        [path], in_steps=2, out_steps=2, stride=1, sub_sample=2,
        window_mode="fixed", preload_to_ram=False,
        spatial_phase_mix_prob=1.0, spatial_phase_seed=123,
    )
    assert ds.spatial_phase_counts()["P00"] == 0
    for index in range(len(ds)):
        dy, dx = ds.spatial_phase(index)
        x, y = ds[index]
        start = ds.refs[index].start
        expected = torch.from_numpy(u[start:start + 4, dy::2, dx::2])
        assert torch.equal(x[..., 0], expected[:2])
        assert torch.equal(y[..., 0], expected[2:])


def test_spatial_phase_preload_matches_direct_hdf5_read(tmp_path: Path) -> None:
    path = tmp_path / "case.h5"
    make_h5(path, frames=20)
    kwargs = dict(
        paths=[path], in_steps=2, out_steps=2, stride=1, sub_sample=2,
        window_mode="fixed", spatial_phase_mix_prob=0.5, spatial_phase_seed=456,
    )
    direct = H5WindowDataset(preload_to_ram=False, **kwargs)
    cached = H5WindowDataset(preload_to_ram=True, **kwargs)
    assert direct.spatial_phase_counts() == cached.spatial_phase_counts()
    assert cached.cache_summary()["full_resolution_spatial_cache"] is True
    for index in range(len(direct)):
        dx, dy = direct[index], cached[index]
        assert torch.equal(dx[0], dy[0])
        assert torch.equal(dx[1], dy[1])


def test_phase_assignment_is_balanced_and_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "case.h5"
    make_h5(path, frames=104)
    a = H5WindowDataset(
        [path], in_steps=2, out_steps=2, stride=1, sub_sample=2,
        window_mode="fixed", spatial_phase_mix_prob=0.5, spatial_phase_seed=20260925,
    )
    b = H5WindowDataset(
        [path], in_steps=2, out_steps=2, stride=1, sub_sample=2,
        window_mode="fixed", spatial_phase_mix_prob=0.5, spatial_phase_seed=20260925,
    )
    counts = a.spatial_phase_counts()
    assert counts == b.spatial_phase_counts()
    assert abs(counts["P00"] / len(a) - 0.5) <= 0.01
    assert max(counts[k] for k in ("P01", "P10", "P11")) - min(
        counts[k] for k in ("P01", "P10", "P11")
    ) <= 1
    assert [a.spatial_phase(i) for i in range(len(a))] == [
        b.spatial_phase(i) for i in range(len(b))
    ]


def test_gate_runner_is_frozen_matched_continuation(tmp_path: Path) -> None:
    args = type("Args", (), {
        "real_root": Path("/data/real"),
        "out_root": Path("/tmp/spatial"),
        "strong_backbone": Path("/tmp/backbone.pth"),
        "strong_residual": Path("/tmp/residual.pth"),
        "kit_root": Path("/tmp/kit"),
        "workers": 4,
    })()
    control = GATE.build_arm_command(args, tmp_path / "control", spatial_mix_prob=0.0)
    candidate = GATE.build_arm_command(args, tmp_path / "candidate", spatial_mix_prob=0.5)

    def value(cmd: list[str], flag: str) -> str:
        return cmd[cmd.index(flag) + 1]

    for cmd in (control, candidate):
        assert value(cmd, "--updates") == "5000"
        assert value(cmd, "--batch-size") == "8"
        assert value(cmd, "--stride") == "1"
        assert value(cmd, "--eval-stride") == "20"
        assert value(cmd, "--train-window-mode") == "fixed"
        assert value(cmd, "--selection-metric") == "point_score"
        assert value(cmd, "--base-model") == "sota_v2_mf"
        assert value(cmd, "--resume-checkpoint") == "/tmp/residual.pth"
        assert value(cmd, "--lr") == "1e-05"
    assert value(control, "--spatial-phase-mix-prob") == "0.0"
    assert value(candidate, "--spatial-phase-mix-prob") == "0.5"


def test_spatial_gate_requires_small_broad_p00_gain() -> None:
    control = {"rel_l2_raw": 1.0, "tke_raw": 1.0, "mvpe_raw": 1.0}
    good = {"rel_l2_raw": 0.995, "tke_raw": 0.994, "mvpe_raw": 0.999}
    tradeoff = {"rel_l2_raw": 0.99, "tke_raw": 0.98, "mvpe_raw": 1.006}
    assert GATE.spatial_gate(good, control)["status"] == "GO"
    assert GATE.spatial_gate(tradeoff, control)["status"] == "NO_GO"


def test_gate_source_has_no_holdout_or_submission_execution_path() -> None:
    text = (ROOT / "tools" / "run_spatial_phase_gate.py").read_text(encoding="utf-8").lower()
    assert "holdout_accessed" in text
    assert "codabench_accessed" in text
    assert "holdout_eval_manifest" not in text
    assert "package_submission" not in text
    assert "submission.py" not in text

def test_state_parity_accepts_matching_empty_state_tensors(tmp_path: Path) -> None:
    left = tmp_path / "left.pth"
    right = tmp_path / "right.pth"
    payload = {
        "model_state_dict": {
            "weight": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "empty_buffer": torch.empty((0,), dtype=torch.float32),
            "counter": torch.tensor([3], dtype=torch.int64),
        }
    }
    torch.save(payload, left)
    torch.save(payload, right)
    assert GATE._state_parity(left, right) == 0.0


def test_state_parity_still_detects_nonempty_difference(tmp_path: Path) -> None:
    left = tmp_path / "left.pth"
    right = tmp_path / "right.pth"
    torch.save(
        {"model_state_dict": {
            "weight": torch.tensor([1.0, 2.0]),
            "empty_buffer": torch.empty((0,), dtype=torch.float32),
        }},
        left,
    )
    torch.save(
        {"model_state_dict": {
            "weight": torch.tensor([1.0, 2.1]),
            "empty_buffer": torch.empty((0,), dtype=torch.float32),
        }},
        right,
    )
    assert GATE._state_parity(left, right) > 0.09

