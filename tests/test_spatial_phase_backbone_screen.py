from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
COLLEAGUE = TOOLS / "colleague_80pt"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(COLLEAGUE))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


SCREEN = load_module(
    "spatial_phase_backbone_screen",
    TOOLS / "run_spatial_phase_backbone_screen.py",
)


def value(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_screen_reuses_original_clean_stage1_trainer(tmp_path: Path) -> None:
    args = type(
        "Args",
        (),
        {
            "real_root": Path("/data/real"),
            "sim_pretrain": Path("/data/sim_real_cno.pth"),
            "kit_root": Path("/repo/model"),
            "workers": 4,
        },
    )()
    cmd = SCREEN.build_candidate_command(
        args,
        train_manifest=Path("/tmp/train_dev.json"),
        out_dir=tmp_path / "candidate",
    )
    assert cmd[3].endswith("tools/colleague_80pt/train_clean_baseline_cno.py")
    assert value(cmd, "--updates") == "8723"
    assert value(cmd, "--eval-interval") == "1000"
    assert value(cmd, "--batch-size") == "8"
    assert value(cmd, "--lr") == "0.0001"
    assert value(cmd, "--seed") == "41"
    assert value(cmd, "--spatial-phase-mix-prob") == "0.5"
    assert value(cmd, "--spatial-phase-seed") == "20260925"
    assert value(cmd, "--protocol-label") == SCREEN.PROTOCOL


def test_historical_baseline_is_frozen_to_clean_run() -> None:
    assert SCREEN.BASELINE_EXECUTION_COMMIT == "65e4f0f1d029eea47c6ba96364ce889e22d437d8"
    assert SCREEN.BASELINE[8000] == {
        "rel_l2_raw": 0.099606201,
        "tke_raw": 0.778031290,
        "mvpe_raw": 0.080289602,
    }
    assert SCREEN.BASELINE[8723] == {
        "rel_l2_raw": 0.099932298,
        "tke_raw": 0.792903721,
        "mvpe_raw": 0.080299616,
    }


def test_gate_uses_matched_8000_and_requires_visible_gain() -> None:
    base = SCREEN.BASELINE[8000]
    good = {
        "rel_l2_raw": base["rel_l2_raw"] * 0.99,
        "tke_raw": base["tke_raw"] * 0.99,
        "mvpe_raw": base["mvpe_raw"] * 0.997,
    }
    flat = dict(base)
    assert SCREEN.decision(good)["status"] == "GO"
    assert SCREEN.decision(flat)["status"] == "NO_GO"


def test_clean_cno_default_remains_p00_and_candidate_is_training_only() -> None:
    text = (COLLEAGUE / "train_clean_baseline_cno.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--spatial-phase-mix-prob", type=float, default=0.0)' in text
    assert "spatial_phase_mix_prob=args.spatial_phase_mix_prob" in text
    assert '"validation_phase": "P00"' in text


def test_screen_has_no_residual_holdout_or_submission_execution() -> None:
    text = (TOOLS / "run_spatial_phase_backbone_screen.py").read_text(encoding="utf-8").lower()
    assert "residual_multi.py" not in text
    assert "holdout_eval_manifest" not in text
    assert "analyze_checkpoints.py" not in text
    assert "submission.py" not in text
    assert "package_submission" not in text
