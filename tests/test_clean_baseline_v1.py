from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "colleague_80pt"
sys.path.insert(0, str(TOOLS))

from run_clean_baseline_v1 import (  # noqa: E402
    BATCH_SIZE,
    CNO_UPDATES,
    EVAL_INTERVAL,
    PROTOCOL,
    RESIDUAL_UPDATES,
    SEED,
    build_stage1_command,
    build_stage2_command,
    validate_split_payload,
)


def test_canonical_split_is_clean_unseen_aoa10() -> None:
    payload = json.loads((ROOT / "configs" / "clean_baseline_v1_split.json").read_text())
    validate_split_payload(payload)
    assert payload["protocol"] == PROTOCOL
    assert len(payload["train"]) == 51
    assert len(payload["dev"]) == 12
    assert len(payload["holdout"]) == 18

    def names(key: str) -> list[str]:
        return [row["file"] if isinstance(row, dict) else row for row in payload[key]]

    train = names("train")
    dev = names("dev")
    holdout = names("holdout")
    assert all(not name.endswith("_10.h5") for name in train + dev)
    assert all(name.endswith("_10.h5") for name in holdout)
    assert len([name for name in dev if name.endswith("_0.h5")]) == 3
    assert len([name for name in dev if name.endswith("_5.h5")]) == 3
    assert len([name for name in dev if name.endswith("_15.h5")]) == 3
    assert len([name for name in dev if name.endswith("_20.h5")]) == 3


def test_stage1_command_freezes_dense_sampling_and_budget(tmp_path: Path) -> None:
    cmd = build_stage1_command(
        python=sys.executable,
        real_root=Path("/data/real"),
        train_manifest=Path("/tmp/train_dev.json"),
        sim_pretrain=Path("/data/sim_real_cno.pth"),
        model_root=Path("/repo/model"),
        out_dir=tmp_path / "stage1",
        workers=4,
    )
    assert cmd[cmd.index("--updates") + 1] == str(CNO_UPDATES)
    assert cmd[cmd.index("--eval-interval") + 1] == str(EVAL_INTERVAL)
    assert cmd[cmd.index("--batch-size") + 1] == str(BATCH_SIZE)
    assert cmd[cmd.index("--seed") + 1] == str(SEED)


def test_stage2_command_is_clean_stride1_point_only(tmp_path: Path) -> None:
    cmd = build_stage2_command(
        python=sys.executable,
        real_root=Path("/data/real"),
        train_manifest=Path("/tmp/train_dev.json"),
        cno_checkpoint=Path("/tmp/stage1/model_best.pth"),
        model_root=Path("/repo/model"),
        out_dir=tmp_path / "stage2",
        workers=4,
    )
    assert "--train-on-all" not in cmd
    assert "--allow-train-dev-overlap" not in cmd
    assert "--resume-checkpoint" not in cmd
    assert cmd[cmd.index("--updates") + 1] == str(RESIDUAL_UPDATES)
    assert cmd[cmd.index("--stride") + 1] == "1"
    assert cmd[cmd.index("--eval-stride") + 1] == "20"
    assert cmd[cmd.index("--train-window-mode") + 1] == "fixed"
    assert cmd[cmd.index("--eval-alphas") + 1] == "1.0"
    assert cmd[cmd.index("--selection-metric") + 1] == "point_score"
    assert cmd[cmd.index("--hidden") + 1] == "96"
    assert cmd[cmd.index("--blocks") + 1] == "2"
    assert cmd[cmd.index("--max-delta") + 1] == "0.04"
    assert cmd[cmd.index("--seed") + 1] == str(SEED)


def test_point_only_checkpoint_selection_ignores_sps_and_time() -> None:
    from residual_multi import checkpoint_selection_score

    summary = {
        "rel_l2_score": 90.0,
        "tke_score": 75.0,
        "mvpe_score": 93.0,
        "best_final_est": 10.0,
    }
    assert checkpoint_selection_score(summary, "point_score") == (90.0 + 75.0 + 93.0) / 3.0
    assert checkpoint_selection_score(summary, "final_est") == 10.0


def test_clean_commands_enable_ram_preload() -> None:
    stage1 = build_stage1_command(
        python=sys.executable,
        real_root=Path("/data/real"),
        train_manifest=Path("/tmp/train_dev.json"),
        sim_pretrain=Path("/data/sim_real_cno.pth"),
        model_root=Path("/repo/model"),
        out_dir=Path("/tmp/stage1"),
        workers=4,
    )
    stage2 = build_stage2_command(
        python=sys.executable,
        real_root=Path("/data/real"),
        train_manifest=Path("/tmp/train_dev.json"),
        cno_checkpoint=Path("/tmp/stage1/model_best.pth"),
        model_root=Path("/repo/model"),
        out_dir=Path("/tmp/stage2"),
        workers=4,
    )
    for command in (stage1, stage2):
        assert "--preload-to-ram" in command
        assert command[command.index("--prefetch-factor") + 1] == "4"
