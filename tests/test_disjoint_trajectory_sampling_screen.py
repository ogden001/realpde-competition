from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "tools" / "colleague_80pt"
sys.path.insert(0, str(SCRIPT_DIR))

from run_disjoint_trajectory_sampling_screen import (  # noqa: E402
    BATCH_SIZE,
    EVAL_INTERVAL,
    EXPECTED_BATCHES_PER_EPOCH,
    EXPECTED_DEV_TRAJECTORIES,
    EXPECTED_FIXED_WINDOWS,
    EXPECTED_FULL_EPOCHS,
    EXPECTED_SAMPLES_PER_EPOCH,
    EXPECTED_TRAIN_TRAJECTORIES,
    LR,
    SEED,
    UPDATES,
    WEIGHT_DECAY,
    build_arm_command,
    derive_residual_disjoint_manifest,
)


def make_source_manifest(path: Path) -> tuple[list[str], list[str]]:
    all81 = [f"traj_{idx:02d}.h5" for idx in range(81)]
    dev16 = [all81[idx] for idx in range(0, 32, 2)]
    payload = {
        "protocol": "colleague_dev16_seed41_all81",
        "seed": 41,
        "train": all81,
        "dev": dev16,
        "train_dev_overlap": True,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return all81, dev16


def test_derive_residual_disjoint_manifest_is_exact_subtraction(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    destination = tmp_path / "derived.json"
    all81, dev16 = make_source_manifest(source)

    derived = derive_residual_disjoint_manifest(source, destination)

    assert len(derived["train"]) == 65
    assert len(derived["dev"]) == 16
    assert set(derived["dev"]) == set(dev16)
    assert set(derived["train"]) == set(all81) - set(dev16)
    assert set(derived["train"]).isdisjoint(set(derived["dev"]))
    assert derived["train_dev_overlap"] is False
    assert derived["residual_stage_disjoint"] is True
    assert derived["stage1_cno_trained_on_all81"] is True
    assert derived["end_to_end_clean_holdout"] is False


def test_disjoint_commands_differ_only_by_sampling_and_output(tmp_path: Path) -> None:
    common = dict(
        python=sys.executable,
        real_root=Path("/data/real"),
        split_manifest=Path("/data/train65_dev16.json"),
        base_checkpoint=Path("/data/cno.pt"),
        model_root=Path("/repo/model"),
        workers=4,
    )
    control = build_arm_command(
        **common,
        out_dir=tmp_path / "A",
        sampling_mode="fixed",
    )
    candidate = build_arm_command(
        **common,
        out_dir=tmp_path / "B",
        sampling_mode="trajectory_stratified_random_start",
    )

    def normalized(command: list[str]) -> list[str]:
        result = list(command)
        result[result.index("--out-dir") + 1] = "<OUT>"
        result[result.index("--train-window-mode") + 1] = "<SAMPLER>"
        return result

    assert normalized(control) == normalized(candidate)

    for command in (control, candidate):
        assert "--train-on-all" not in command
        assert "--allow-train-dev-overlap" not in command
        assert "--resume-checkpoint" not in command
        assert "--disable-phase-count-equalization" in command
        assert command[command.index("--stride") + 1] == "20"
        assert command[command.index("--eval-stride") + 1] == "20"
        assert command[command.index("--batch-size") + 1] == "8"
        assert command[command.index("--lr") + 1] == "0.0002"
        assert command[command.index("--weight-decay") + 1] == "1e-05"
        assert command[command.index("--seed") + 1] == "41"


def test_disjoint_budget_is_complete_matched_epochs() -> None:
    assert EXPECTED_TRAIN_TRAJECTORIES == 65
    assert EXPECTED_DEV_TRAJECTORIES == 16
    assert EXPECTED_FIXED_WINDOWS == 2701
    assert EXPECTED_SAMPLES_PER_EPOCH == 2696
    assert EXPECTED_BATCHES_PER_EPOCH == 337
    assert EXPECTED_FULL_EPOCHS == 15
    assert UPDATES == EXPECTED_BATCHES_PER_EPOCH * EXPECTED_FULL_EPOCHS == 5055
    assert EXPECTED_SAMPLES_PER_EPOCH == EXPECTED_BATCHES_PER_EPOCH * BATCH_SIZE
    assert EVAL_INTERVAL == 1000
    assert LR == 2e-4
    assert WEIGHT_DECAY == 1e-5
    assert SEED == 41
