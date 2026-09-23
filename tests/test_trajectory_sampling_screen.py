from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "tools" / "colleague_80pt"
sys.path.insert(0, str(SCRIPT_DIR))

from realpde_h5_feature_adapter_train import (  # noqa: E402
    TrajectoryStratifiedRandomStartBatchSampler,
    WindowRef,
)
from residual_multi import summarize_sampling_exposure  # noqa: E402
from run_trajectory_sampling_screen import (  # noqa: E402
    BATCH_SIZE,
    EVAL_INTERVAL,
    LR,
    SEED,
    UPDATES,
    WEIGHT_DECAY,
    build_arm_command,
)


def fake_dataset(n_trajectories: int = 10, starts_per_trajectory: int = 20):
    paths = [Path(f"traj_{idx}.h5") for idx in range(n_trajectories)]
    refs = [
        WindowRef(path=path, start=start)
        for path in paths
        for start in range(starts_per_trajectory)
    ]
    return SimpleNamespace(
        window_mode="random_phase",
        paths=paths,
        refs=refs,
    )


def flatten(plan: list[list[int]]) -> list[int]:
    return [index for batch in plan for index in batch]


def test_stratified_sampler_matches_trajectory_quotas_and_unique_batches() -> None:
    dataset = fake_dataset()
    sampler = TrajectoryStratifiedRandomStartBatchSampler(
        dataset,
        batch_size=4,
        seed=41,
    )
    quotas = {path.name: 4 for path in dataset.paths}
    sampler.set_epoch(0, trajectory_quotas=quotas)

    counts = {path.name: 0 for path in dataset.paths}
    starts = {path.name: [] for path in dataset.paths}
    for batch in sampler.current_plan:
        names = [dataset.refs[index].path.name for index in batch]
        assert len(names) == len(set(names)) == 4
        for index in batch:
            ref = dataset.refs[index]
            counts[ref.path.name] += 1
            starts[ref.path.name].append(ref.start)

    assert counts == quotas
    assert all(len(values) == len(set(values)) == 4 for values in starts.values())


def test_stratified_sampler_shuffled_bag_avoids_repeats_across_epochs() -> None:
    dataset = fake_dataset(starts_per_trajectory=12)
    sampler = TrajectoryStratifiedRandomStartBatchSampler(
        dataset,
        batch_size=4,
        seed=41,
    )
    quotas = {path.name: 2 for path in dataset.paths}
    sampler.set_epoch(0, trajectory_quotas=quotas)
    first = sampler.current_plan
    sampler.set_epoch(1, trajectory_quotas=quotas)
    second = sampler.current_plan

    starts = {path.name: [] for path in dataset.paths}
    for index in flatten(first) + flatten(second):
        ref = dataset.refs[index]
        starts[ref.path.name].append(ref.start)
    assert all(len(values) == len(set(values)) == 4 for values in starts.values())


def test_stratified_sampler_is_deterministic() -> None:
    dataset = fake_dataset()
    quotas = {path.name: 4 for path in dataset.paths}
    left = TrajectoryStratifiedRandomStartBatchSampler(dataset, batch_size=4, seed=41)
    right = TrajectoryStratifiedRandomStartBatchSampler(dataset, batch_size=4, seed=41)
    left.set_epoch(0, trajectory_quotas=quotas)
    right.set_epoch(0, trajectory_quotas=quotas)
    assert left.current_plan == right.current_plan


def test_sampling_audit_reports_batch_trajectory_duplicates() -> None:
    dataset = fake_dataset(n_trajectories=8, starts_per_trajectory=10)
    sampler = TrajectoryStratifiedRandomStartBatchSampler(
        dataset,
        batch_size=4,
        seed=41,
    )
    quotas = {path.name: 2 for path in dataset.paths}
    sampler.set_epoch(0, trajectory_quotas=quotas)
    audit = summarize_sampling_exposure(
        dataset,
        [flatten(sampler.current_plan)],
        updates=4,
        batch_size=4,
        sampling_policy="test",
    )
    assert audit["samples_consumed"] == 16
    assert audit["duplicate_trajectory_batches"] == 0
    assert audit["unique_windows_consumed"] == 16


def test_matched_runner_freezes_all_non_sampling_semantics(tmp_path: Path) -> None:
    common = dict(
        python=sys.executable,
        real_root=Path("/data/real"),
        split_manifest=Path("/data/split.json"),
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
        out_idx = result.index("--out-dir") + 1
        mode_idx = result.index("--train-window-mode") + 1
        result[out_idx] = "<OUT>"
        result[mode_idx] = "<SAMPLER>"
        return result

    assert normalized(control) == normalized(candidate)
    assert "--resume-checkpoint" not in control
    assert "--resume-checkpoint" not in candidate
    assert control[control.index("--stride") + 1] == "20"
    assert candidate[candidate.index("--stride") + 1] == "20"
    assert control[control.index("--eval-stride") + 1] == "20"
    assert candidate[candidate.index("--eval-stride") + 1] == "20"
    assert "--disable-phase-count-equalization" in control
    assert "--disable-phase-count-equalization" in candidate
    assert UPDATES == 5004
    assert EVAL_INTERVAL == 1000
    assert BATCH_SIZE == 8
    assert LR == 2e-4
    assert WEIGHT_DECAY == 1e-5
    assert SEED == 41
