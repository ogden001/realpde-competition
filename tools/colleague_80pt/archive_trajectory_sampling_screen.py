#!/usr/bin/env python3
"""Whitelist the matched trajectory-sampling screen evidence into Git."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT_FILES = [
    "campaign_manifest.json",
    "diagnostic_manifest.json",
    "decision_input.json",
    "initialization_parity.json",
    "sampling_comparison.json",
    "matched_progress.csv",
    "commands.jsonl",
    "colleague_dev16_manifest.json",
    "data_manifest.tsv",
    "A_fixed_stride20.train.review.log",
    "A_fixed_stride20.train.review.log.meta.json",
    "B_stratified_random_start.train.review.log",
    "B_stratified_random_start.train.review.log.meta.json",
]
ARM_FILES = [
    "run_config.json",
    "summary.json",
    "final_primary_metrics.json",
    "sampling_audit.json",
    "window_audit.jsonl",
    "by_trajectory.csv",
    "by_horizon.csv",
    "by_trajectory_horizon.csv",
    "diagnostics/summary.json",
    "diagnostics/by_horizon.csv",
    "diagnostics/by_trajectory.csv",
    "diagnostics/by_trajectory_horizon.csv",
    "diagnostics/spatial_maps.npz",
]
COMPARISON_FILES = [
    "comparison_summary.csv",
    "comparison_by_horizon.csv",
    "manifest.json",
]
MODEL_DIAGNOSTICS = [
    "summary.json",
    "by_horizon.csv",
    "by_trajectory.csv",
    "by_trajectory_horizon.csv",
    "spatial_maps.npz",
]


def copy_required(src_root: Path, dst_root: Path, relative_paths: list[str]) -> None:
    for relative in relative_paths:
        src = src_root / relative
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = dst_root / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()

    if not (args.run_root / "DONE").is_file():
        raise RuntimeError("campaign is not complete: DONE marker missing")
    if args.dest.exists():
        raise FileExistsError(f"refusing to overwrite destination: {args.dest}")
    args.dest.mkdir(parents=True)

    copy_required(args.run_root, args.dest, ROOT_FILES)

    for arm in ("A_fixed_stride20", "B_stratified_random_start"):
        src = args.run_root / arm
        dst = args.dest / arm
        copy_required(src, dst, ARM_FILES)
        for step in range(0, 5001, 1000):
            copy_required(src, dst, [f"eval_step_{step:05d}.json"])

    comparison_src = args.run_root / "checkpoint_comparison"
    comparison_dst = args.dest / "checkpoint_comparison"
    copy_required(comparison_src, comparison_dst, COMPARISON_FILES)
    for label in (
        "init",
        "control_best",
        "control_final",
        "candidate_best",
        "candidate_final",
    ):
        copy_required(
            comparison_src / label,
            comparison_dst / label,
            MODEL_DIAGNOSTICS,
        )

    manifest = {
        "source_run_root": str(args.run_root),
        "destination": str(args.dest),
        "scientific_variable": "training sampling policy only",
        "arms": [
            "A_fixed_stride20",
            "B_stratified_random_start",
        ],
        "checkpoints_copied": False,
        "h5_copied": False,
        "raw_training_logs_copied": False,
        "prediction_caches_copied": False,
        "locked_final_copied": False,
    }
    (args.dest / "ARCHIVE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ARCHIVED", **manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
