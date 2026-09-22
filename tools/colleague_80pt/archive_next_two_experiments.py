#!/usr/bin/env python3
"""Whitelist-copy lightweight evidence from the two-experiment campaign into Git.

This script deliberately excludes model checkpoints, HDF5 data, raw prediction
caches, and raw long logs.  It copies only reviewable evidence required by the
RealPDE diagnostic protocol.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


A_FILES = [
    "diagnostic_manifest.json",
    "scan.csv",
    "comparison_by_horizon.csv",
    "trajectory_comparison_summary.csv",
    "spatial_summary.csv",
]
A_CANDIDATE_FILES = [
    "final_primary_metrics.json",
    "diagnostics/summary.json",
    "diagnostics/by_horizon.csv",
    "diagnostics/by_trajectory.csv",
    "diagnostics/by_trajectory_horizon.csv",
    "diagnostics/spatial_maps.npz",
]
B_FILES = [
    "run_config.json",
    "final_primary_metrics.json",
    "summary.json",
    "training_progress.csv",
    "comparison_summary.csv",
    "comparison_by_horizon.csv",
    "trajectory_comparison_summary.csv",
    "spatial_summary.csv",
    "diagnostic_manifest.json",
    "window_audit.jsonl",
]
B_DIAGNOSTIC_FILES = [
    "diagnostics/summary.json",
    "diagnostics/by_horizon.csv",
    "diagnostics/by_trajectory.csv",
    "diagnostics/by_trajectory_horizon.csv",
    "diagnostics/spatial_maps.npz",
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
        raise FileExistsError(f"refusing to overwrite existing destination: {args.dest}")
    args.dest.mkdir(parents=True)

    copy_required(
        args.run_root,
        args.dest,
        ["campaign_manifest.json", "commands.jsonl"],
    )

    experiment_a = args.run_root / "A_fluctuation_calibration"
    dest_a = args.dest / "A_fluctuation_calibration"
    copy_required(experiment_a, dest_a, A_FILES)
    candidates = sorted(
        path for path in experiment_a.iterdir()
        if path.is_dir() and path.name.startswith("fluct_ramp_end_")
    )
    if len(candidates) != 4:
        raise RuntimeError(f"expected 4 fluctuation candidates, found {len(candidates)}")
    for candidate in candidates:
        copy_required(candidate, dest_a / candidate.name, A_CANDIDATE_FILES)

    experiment_b = args.run_root / "B_angle_aug_2deg"
    dest_b = args.dest / "B_angle_aug_2deg"
    copy_required(experiment_b, dest_b, B_FILES)
    copy_required(experiment_b, dest_b, B_DIAGNOSTIC_FILES)
    for step in (1000, 2000, 3000, 4000, 5000):
        copy_required(
            experiment_b,
            dest_b,
            [f"eval_step_{step:05d}.json"],
        )

    review_log = args.run_root / "B_angle_aug_2deg.train.review.log"
    review_meta = args.run_root / "B_angle_aug_2deg.train.review.log.meta.json"
    for path in (review_log, review_meta):
        if not path.is_file():
            raise FileNotFoundError(path)
        shutil.copy2(path, args.dest / path.name)

    archive_manifest = {
        "source_run_root": str(args.run_root),
        "destination": str(args.dest),
        "fluctuation_candidates": [path.name for path in candidates],
        "angle_aug_eval_steps": [1000, 2000, 3000, 4000, 5000],
        "checkpoints_copied": False,
        "h5_copied": False,
        "raw_prediction_cache_copied": False,
        "raw_training_log_copied": False,
    }
    (args.dest / "ARCHIVE_MANIFEST.json").write_text(
        json.dumps(archive_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ARCHIVED", **archive_manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
