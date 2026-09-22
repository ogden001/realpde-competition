#!/usr/bin/env python3
"""Whitelist lightweight evidence from the long AoA mean-field screen into Git."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT_FILES = [
    "campaign_manifest.json",
    "commands.jsonl",
    "diagnostic_manifest.json",
    "A1_aoa_meanfield.train.review.log",
    "A1_aoa_meanfield.train.review.log.meta.json",
]
CANDIDATE_FILES = [
    "run_config.json",
    "aoa_augmentation_audit.json",
    "final_primary_metrics.json",
    "summary.json",
    "training_progress.csv",
    "comparison_at_5k.csv",
    "window_audit.jsonl",
    "diagnostics/summary.json",
    "diagnostics/by_horizon.csv",
    "diagnostics/by_trajectory.csv",
    "diagnostics/by_trajectory_horizon.csv",
    "diagnostics/spatial_maps.npz",
]
COMPARISON_ROOT_FILES = [
    "comparison_summary.csv",
    "comparison_by_horizon.csv",
    "manifest.json",
]
COMPARISON_MODEL_FILES = [
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
        raise FileExistsError(f"refusing to overwrite existing destination: {args.dest}")
    args.dest.mkdir(parents=True)

    copy_required(args.run_root, args.dest, ROOT_FILES)

    candidate = args.run_root / "A1_aoa_meanfield"
    dest_candidate = args.dest / "A1_aoa_meanfield"
    copy_required(candidate, dest_candidate, CANDIDATE_FILES)
    for step in range(2500, 20001, 2500):
        copy_required(candidate, dest_candidate, [f"eval_step_{step:05d}.json"])

    comparison = args.run_root / "checkpoint_comparison"
    dest_comparison = args.dest / "checkpoint_comparison"
    copy_required(comparison, dest_comparison, COMPARISON_ROOT_FILES)
    for label in ("current80", "aoa_best", "aoa_final"):
        copy_required(
            comparison / label,
            dest_comparison / label,
            COMPARISON_MODEL_FILES,
        )

    archive_manifest = {
        "source_run_root": str(args.run_root),
        "destination": str(args.dest),
        "candidate_eval_steps": list(range(2500, 20001, 2500)),
        "comparison_models": ["current80", "aoa_best", "aoa_final"],
        "checkpoints_copied": False,
        "h5_copied": False,
        "raw_prediction_cache_copied": False,
        "raw_training_log_copied": False,
        "locked_final_copied": False,
    }
    (args.dest / "ARCHIVE_MANIFEST.json").write_text(
        json.dumps(archive_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ARCHIVED", **archive_manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
