#!/usr/bin/env python3
"""Whitelist evidence from the dense residual continuation screen into Git."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT_FILES = [
    "campaign_manifest.json",
    "diagnostic_manifest.json",
    "decision_input.json",
    "commands.jsonl",
    "colleague_dev16_manifest.json",
    "data_manifest.tsv",
    "dense_stride1_5k.train.review.log",
    "dense_stride1_5k.train.review.log.meta.json",
]
CANDIDATE_FILES = [
    "run_config.json",
    "summary.json",
    "final_primary_metrics.json",
    "training_progress.csv",
    "comparison_at_5k.csv",
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

    candidate_src = args.run_root / "dense_stride1_5k"
    candidate_dst = args.dest / "dense_stride1_5k"
    copy_required(candidate_src, candidate_dst, CANDIDATE_FILES)
    for step in range(1000, 5001, 1000):
        copy_required(candidate_src, candidate_dst, [f"eval_step_{step:05d}.json"])

    comparison_src = args.run_root / "checkpoint_comparison"
    comparison_dst = args.dest / "checkpoint_comparison"
    copy_required(comparison_src, comparison_dst, COMPARISON_FILES)
    for label in ("current80", "dense_best", "dense_final"):
        copy_required(
            comparison_src / label,
            comparison_dst / label,
            MODEL_DIAGNOSTICS,
        )

    archive_manifest = {
        "source_run_root": str(args.run_root),
        "destination": str(args.dest),
        "scientific_variable": "residual continuation stride 20 -> 1",
        "historical_sparse_control_reused": True,
        "historical_sparse_control_path": "docs/colleague_screening/results/20260921",
        "checkpoints_copied": False,
        "h5_copied": False,
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
