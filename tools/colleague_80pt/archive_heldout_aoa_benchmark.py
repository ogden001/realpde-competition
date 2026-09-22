#!/usr/bin/env python3
"""Whitelist evidence from the heldout-AoA benchmark into Git."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT_FILES = [
    "campaign_manifest.json",
    "diagnostic_manifest.json",
    "commands.jsonl",
    "heldout_aoa_split_audit.json",
    "train_innerdev_manifest.json",
    "heldout_eval_manifest.json",
]
ARM_FILES = [
    "cno_stage1.json",
    "cno_stage1.train.review.log",
    "cno_stage1.train.review.log.meta.json",
    "residual/run_config.json",
    "residual/final_primary_metrics.json",
    "residual/summary.json",
    "residual/window_audit.jsonl",
    "residual/by_trajectory.csv",
    "residual/by_horizon.csv",
    "residual/by_trajectory_horizon.csv",
    "residual/diagnostics/summary.json",
    "residual/diagnostics/by_horizon.csv",
    "residual/diagnostics/by_trajectory.csv",
    "residual/diagnostics/by_trajectory_horizon.csv",
    "residual/diagnostics/spatial_maps.npz",
    "residual.train.review.log",
    "residual.train.review.log.meta.json",
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
        raise RuntimeError("DONE marker missing")
    if args.dest.exists():
        raise FileExistsError(args.dest)
    args.dest.mkdir(parents=True)

    copy_required(args.run_root, args.dest, ROOT_FILES)
    for arm in ("control", "aug"):
        arm_src = args.run_root / arm
        arm_dst = args.dest / arm
        copy_required(arm_src, arm_dst, ARM_FILES)
        if arm == "aug":
            copy_required(arm_src, arm_dst, [
                "cno_stage1.aoa_audit.json",
                "residual/aoa_augmentation_audit.json",
            ])
        for step in range(2500, 20001, 2500):
            copy_required(
                arm_src,
                arm_dst,
                [f"residual/eval_step_{step:05d}.json"],
            )

    model_labels = [
        "control_cno",
        "control_best",
        "control_final",
        "aug_cno",
        "aug_best",
        "aug_final",
    ]
    for comparison_name in ("inner_dev_comparison", "heldout10_comparison"):
        src = args.run_root / comparison_name
        dst = args.dest / comparison_name
        copy_required(src, dst, COMPARISON_FILES)
        for label in model_labels:
            copy_required(src / label, dst / label, MODEL_DIAGNOSTICS)

    archive_manifest = {
        "source_run_root": str(args.run_root),
        "destination": str(args.dest),
        "arms": ["control", "aug"],
        "heldout_aoa": 10.0,
        "comparison_sets": ["inner_dev", "heldout10"],
        "checkpoints_copied": False,
        "h5_copied": False,
        "raw_training_logs_copied": False,
        "locked_final_copied": False,
    }
    (args.dest / "ARCHIVE_MANIFEST.json").write_text(
        json.dumps(archive_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ARCHIVED", **archive_manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
