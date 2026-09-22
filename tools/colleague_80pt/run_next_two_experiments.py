#!/usr/bin/env python3
"""Run the two frozen next-step screens from the colleague 80-point baseline.

A. Mean-preserving fluctuation-amplitude calibration, no training.
B. Small-angle velocity-direction augmentation, 5000-step residual continuation.

This campaign never accesses locked-final data, never submits to Codabench, and
never starts a full-train follow-up automatically.  It only produces evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from run_incremental_screen import (
    EXPECTED_BASE_SHA256,
    EXPECTED_START_SHA256,
    FIXED_TIME_SECONDS,
    require_clean_main_checkout,
    require_gpu,
    sha256,
    verify_allowed_data,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
UPDATES = 5000
EVAL_INTERVAL = 1000
BASELINE_METRICS = {
    "rel_l2_raw": 0.0804204195737838,
    "tke_raw": 0.4491582512855530,
    "mvpe_raw": 0.0711169168353080,
}
HISTORICAL_NO_AUG_5K = {
    "rel_l2_raw": 0.0802410691976547,
    "tke_raw": 0.4503658413887024,
    "mvpe_raw": 0.0710276663303375,
}
FROZEN_BASELINE_DIAGNOSTICS = (
    REPO_ROOT
    / "docs"
    / "colleague_screening"
    / "results"
    / "20260921"
    / "diagnostics"
    / "current_80pt_residual"
)


def run(command: list[str], log_path: Path, command_log: Path) -> None:
    record = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "log": str(log_path),
    }
    with command_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    print("RUN " + " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {command}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float, baseline: float) -> float:
    return 100.0 * (float(value) - float(baseline)) / max(abs(float(baseline)), 1e-12)


def alpha_one_metrics(step_path: Path) -> dict[str, float]:
    rows = json.loads(step_path.read_text(encoding="utf-8"))
    for row in rows:
        if abs(float(row["alpha"]) - 1.0) < 1e-12:
            return {
                "rel_l2_raw": float(row["rel_l2_raw"]),
                "tke_raw": float(row["tke_raw"]),
                "mvpe_raw": float(row["mvpe_raw"]),
                "final_est": float(row["best_final_est"]),
                "best_alpha": 1.0,
            }
    raise RuntimeError(f"alpha=1.0 missing from {step_path}")


def build_training_progress(run_dir: Path) -> None:
    rows: list[dict[str, object]] = [{
        "step": 0,
        **BASELINE_METRICS,
        "final_est": "",
        "best_alpha": "",
    }]
    for step in range(EVAL_INTERVAL, UPDATES + 1, EVAL_INTERVAL):
        rows.append({"step": step, **alpha_one_metrics(run_dir / f"eval_step_{step:05d}.json")})
    write_csv(run_dir / "training_progress.csv", rows)


def summarize_spatial_npz(npz_path: Path, output: Path) -> None:
    archive = np.load(npz_path)
    rows: list[dict[str, object]] = []
    for metric in sorted(archive.files):
        value = np.asarray(archive[metric], dtype=np.float64)
        flat_index = int(np.argmax(value))
        max_row, max_col = np.unravel_index(flat_index, value.shape)
        rows.append({
            "metric": metric,
            "mean": float(value.mean()),
            "p95": float(np.percentile(value, 95.0)),
            "max": float(value.max()),
            "max_row": int(max_row),
            "max_col": int(max_col),
        })
    write_csv(output, rows)


def compare_horizon(candidate_path: Path, output: Path) -> None:
    baseline = {
        int(row["horizon"]): row
        for row in read_csv(FROZEN_BASELINE_DIAGNOSTICS / "by_horizon.csv")
    }
    rows: list[dict[str, object]] = []
    for row in read_csv(candidate_path):
        horizon = int(row["horizon"])
        base = baseline[horizon]
        rows.append({
            **row,
            "delta_rel_pct_vs_current80": pct(
                float(row["frame_rel_l2"]), float(base["frame_rel_l2"])
            ),
            "delta_tke_contrib_pct_vs_current80": pct(
                float(row["tke_contrib_rel_l2"]),
                float(base["tke_contrib_rel_l2"]),
            ),
        })
    write_csv(output, rows)


def compare_trajectory(candidate_path: Path, output: Path) -> None:
    baseline = {
        row["trajectory"]: row
        for row in read_csv(FROZEN_BASELINE_DIAGNOSTICS / "by_trajectory.csv")
    }
    rows: list[dict[str, object]] = []
    for row in read_csv(candidate_path):
        base = baseline[row["trajectory"]]
        rows.append({
            **row,
            "delta_rel_pct_vs_current80": pct(
                float(row["rel_l2"]), float(base["rel_l2"])
            ),
            "delta_tke_pct_vs_current80": pct(
                float(row["tke_rel_l2"]), float(base["tke_rel_l2"])
            ),
            "delta_fluctuation_pct_vs_current80": pct(
                float(row["fluctuation_rel_l2"]),
                float(base["fluctuation_rel_l2"]),
            ),
        })
    write_csv(output, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--start-checkpoint", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()

    for path in (
        args.real_root,
        args.base_checkpoint,
        args.start_checkpoint,
        args.model_root,
        args.data_manifest,
        args.split_manifest,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_root}")
    if sha256(args.base_checkpoint) != EXPECTED_BASE_SHA256:
        raise RuntimeError("Stage-1 CNO checkpoint SHA-256 mismatch")
    if sha256(args.start_checkpoint) != EXPECTED_START_SHA256:
        raise RuntimeError("frozen 80-point residual checkpoint SHA-256 mismatch")
    if not FROZEN_BASELINE_DIAGNOSTICS.is_dir():
        raise FileNotFoundError(FROZEN_BASELINE_DIAGNOSTICS)

    execution_commit = require_clean_main_checkout(REPO_ROOT)
    gpu = require_gpu()
    verified_data = verify_allowed_data(
        args.real_root,
        args.data_manifest,
        args.split_manifest,
    )

    args.out_root.mkdir(parents=True)
    command_log = args.out_root / "commands.jsonl"
    (args.out_root / "RUNNING").touch()
    manifest = {
        "campaign": "colleague80_next_two_experiments_20260922",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_commit": execution_commit,
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "start_checkpoint": str(args.start_checkpoint),
        "start_checkpoint_sha256": sha256(args.start_checkpoint),
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": sha256(args.split_manifest),
        "data_manifest": str(args.data_manifest),
        "data_manifest_sha256": sha256(args.data_manifest),
        "verified_data": verified_data,
        "gpu": gpu,
        "experiments": {
            "A": {
                "name": "mean_preserving_fluctuation_amplitude_calibration",
                "training": False,
                "end_scales": [1.0, 1.2, 1.4, 1.6],
            },
            "B": {
                "name": "small_angle_velocity_direction_augmentation",
                "training": True,
                "updates": UPDATES,
                "angle_aug_max_deg": 2.0,
                "all_other_residual_hyperparameters": "matched to historical R0 control",
            },
        },
        "locked_final_accessed": False,
        "codabench_accessed": False,
        "automatic_full_train_started": False,
        "environment": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES",)
            if os.environ.get(key)
        },
    }
    (args.out_root / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    try:
        experiment_a = args.out_root / "A_fluctuation_calibration"
        run([
            sys.executable,
            "-u",
            "-B",
            str(SCRIPT_DIR / "fluctuation_calibration.py"),
            "--real-root",
            str(args.real_root),
            "--split-manifest",
            str(args.split_manifest),
            "--checkpoint",
            str(args.start_checkpoint),
            "--realpdebench-root",
            str(args.model_root),
            "--out-dir",
            str(experiment_a),
            "--end-scales",
            "1.0,1.2,1.4,1.6",
            "--batch-size",
            "32",
            "--workers",
            "2",
        ], args.out_root / "A_fluctuation_calibration.log", command_log)

        experiment_b = args.out_root / "B_angle_aug_2deg"
        run([
            sys.executable,
            "-u",
            "-B",
            str(SCRIPT_DIR / "residual_multi.py"),
            "--real-root",
            str(args.real_root),
            "--split-manifest",
            str(args.split_manifest),
            "--allow-train-dev-overlap",
            "--checkpoint",
            str(args.base_checkpoint),
            "--resume-checkpoint",
            str(args.start_checkpoint),
            "--realpdebench-root",
            str(args.model_root),
            "--base-model",
            "cno",
            "--out-dir",
            str(experiment_b),
            "--updates",
            str(UPDATES),
            "--eval-interval",
            str(EVAL_INTERVAL),
            "--batch-size",
            "8",
            "--test-batch-size",
            "32",
            "--workers",
            "2",
            "--lr",
            "0.0002",
            "--weight-decay",
            "0.00001",
            "--hidden",
            "96",
            "--blocks",
            "2",
            "--max-delta",
            "0.04",
            "--stride",
            "20",
            "--train-alpha",
            "1.0",
            "--tke",
            "0.06",
            "--train-window-mode",
            "fixed",
            "--angle-aug-max-deg",
            "2.0",
            "--bound-abs",
            "0.0075",
            "--bound-rel",
            "0.0075",
            "--fixed-time-seconds",
            str(FIXED_TIME_SECONDS),
            "--seed",
            "41",
        ], args.out_root / "B_angle_aug_2deg.log", command_log)

        run([
            sys.executable,
            "-u",
            "-B",
            str(REPO_ROOT / "tools" / "build_training_review_log.py"),
            "--input",
            str(args.out_root / "B_angle_aug_2deg.log"),
            "--output",
            str(args.out_root / "B_angle_aug_2deg.train.review.log"),
        ], args.out_root / "B_review_log_builder.log", command_log)

        build_training_progress(experiment_b)
        diagnostics_b = experiment_b / "diagnostics"
        summarize_spatial_npz(
            diagnostics_b / "spatial_maps.npz",
            experiment_b / "spatial_summary.csv",
        )
        compare_horizon(
            diagnostics_b / "by_horizon.csv",
            experiment_b / "comparison_by_horizon.csv",
        )
        compare_trajectory(
            diagnostics_b / "by_trajectory.csv",
            experiment_b / "trajectory_comparison_summary.csv",
        )

        b_metrics = json.loads(
            (experiment_b / "final_primary_metrics.json").read_text(encoding="utf-8")
        )
        comparison_rows = [
            {
                "model": "current_80pt_residual",
                "role": "frozen_decision_baseline",
                **BASELINE_METRICS,
            },
            {
                "model": "historical_no_aug_5000",
                "role": "matched_optimization_control_from_20260921",
                **HISTORICAL_NO_AUG_5K,
            },
            {
                "model": "angle_aug_2deg_5000",
                "role": "candidate",
                "rel_l2_raw": float(b_metrics["rel_l2_raw"]),
                "tke_raw": float(b_metrics["tke_raw"]),
                "mvpe_raw": float(b_metrics["mvpe_raw"]),
            },
        ]
        for row in comparison_rows:
            row["delta_rel_pct_vs_current80"] = pct(
                float(row["rel_l2_raw"]), BASELINE_METRICS["rel_l2_raw"]
            )
            row["delta_tke_pct_vs_current80"] = pct(
                float(row["tke_raw"]), BASELINE_METRICS["tke_raw"]
            )
            row["delta_mvpe_pct_vs_current80"] = pct(
                float(row["mvpe_raw"]), BASELINE_METRICS["mvpe_raw"]
            )
        write_csv(experiment_b / "comparison_summary.csv", comparison_rows)

        b_manifest = {
            "experiment": "small_angle_velocity_direction_augmentation",
            "execution_commit": execution_commit,
            "start_checkpoint": str(args.start_checkpoint),
            "start_checkpoint_sha256": sha256(args.start_checkpoint),
            "base_checkpoint": str(args.base_checkpoint),
            "base_checkpoint_sha256": sha256(args.base_checkpoint),
            "split_manifest": str(args.split_manifest),
            "split_manifest_sha256": sha256(args.split_manifest),
            "updates": UPDATES,
            "angle_aug_max_deg": 2.0,
            "augmentation_semantics": (
                "one per-window angle sampled uniformly in [-2,+2] degrees; "
                "same u/v vector rotation applied to Past20 and Future20; "
                "spatial grid unchanged; training only; approximate flow-direction "
                "augmentation, not an exact CFD AoA transform"
            ),
            "dev_augmentation": False,
            "locked_final_accessed": False,
            "codabench_accessed": False,
            "automatic_full_train_started": False,
        }
        (experiment_b / "diagnostic_manifest.json").write_text(
            json.dumps(b_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        (args.out_root / "DONE").touch()
        print(json.dumps({
            "status": "REVIEW_REQUIRED",
            "experiment_a": str(experiment_a),
            "experiment_b": str(experiment_b),
            "new_training_started": True,
            "full_training_started": False,
            "codabench_accessed": False,
        }, indent=2), flush=True)
    except BaseException as error:
        (args.out_root / "FAILED.json").write_text(
            json.dumps({"type": type(error).__name__, "message": str(error)}, indent=2) + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        (args.out_root / "RUNNING").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
