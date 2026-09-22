#!/usr/bin/env python3
"""Run the long adjacent-AoA mean-field augmentation screen.

One 20k-update candidate is trained from the frozen colleague 80-point residual.
At 5k it is compared against the already archived matched no-augmentation R0
control; training then continues to 20k to avoid rejecting a data-augmentation
method merely because it converges more slowly.

No full-data refit, packaging, locked-final access, or Codabench access occurs.
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

from run_incremental_screen import (
    EXPECTED_BASE_SHA256,
    EXPECTED_START_SHA256,
    FIXED_TIME_SECONDS,
    require_clean_main_checkout,
    require_gpu,
    sha256,
    verify_allowed_data,
)
from run_next_two_experiments import alpha_one_metrics


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_UPDATES = 20_000
DEFAULT_EVAL_INTERVAL = 2_500
DEFAULT_MATCHED_CONTROL_STEP = 5_000
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 2e-4
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


def pct(value: float, baseline: float) -> float:
    return 100.0 * (float(value) - float(baseline)) / max(abs(float(baseline)), 1e-12)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_training_progress(
    run_dir: Path,
    *,
    updates: int,
    eval_interval: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [{
        "step": 0,
        **BASELINE_METRICS,
        "final_est": "",
        "best_alpha": "",
    }]
    for step in range(eval_interval, updates + 1, eval_interval):
        rows.append({"step": step, **alpha_one_metrics(run_dir / f"eval_step_{step:05d}.json")})
    write_csv(run_dir / "training_progress.csv", rows)
    return rows


def build_matched_budget_comparison(
    run_dir: Path,
    *,
    matched_control_step: int,
    batch_size: int,
) -> None:
    candidate = alpha_one_metrics(
        run_dir / f"eval_step_{matched_control_step:05d}.json"
    )
    rows = [
        {
            "model": "current_80pt_residual",
            "role": "frozen_decision_baseline",
            **BASELINE_METRICS,
        },
        {
            "model": "historical_no_aug_5000",
            "role": "matched_same_budget_control_from_20260921",
            **HISTORICAL_NO_AUG_5K,
        },
        {
            "model": f"aoa_meanfield_aug_step_{matched_control_step}",
            "role": (
                "sample_exposure_matched_to_historical_b8_step5000"
                if batch_size != 8
                else "matched_same_budget"
            ),
            "rel_l2_raw": candidate["rel_l2_raw"],
            "tke_raw": candidate["tke_raw"],
            "mvpe_raw": candidate["mvpe_raw"],
        },
    ]
    for row in rows:
        for key in ("rel_l2_raw", "tke_raw", "mvpe_raw"):
            row[f"delta_{key}_pct_vs_current80"] = pct(
                float(row[key]), BASELINE_METRICS[key]
            )
    candidate_row = rows[-1]
    for key in ("rel_l2_raw", "tke_raw", "mvpe_raw"):
        candidate_row[f"delta_{key}_pct_vs_no_aug_5k"] = pct(
            float(candidate_row[key]), HISTORICAL_NO_AUG_5K[key]
        )
    write_csv(run_dir / "comparison_at_matched_budget.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--start-checkpoint", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument(
        "--matched-control-step",
        type=int,
        default=DEFAULT_MATCHED_CONTROL_STEP,
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader worker count; environment-only tuning, does not change experiment semantics.",
    )
    args = parser.parse_args()

    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_root}")
    if args.updates < 1 or args.eval_interval < 1:
        raise ValueError("updates/eval_interval must be positive")
    if args.updates % args.eval_interval:
        raise ValueError("updates must be divisible by eval_interval")
    if args.matched_control_step < 1 or args.matched_control_step > args.updates:
        raise ValueError("matched_control_step must be inside training budget")
    if args.matched_control_step % args.eval_interval:
        raise ValueError("matched_control_step must be an evaluation milestone")
    if args.batch_size not in (8, 16):
        raise ValueError("only frozen batch sizes 8/16 are allowed")
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
    if sha256(args.base_checkpoint) != EXPECTED_BASE_SHA256:
        raise RuntimeError("Stage-1 CNO checkpoint SHA-256 mismatch")
    if sha256(args.start_checkpoint) != EXPECTED_START_SHA256:
        raise RuntimeError("frozen 80-point residual checkpoint SHA-256 mismatch")

    execution_commit = require_clean_main_checkout(REPO_ROOT)
    gpu = require_gpu()
    verified_data = verify_allowed_data(
        args.real_root,
        args.data_manifest,
        args.split_manifest,
    )

    args.out_root.mkdir(parents=True)
    command_log = args.out_root / "commands.jsonl"
    candidate_dir = args.out_root / "A1_aoa_meanfield"
    manifest = {
        "campaign": "colleague80_aoa_meanfield_long_screen_20260922",
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
        "experiment": {
            "name": "adjacent_aoa_mean_field_interpolation",
            "updates": args.updates,
            "eval_interval": args.eval_interval,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "augmentation_probability": 0.5,
            "lambda_range": [0.2, 0.5],
            "same_re_required": True,
            "max_neighbor_gap_deg": 5.1,
            "min_eligible_fraction": 0.8,
            "input_shift_source": "Past20 mean spatial velocity field only",
            "future_used_to_construct_input_shift": False,
            "aoa_re_used_at_inference": False,
        },
        "historical_matched_control": {
            "step": args.matched_control_step,
            "historical_reference_batch_size": 8,
            "candidate_batch_size": args.batch_size,
            **HISTORICAL_NO_AUG_5K,
        },
        "automatic_full_train_started": False,
        "codabench_accessed": False,
        "locked_final_accessed": False,
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
    (args.out_root / "RUNNING").touch()

    try:
        run([
            sys.executable,
            "-u",
            "-B",
            str(SCRIPT_DIR / "residual_multi.py"),
            "--real-root", str(args.real_root),
            "--split-manifest", str(args.split_manifest),
            "--allow-train-dev-overlap",
            "--checkpoint", str(args.base_checkpoint),
            "--resume-checkpoint", str(args.start_checkpoint),
            "--realpdebench-root", str(args.model_root),
            "--base-model", "cno",
            "--out-dir", str(candidate_dir),
            "--updates", str(args.updates),
            "--eval-interval", str(args.eval_interval),
            "--batch-size", str(args.batch_size),
            "--test-batch-size", "32",
            "--workers", str(args.workers),
            "--lr", str(args.lr),
            "--weight-decay", "0.00001",
            "--hidden", "96",
            "--blocks", "2",
            "--max-delta", "0.04",
            "--stride", "20",
            "--train-alpha", "1.0",
            "--tke", "0.06",
            "--train-window-mode", "fixed",
            "--angle-aug-max-deg", "0",
            "--aoa-meanfield-aug-prob", "0.5",
            "--aoa-meanfield-lambda-min", "0.2",
            "--aoa-meanfield-lambda-max", "0.5",
            "--aoa-neighbor-max-gap-deg", "5.1",
            "--aoa-min-eligible-fraction", "0.8",
            "--bound-abs", "0.0075",
            "--bound-rel", "0.0075",
            "--fixed-time-seconds", str(FIXED_TIME_SECONDS),
            "--seed", "41",
        ], args.out_root / "A1_aoa_meanfield.log", command_log)

        run([
            sys.executable,
            "-u",
            "-B",
            str(REPO_ROOT / "tools" / "build_training_review_log.py"),
            "--input", str(args.out_root / "A1_aoa_meanfield.log"),
            "--output", str(args.out_root / "A1_aoa_meanfield.train.review.log"),
        ], args.out_root / "review_log_builder.log", command_log)

        build_training_progress(
            candidate_dir,
            updates=args.updates,
            eval_interval=args.eval_interval,
        )
        build_matched_budget_comparison(
            candidate_dir,
            matched_control_step=args.matched_control_step,
            batch_size=args.batch_size,
        )

        comparison_dir = args.out_root / "checkpoint_comparison"
        run([
            sys.executable,
            "-u",
            "-B",
            str(SCRIPT_DIR / "analyze_checkpoints.py"),
            "--real-root", str(args.real_root),
            "--split-manifest", str(args.split_manifest),
            "--realpdebench-root", str(args.model_root),
            "--out-root", str(comparison_dir),
            "--model", f"current80={args.start_checkpoint}",
            "--model", f"aoa_best={candidate_dir / 'model_best.pth'}",
            "--model", f"aoa_final={candidate_dir / 'model_final.pth'}",
            "--baseline-label", "current80",
            "--batch-size", "32",
            "--workers", str(args.workers),
        ], args.out_root / "checkpoint_comparison.log", command_log)

        recovery = {
            "execution_commit": execution_commit,
            "candidate_best_checkpoint_sha256": sha256(candidate_dir / "model_best.pth"),
            "candidate_final_checkpoint_sha256": sha256(candidate_dir / "model_final.pth"),
            "locked_final_accessed": False,
            "codabench_accessed": False,
            "automatic_full_train_started": False,
        }
        (args.out_root / "diagnostic_manifest.json").write_text(
            json.dumps(recovery, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.out_root / "DONE").touch()
        print(json.dumps({
            "status": "REVIEW_REQUIRED",
            "execution_commit": execution_commit,
            "candidate": str(candidate_dir),
            "comparison": str(comparison_dir),
            "full_training_started": False,
            "codabench_accessed": False,
            "locked_final_accessed": False,
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
