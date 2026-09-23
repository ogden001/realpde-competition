#!/usr/bin/env python3
"""Screen dense temporal continuation of the frozen colleague 80-point residual.

Scientific variable:
    historical matched control: continue current80 residual for 5k updates at stride=20
    candidate:                  continue current80 residual for 5k updates at stride=1

Everything else is frozen to the 2026-09-21 R0 continuation protocol.
No long follow-up, packaging, Codabench, or locked-final access is allowed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
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
UPDATES = 5_000
EVAL_INTERVAL = 1_000

CURRENT80 = {
    "rel_l2_raw": 0.0804204195737838,
    "tke_raw": 0.4491582512855530,
    "mvpe_raw": 0.0711169168353080,
}
HISTORICAL_SPARSE_5K = {
    "rel_l2_raw": 0.0802410691976547,
    "tke_raw": 0.4503658413887024,
    "mvpe_raw": 0.0710276663303375,
}
HISTORICAL_CONTROL = {
    "name": "R0_sparse_stride20_continuation_5k",
    "execution_commit": "ef9c54f621efcb82703cb2de40ce979c40df3c6a",
    "evidence_path": "docs/colleague_screening/results/20260921",
    "updates": 5_000,
    "stride": 20,
    "eval_interval": 1_000,
    "batch_size": 8,
    "lr": 0.0002,
    "weight_decay": 0.00001,
    "hidden": 96,
    "blocks": 2,
    "max_delta": 0.04,
    "tke": 0.06,
    "seed": 41,
    **HISTORICAL_SPARSE_5K,
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
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: {command}"
        )


def pct(candidate: float, baseline: float) -> float:
    return 100.0 * (float(candidate) - float(baseline)) / max(abs(float(baseline)), 1e-12)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_progress(candidate_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [{
        "step": 0,
        **CURRENT80,
        "source": "frozen_current80",
    }]
    for step in range(EVAL_INTERVAL, UPDATES + 1, EVAL_INTERVAL):
        metrics = alpha_one_metrics(candidate_dir / f"eval_step_{step:05d}.json")
        rows.append({
            "step": step,
            "rel_l2_raw": metrics["rel_l2_raw"],
            "tke_raw": metrics["tke_raw"],
            "mvpe_raw": metrics["mvpe_raw"],
            "source": "dense_stride1_continuation",
        })
    write_csv(candidate_dir / "training_progress.csv", rows)
    return rows


def build_matched_comparison(candidate_dir: Path) -> list[dict[str, object]]:
    dense = alpha_one_metrics(candidate_dir / f"eval_step_{UPDATES:05d}.json")
    rows = [
        {
            "model": "current80",
            "role": "frozen_start",
            **CURRENT80,
        },
        {
            "model": "historical_sparse_5k",
            "role": "matched_control_stride20",
            **HISTORICAL_SPARSE_5K,
        },
        {
            "model": "dense_5k",
            "role": "candidate_stride1",
            "rel_l2_raw": dense["rel_l2_raw"],
            "tke_raw": dense["tke_raw"],
            "mvpe_raw": dense["mvpe_raw"],
        },
    ]
    for row in rows:
        for metric in ("rel_l2_raw", "tke_raw", "mvpe_raw"):
            row[f"delta_{metric}_pct_vs_current80"] = pct(
                float(row[metric]), CURRENT80[metric]
            )
    for metric in ("rel_l2_raw", "tke_raw", "mvpe_raw"):
        rows[-1][f"delta_{metric}_pct_vs_sparse5k"] = pct(
            float(rows[-1][metric]), HISTORICAL_SPARSE_5K[metric]
        )
    write_csv(candidate_dir / "comparison_at_5k.csv", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--start-checkpoint", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_root}")
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
        raise RuntimeError("Stage-1 all81 CNO checkpoint SHA-256 mismatch")
    if sha256(args.start_checkpoint) != EXPECTED_START_SHA256:
        raise RuntimeError("frozen current80 residual checkpoint SHA-256 mismatch")

    execution_commit = require_clean_main_checkout(REPO_ROOT)
    gpu = require_gpu()
    verified_data = verify_allowed_data(
        args.real_root,
        args.data_manifest,
        args.split_manifest,
    )

    split_sha = sha256(args.split_manifest)
    data_sha = sha256(args.data_manifest)
    if split_sha != "d127e851f3f5eefb011313b1b0b1d79e65db6d99c0754d264119ff12046a83a2":
        raise RuntimeError("split manifest SHA does not match historical R0 control")
    if data_sha != "3612185c939f6c4cd4890afd3968d160a44e88d3e6f7733b8cb9ce71501d532c":
        raise RuntimeError("data manifest SHA does not match historical R0 control")

    args.out_root.mkdir(parents=True)
    command_log = args.out_root / "commands.jsonl"
    candidate_dir = args.out_root / "dense_stride1_5k"
    shutil.copy2(args.split_manifest, args.out_root / "colleague_dev16_manifest.json")
    shutil.copy2(args.data_manifest, args.out_root / "data_manifest.tsv")

    campaign = {
        "campaign": "current80_dense_residual_continuation_5k",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_commit": execution_commit,
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "start_checkpoint": str(args.start_checkpoint),
        "start_checkpoint_sha256": sha256(args.start_checkpoint),
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": split_sha,
        "data_manifest": str(args.data_manifest),
        "data_manifest_sha256": data_sha,
        "verified_data": verified_data,
        "gpu": gpu,
        "candidate": {
            "scientific_variable": "residual_continuation_training_stride",
            "stride": 1,
            "updates": UPDATES,
            "eval_interval": EVAL_INTERVAL,
            "batch_size": 8,
            "lr": 0.0002,
            "weight_decay": 0.00001,
            "hidden": 96,
            "blocks": 2,
            "max_delta": 0.04,
            "tke": 0.06,
            "train_window_mode": "fixed",
            "seed": 41,
            "optimizer_policy": "reset AdamW and cosine scheduler, matching historical R0",
        },
        "historical_matched_control": HISTORICAL_CONTROL,
        "known_prior": {
            "dense_from_scratch_residual": "NO_GO (-2.9% in colleague handoff history)",
            "question_here": "Does dense stride help specifically as continuation from the mature current80 residual?",
        },
        "automatic_long_training_started": False,
        "codabench_accessed": False,
        "locked_final_accessed": False,
        "environment": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES",)
            if os.environ.get(key)
        },
    }
    (args.out_root / "campaign_manifest.json").write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_root / "RUNNING").touch()

    try:
        raw_log = args.out_root / "dense_stride1_5k.log"
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
            "--updates", str(UPDATES),
            "--eval-interval", str(EVAL_INTERVAL),
            "--batch-size", "8",
            "--test-batch-size", "32",
            "--workers", str(args.workers),
            "--lr", "0.0002",
            "--weight-decay", "0.00001",
            "--hidden", "96",
            "--blocks", "2",
            "--max-delta", "0.04",
            "--stride", "1",
            "--train-alpha", "1.0",
            "--tke", "0.06",
            "--train-window-mode", "fixed",
            "--bound-abs", "0.0075",
            "--bound-rel", "0.0075",
            "--fixed-time-seconds", str(FIXED_TIME_SECONDS),
            "--seed", "41",
        ], raw_log, command_log)

        run([
            sys.executable,
            "-u",
            "-B",
            str(REPO_ROOT / "tools" / "build_training_review_log.py"),
            "--input", str(raw_log),
            "--output", str(args.out_root / "dense_stride1_5k.train.review.log"),
        ], args.out_root / "review_log_builder.log", command_log)

        progress = build_progress(candidate_dir)
        comparison = build_matched_comparison(candidate_dir)

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
            "--model", f"dense_best={candidate_dir / 'model_best.pth'}",
            "--model", f"dense_final={candidate_dir / 'model_final.pth'}",
            "--baseline-label", "current80",
            "--batch-size", "32",
            "--workers", str(args.workers),
        ], args.out_root / "checkpoint_comparison.log", command_log)

        final_dense = comparison[-1]
        decision_input = {
            "current80": CURRENT80,
            "historical_sparse_5k": HISTORICAL_SPARSE_5K,
            "dense_5k": {
                key: float(final_dense[key])
                for key in ("rel_l2_raw", "tke_raw", "mvpe_raw")
            },
            "dense_vs_sparse_pct": {
                metric: pct(
                    float(final_dense[metric]),
                    HISTORICAL_SPARSE_5K[metric],
                )
                for metric in ("rel_l2_raw", "tke_raw", "mvpe_raw")
            },
            "note": "No automatic GO/NO-GO. Sol reviews evidence after Git archive.",
        }
        (args.out_root / "decision_input.json").write_text(
            json.dumps(decision_input, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        evidence = {
            "execution_commit": execution_commit,
            "candidate_best_checkpoint_sha256": sha256(candidate_dir / "model_best.pth"),
            "candidate_final_checkpoint_sha256": sha256(candidate_dir / "model_final.pth"),
            "candidate_best_iteration": json.loads(
                (candidate_dir / "summary.json").read_text(encoding="utf-8")
            )["best_iter"],
            "training_progress_rows": len(progress),
            "historical_control_execution_commit": HISTORICAL_CONTROL["execution_commit"],
            "historical_control_evidence_path": HISTORICAL_CONTROL["evidence_path"],
            "automatic_long_training_started": False,
            "codabench_accessed": False,
            "locked_final_accessed": False,
        }
        (args.out_root / "diagnostic_manifest.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.out_root / "DONE").touch()
        print(json.dumps({"status": "REVIEW_REQUIRED", **evidence}, indent=2), flush=True)
    except BaseException as error:
        (args.out_root / "FAILED.json").write_text(
            json.dumps(
                {"type": type(error).__name__, "message": str(error)},
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        (args.out_root / "RUNNING").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
