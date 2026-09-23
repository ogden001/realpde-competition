#!/usr/bin/env python3
"""Matched residual sampler screen for RealPDE Track 1.

Arm A (control):
    frozen all81 CNO -> zero-init h96/b2 residual
    historical fixed stride=20 start set, global shuffle

Arm B (candidate):
    same frozen CNO -> identically initialized zero-init h96/b2 residual
    same per-trajectory sample quotas as Arm A for every epoch
    but starts are drawn from deterministic shuffled per-trajectory bags and
    each batch contains distinct trajectories.

Only the training sampling policy may differ.
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
from typing import Sequence

import torch

from run_incremental_screen import (
    EXPECTED_BASE_SHA256,
    FIXED_TIME_SECONDS,
    require_clean_main_checkout,
    require_gpu,
    sha256,
    verify_allowed_data,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
UPDATES = 5_004
EVAL_INTERVAL = 1_000
BATCH_SIZE = 8
LR = 2e-4
WEIGHT_DECAY = 1e-5
SEED = 41
EXPECTED_SPLIT_SHA256 = "d127e851f3f5eefb011313b1b0b1d79e65db6d99c0754d264119ff12046a83a2"
EXPECTED_DATA_SHA256 = "3612185c939f6c4cd4890afd3968d160a44e88d3e6f7733b8cb9ce71501d532c"


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


def alpha_one_metrics(path: Path) -> dict[str, float]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        if abs(float(row["alpha"]) - 1.0) < 1e-12:
            return {
                "rel_l2_raw": float(
                    row["rel_l2_raw"] if "rel_l2_raw" in row else row["rel_l2"]
                ),
                "tke_raw": float(
                    row["tke_raw"] if "tke_raw" in row else row["tke"]
                ),
                "mvpe_raw": float(
                    row["mvpe_raw"] if "mvpe_raw" in row else row["mvpe"]
                ),
                "final_est": float(row["best_final_est"]),
            }
    raise RuntimeError(f"alpha=1.0 missing from {path}")


def pct(candidate: float, control: float) -> float:
    return 100.0 * (float(candidate) - float(control)) / max(abs(float(control)), 1e-12)


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
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


def build_arm_command(
    *,
    python: str,
    real_root: Path,
    split_manifest: Path,
    base_checkpoint: Path,
    model_root: Path,
    out_dir: Path,
    workers: int,
    sampling_mode: str,
) -> list[str]:
    if sampling_mode not in {"fixed", "trajectory_stratified_random_start"}:
        raise ValueError(f"unsupported sampling mode: {sampling_mode}")
    return [
        python,
        "-u",
        "-B",
        str(SCRIPT_DIR / "residual_multi.py"),
        "--real-root", str(real_root),
        "--split-manifest", str(split_manifest),
        "--allow-train-dev-overlap",
        "--checkpoint", str(base_checkpoint),
        "--realpdebench-root", str(model_root),
        "--base-model", "cno",
        "--out-dir", str(out_dir),
        "--updates", str(UPDATES),
        "--eval-interval", str(EVAL_INTERVAL),
        "--batch-size", str(BATCH_SIZE),
        "--test-batch-size", "32",
        "--workers", str(workers),
        "--lr", str(LR),
        "--weight-decay", str(WEIGHT_DECAY),
        "--hidden", "96",
        "--blocks", "2",
        "--max-delta", "0.04",
        "--stride", "20",
        "--eval-stride", "20",
        "--train-alpha", "1.0",
        "--tke", "0.06",
        "--train-window-mode", sampling_mode,
        "--disable-phase-count-equalization",
        "--bound-abs", "0.0075",
        "--bound-rel", "0.0075",
        "--fixed-time-seconds", str(FIXED_TIME_SECONDS),
        "--seed", str(SEED),
        "--train-on-all",
    ]


def compare_initial_states(control_path: Path, candidate_path: Path) -> dict[str, object]:
    control = torch.load(control_path, map_location="cpu")
    candidate = torch.load(candidate_path, map_location="cpu")
    control_state = control.get("model_state_dict", control)
    candidate_state = candidate.get("model_state_dict", candidate)
    if set(control_state) != set(candidate_state):
        return {
            "exact": False,
            "reason": "state_dict_keys_differ",
            "control_keys": len(control_state),
            "candidate_keys": len(candidate_state),
        }

    max_abs = 0.0
    mismatch_keys: list[str] = []
    for key in sorted(control_state):
        left = control_state[key]
        right = candidate_state[key]
        if torch.is_tensor(left) and torch.is_tensor(right):
            if left.shape != right.shape:
                mismatch_keys.append(key)
                continue
            if left.numel():
                max_abs = max(max_abs, float((left - right).abs().max().item()))
            if not torch.equal(left, right):
                mismatch_keys.append(key)
        elif left != right:
            mismatch_keys.append(key)
    return {
        "exact": not mismatch_keys,
        "state_keys": len(control_state),
        "max_abs_difference": max_abs,
        "mismatch_keys": mismatch_keys[:20],
    }


def build_progress(control_dir: Path, candidate_dir: Path, out_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    steps = list(range(0, 5_001, EVAL_INTERVAL)) + [UPDATES]
    for step in steps:
        control = alpha_one_metrics(control_dir / f"eval_step_{step:05d}.json")
        candidate = alpha_one_metrics(candidate_dir / f"eval_step_{step:05d}.json")
        row: dict[str, object] = {
            "step": step,
            "control_rel_l2_raw": control["rel_l2_raw"],
            "candidate_rel_l2_raw": candidate["rel_l2_raw"],
            "delta_rel_l2_pct_candidate_vs_control": pct(
                candidate["rel_l2_raw"], control["rel_l2_raw"]
            ),
            "control_tke_raw": control["tke_raw"],
            "candidate_tke_raw": candidate["tke_raw"],
            "delta_tke_pct_candidate_vs_control": pct(
                candidate["tke_raw"], control["tke_raw"]
            ),
            "control_mvpe_raw": control["mvpe_raw"],
            "candidate_mvpe_raw": candidate["mvpe_raw"],
            "delta_mvpe_pct_candidate_vs_control": pct(
                candidate["mvpe_raw"], control["mvpe_raw"]
            ),
            "control_final_est": control["final_est"],
            "candidate_final_est": candidate["final_est"],
        }
        rows.append(row)
    write_csv(out_path, rows)
    return rows


def sampling_summary(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: payload[key]
        for key in (
            "sampling_policy",
            "samples_consumed",
            "unique_windows_consumed",
            "candidate_legal_windows",
            "unique_window_fraction",
            "duplicate_trajectory_batches",
            "trajectory_draw_min",
            "trajectory_draw_median",
            "trajectory_draw_max",
            "unique_start_min",
            "unique_start_median",
            "unique_start_max",
            "unique_start_fraction_median",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
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
        args.model_root,
        args.data_manifest,
        args.split_manifest,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    if sha256(args.base_checkpoint) != EXPECTED_BASE_SHA256:
        raise RuntimeError("all81 Stage-1 CNO checkpoint SHA-256 mismatch")
    if sha256(args.split_manifest) != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("split manifest SHA does not match frozen colleague protocol")
    if sha256(args.data_manifest) != EXPECTED_DATA_SHA256:
        raise RuntimeError("data manifest SHA does not match frozen colleague protocol")

    execution_commit = require_clean_main_checkout(REPO_ROOT)
    gpu = require_gpu()
    verified_data = verify_allowed_data(
        args.real_root,
        args.data_manifest,
        args.split_manifest,
    )

    args.out_root.mkdir(parents=True)
    command_log = args.out_root / "commands.jsonl"
    control_dir = args.out_root / "A_fixed_stride20"
    candidate_dir = args.out_root / "B_stratified_random_start"
    shutil.copy2(args.split_manifest, args.out_root / "colleague_dev16_manifest.json")
    shutil.copy2(args.data_manifest, args.out_root / "data_manifest.tsv")

    campaign = {
        "campaign": "zero_init_residual_matched_sampling_screen",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_commit": execution_commit,
        "scientific_question": (
            "With identical all81 data, frozen CNO, zero-init residual, optimizer, "
            "loss and ~5k matched-update budget, does baseline-weight-matched trajectory-"
            "stratified random-start sampling outperform fixed stride20 sampling?"
        ),
        "scientific_variable": "training_sampling_policy_only",
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": sha256(args.split_manifest),
        "data_manifest": str(args.data_manifest),
        "data_manifest_sha256": sha256(args.data_manifest),
        "verified_data": verified_data,
        "gpu": gpu,
        "common_training": {
            "resume_checkpoint": None,
            "residual_initialization": "fresh_zero_output_layer_same_seed",
            "updates": UPDATES,
            "budget_note": "5004 updates = exactly 12 x 417 matched batches",
            "eval_interval": EVAL_INTERVAL,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "hidden": 96,
            "blocks": 2,
            "max_delta": 0.04,
            "tke": 0.06,
            "seed": SEED,
            "train_stride_parameter": 20,
            "eval_stride": 20,
            "phase_count_equalization": False,
            "expected_original_stride20_windows": 3341,
            "expected_consumed_samples_per_epoch_after_drop_last": 3336,
            "precision": "fp32",
        },
        "arms": {
            "A_fixed_stride20": {
                "sampling_policy": "baseline fixed starts 0,20,40,... with global shuffle"
            },
            "B_stratified_random_start": {
                "sampling_policy": (
                    "same per-trajectory draw quotas as A each epoch; distinct trajectories "
                    "within each batch; shuffled all-legal-start bag per trajectory"
                )
            },
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "PYTORCH_CUDA_ALLOC_CONF",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "TMPDIR",
                "HDF5_USE_FILE_LOCKING",
                "PYTHONPATH",
            )
            if os.environ.get(key) is not None
        },
        "automatic_long_training_started": False,
        "codabench_accessed": False,
        "locked_final_accessed": False,
    }
    (args.out_root / "campaign_manifest.json").write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_root / "RUNNING").touch()

    try:
        for label, mode, out_dir in (
            ("A_fixed_stride20", "fixed", control_dir),
            (
                "B_stratified_random_start",
                "trajectory_stratified_random_start",
                candidate_dir,
            ),
        ):
            raw_log = args.out_root / f"{label}.log"
            command = build_arm_command(
                python=sys.executable,
                real_root=args.real_root,
                split_manifest=args.split_manifest,
                base_checkpoint=args.base_checkpoint,
                model_root=args.model_root,
                out_dir=out_dir,
                workers=args.workers,
                sampling_mode=mode,
            )
            run(command, raw_log, command_log)
            run(
                [
                    sys.executable,
                    "-u",
                    "-B",
                    str(REPO_ROOT / "tools" / "build_training_review_log.py"),
                    "--input", str(raw_log),
                    "--output", str(args.out_root / f"{label}.train.review.log"),
                ],
                args.out_root / f"{label}.review_builder.log",
                command_log,
            )

        control_config = json.loads(
            (control_dir / "run_config.json").read_text(encoding="utf-8")
        )
        candidate_config = json.loads(
            (candidate_dir / "run_config.json").read_text(encoding="utf-8")
        )
        for label, config in (
            ("control", control_config),
            ("candidate", candidate_config),
        ):
            if int(config["reference_fixed_stride_windows"]) != 3341:
                raise RuntimeError(
                    f"{label} does not reproduce original 3341-window Stage-2 baseline"
                )
            if int(config["train_samples_per_epoch"]) != 3336:
                raise RuntimeError(
                    f"{label} consumed-samples-per-epoch mismatch after batch drop-last"
                )
            if bool(config["phase_counts_equalized"]):
                raise RuntimeError(f"{label} unexpectedly equalized phase counts")

        init_parity = compare_initial_states(
            control_dir / "model_init.pth",
            candidate_dir / "model_init.pth",
        )
        (args.out_root / "initialization_parity.json").write_text(
            json.dumps(init_parity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not bool(init_parity["exact"]):
            raise RuntimeError("A/B residual initialization parity check failed")

        progress = build_progress(
            control_dir,
            candidate_dir,
            args.out_root / "matched_progress.csv",
        )
        final_row = progress[-1]

        control_sampling_path = control_dir / "sampling_audit.json"
        candidate_sampling_path = candidate_dir / "sampling_audit.json"
        control_sampling_full = json.loads(control_sampling_path.read_text(encoding="utf-8"))
        candidate_sampling_full = json.loads(candidate_sampling_path.read_text(encoding="utf-8"))
        control_sampling = sampling_summary(control_sampling_path)
        candidate_sampling = sampling_summary(candidate_sampling_path)
        control_draws = {
            row["trajectory"]: int(row["draws"])
            for row in control_sampling_full["per_trajectory"]
        }
        candidate_draws = {
            row["trajectory"]: int(row["draws"])
            for row in candidate_sampling_full["per_trajectory"]
        }
        per_trajectory_draws_matched = control_draws == candidate_draws
        sampling_comparison = {
            "control": control_sampling,
            "candidate": candidate_sampling,
            "same_samples_consumed": (
                int(control_sampling["samples_consumed"])
                == int(candidate_sampling["samples_consumed"])
            ),
            "candidate_batch_trajectory_duplicates": int(
                candidate_sampling["duplicate_trajectory_batches"]
            ),
            "per_trajectory_draws_matched": per_trajectory_draws_matched,
            "candidate_unique_window_gain": (
                int(candidate_sampling["unique_windows_consumed"])
                - int(control_sampling["unique_windows_consumed"])
            ),
        }
        (args.out_root / "sampling_comparison.json").write_text(
            json.dumps(sampling_comparison, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not sampling_comparison["same_samples_consumed"]:
            raise RuntimeError("A/B consumed sample counts are not matched")
        if sampling_comparison["candidate_batch_trajectory_duplicates"] != 0:
            raise RuntimeError("candidate sampler produced duplicate trajectories in a batch")
        if not per_trajectory_draws_matched:
            raise RuntimeError("A/B per-trajectory sample exposure is not exactly matched")
        if int(sampling_comparison["candidate_unique_window_gain"]) <= 0:
            raise RuntimeError("candidate sampler failed to increase unique temporal starts")

        comparison_dir = args.out_root / "checkpoint_comparison"
        run(
            [
                sys.executable,
                "-u",
                "-B",
                str(SCRIPT_DIR / "analyze_checkpoints.py"),
                "--real-root", str(args.real_root),
                "--split-manifest", str(args.split_manifest),
                "--realpdebench-root", str(args.model_root),
                "--out-root", str(comparison_dir),
                "--model", f"init={control_dir / 'model_init.pth'}",
                "--model", f"control_best={control_dir / 'model_best.pth'}",
                "--model", f"control_final={control_dir / 'model_final.pth'}",
                "--model", f"candidate_best={candidate_dir / 'model_best.pth'}",
                "--model", f"candidate_final={candidate_dir / 'model_final.pth'}",
                "--baseline-label", "control_final",
                "--batch-size", "32",
                "--workers", str(args.workers),
            ],
            args.out_root / "checkpoint_comparison.log",
            command_log,
        )

        decision_input = {
            "primary_comparison": "candidate_final_vs_control_final_at_5004",
            "step_5004": final_row,
            "sampling_exposure": sampling_comparison,
            "initialization_parity": init_parity,
            "note": (
                "No automatic scientific verdict and no automatic extension. "
                "Sol reviews trajectory/horizon diagnostics and sampling exposure."
            ),
        }
        (args.out_root / "decision_input.json").write_text(
            json.dumps(decision_input, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        evidence = {
            "execution_commit": execution_commit,
            "initialization_parity": "PASS",
            "original_3341_window_baseline_reproduced": "PASS",
            "control_best_iteration": json.loads(
                (control_dir / "summary.json").read_text(encoding="utf-8")
            )["best_iter"],
            "candidate_best_iteration": json.loads(
                (candidate_dir / "summary.json").read_text(encoding="utf-8")
            )["best_iter"],
            "matched_progress_rows": len(progress),
            "control_sampling_audit": "PASS",
            "candidate_sampling_audit": "PASS",
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
