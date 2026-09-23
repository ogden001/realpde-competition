#!/usr/bin/env python3
"""Residual-level disjoint trajectory-sampling A/B screen.

This experiment isolates the sampling-policy question at the residual stage:

* Stage-1 CNO is frozen and identical for both arms.  It was historically fit
  on all81 released PIV trajectories, so this is NOT an end-to-end clean
  holdout experiment.
* Residual training is restricted to Train65 = all81 - frozen Dev16.
* Residual evaluation uses the untouched Dev16 trajectories.
* Both residuals start fresh from the exact same initialization.
* The only scientific variable is the residual training sampling policy.

Arm A:
    Train65 fixed starts 0,20,40,..., global shuffle.

Arm B:
    Same Train65 and exact per-trajectory draw quotas as Arm A, but each
    trajectory draws from a deterministic shuffled bag of all legal starts and
    batches contain distinct trajectories.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from run_incremental_screen import (
    EXPECTED_BASE_SHA256,
    FIXED_TIME_SECONDS,
    require_clean_main_checkout,
    require_gpu,
    sha256,
    verify_allowed_data,
)
from run_trajectory_sampling_screen import (
    alpha_one_metrics,
    compare_initial_states,
    pct,
    sampling_summary,
    write_csv,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

UPDATES = 5_055
EVAL_INTERVAL = 1_000
BATCH_SIZE = 8
LR = 2e-4
WEIGHT_DECAY = 1e-5
SEED = 41

EXPECTED_SOURCE_SPLIT_SHA256 = (
    "d127e851f3f5eefb011313b1b0b1d79e65db6d99c0754d264119ff12046a83a2"
)
EXPECTED_DATA_SHA256 = (
    "3612185c939f6c4cd4890afd3968d160a44e88d3e6f7733b8cb9ce71501d532c"
)
EXPECTED_TRAIN_TRAJECTORIES = 65
EXPECTED_DEV_TRAJECTORIES = 16
EXPECTED_FIXED_WINDOWS = 2_701
EXPECTED_SAMPLES_PER_EPOCH = 2_696
EXPECTED_BATCHES_PER_EPOCH = 337
EXPECTED_FULL_EPOCHS = 15


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


def _manifest_names(payload: dict[str, object], key: str) -> list[str]:
    rows = payload.get(key)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"source manifest has no non-empty {key!r} list")
    names = [
        str(row["file"] if isinstance(row, dict) else row)
        for row in rows
    ]
    if len(names) != len(set(names)):
        raise ValueError(f"source manifest has duplicate {key} filenames")
    return names


def derive_residual_disjoint_manifest(
    source_manifest: Path,
    destination: Path,
) -> dict[str, object]:
    """Create Train65/Dev16 manifest by subtracting frozen Dev16 from all81."""
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    if payload.get("protocol") != "colleague_dev16_seed41_all81":
        raise ValueError("unexpected source split protocol")
    if payload.get("seed") != SEED:
        raise ValueError("source split seed mismatch")
    if payload.get("train_dev_overlap") is not True:
        raise ValueError("source split must be the historical all81/Dev16 overlap manifest")

    all81 = _manifest_names(payload, "train")
    dev16 = _manifest_names(payload, "dev")
    dev_set = set(dev16)
    if len(all81) != 81 or len(dev16) != EXPECTED_DEV_TRAJECTORIES:
        raise ValueError("source split must contain all81 train and Dev16")
    if not dev_set.issubset(set(all81)):
        raise ValueError("Dev16 must be a subset of source all81")

    train65 = [name for name in all81 if name not in dev_set]
    if len(train65) != EXPECTED_TRAIN_TRAJECTORIES:
        raise ValueError(f"derived residual train split has {len(train65)} trajectories, expected 65")
    if set(train65) & dev_set:
        raise RuntimeError("derived residual Train65 overlaps Dev16")

    derived = {
        "protocol": "colleague_residual_train65_dev16_disjoint_v1",
        "source_protocol": payload.get("protocol"),
        "source_manifest_sha256": sha256(source_manifest),
        "seed": SEED,
        "train": train65,
        "dev": dev16,
        "train_dev_overlap": False,
        "residual_stage_disjoint": True,
        "stage1_cno_trained_on_all81": True,
        "end_to_end_clean_holdout": False,
        "note": (
            "Residual-only disjoint split: Train65 = source all81 - frozen Dev16. "
            "The frozen Stage-1 CNO historically saw all81, so this tests residual "
            "sampler generalization only, not end-to-end unseen-trajectory generalization."
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(derived, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return derived


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
    ]


def build_progress(
    control_dir: Path,
    candidate_dir: Path,
    out_path: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    steps = [0, 1000, 2000, 3000, 4000, 5000, UPDATES]
    for step in steps:
        control = alpha_one_metrics(control_dir / f"eval_step_{step:05d}.json")
        candidate = alpha_one_metrics(candidate_dir / f"eval_step_{step:05d}.json")
        rows.append(
            {
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
        )
    write_csv(out_path, rows)
    return rows


def per_trajectory_draws(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["trajectory"]): int(row["draws"])
        for row in payload["per_trajectory"]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--source-split-manifest", type=Path, required=True)
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
        args.source_split_manifest,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    if sha256(args.base_checkpoint) != EXPECTED_BASE_SHA256:
        raise RuntimeError("all81 Stage-1 CNO checkpoint SHA-256 mismatch")
    if sha256(args.source_split_manifest) != EXPECTED_SOURCE_SPLIT_SHA256:
        raise RuntimeError("source all81/Dev16 split manifest SHA mismatch")
    if sha256(args.data_manifest) != EXPECTED_DATA_SHA256:
        raise RuntimeError("data manifest SHA mismatch")

    execution_commit = require_clean_main_checkout(REPO_ROOT)
    gpu = require_gpu()

    # Validate the original released-data provenance first.  This function
    # intentionally validates the historical all81/Dev16 source manifest.
    verified_source_data = verify_allowed_data(
        args.real_root,
        args.data_manifest,
        args.source_split_manifest,
    )

    args.out_root.mkdir(parents=True)
    command_log = args.out_root / "commands.jsonl"
    control_dir = args.out_root / "A_train65_fixed_stride20"
    candidate_dir = args.out_root / "B_train65_stratified_random_start"
    derived_manifest_path = args.out_root / "residual_train65_dev16_manifest.json"
    derived = derive_residual_disjoint_manifest(
        args.source_split_manifest,
        derived_manifest_path,
    )
    shutil.copy2(args.source_split_manifest, args.out_root / "source_all81_dev16_manifest.json")
    shutil.copy2(args.data_manifest, args.out_root / "data_manifest.tsv")

    campaign = {
        "campaign": "residual_train65_dev16_disjoint_sampling_screen",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_commit": execution_commit,
        "scientific_question": (
            "At the residual stage, with frozen all81 CNO and fresh identical residuals, "
            "does broader temporal-start coverage on Train65 improve performance on "
            "residual-unseen Dev16 trajectories compared with fixed stride20 sampling?"
        ),
        "scientific_variable": "residual_training_sampling_policy_only",
        "scope": {
            "residual_stage_train_dev_disjoint": True,
            "stage1_cno_saw_dev16": True,
            "end_to_end_clean_holdout": False,
        },
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "source_split_manifest": str(args.source_split_manifest),
        "source_split_manifest_sha256": sha256(args.source_split_manifest),
        "derived_split_manifest": str(derived_manifest_path),
        "derived_split": derived,
        "data_manifest": str(args.data_manifest),
        "data_manifest_sha256": sha256(args.data_manifest),
        "verified_source_data": verified_source_data,
        "gpu": gpu,
        "common_training": {
            "residual_initialization": "fresh_zero_output_layer_same_seed",
            "resume_checkpoint": None,
            "train_trajectories": EXPECTED_TRAIN_TRAJECTORIES,
            "dev_trajectories": EXPECTED_DEV_TRAJECTORIES,
            "updates": UPDATES,
            "budget_note": (
                f"{UPDATES} updates = {EXPECTED_FULL_EPOCHS} x "
                f"{EXPECTED_BATCHES_PER_EPOCH} matched full epochs"
            ),
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
            "expected_fixed_stride_windows": EXPECTED_FIXED_WINDOWS,
            "expected_samples_per_epoch_after_drop_last": EXPECTED_SAMPLES_PER_EPOCH,
            "precision": "fp32",
        },
        "arms": {
            "A_train65_fixed_stride20": {
                "sampling_policy": "Train65 fixed starts 0,20,40,... with global shuffle"
            },
            "B_train65_stratified_random_start": {
                "sampling_policy": (
                    "same Train65 per-trajectory draw quotas as A each epoch; "
                    "distinct trajectories within each batch; shuffled all-legal-start "
                    "bag per trajectory"
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
            ("A_train65_fixed_stride20", "fixed", control_dir),
            (
                "B_train65_stratified_random_start",
                "trajectory_stratified_random_start",
                candidate_dir,
            ),
        ):
            raw_log = args.out_root / f"{label}.log"
            command = build_arm_command(
                python=sys.executable,
                real_root=args.real_root,
                split_manifest=derived_manifest_path,
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
            if int(config["train_trajectories"]) != EXPECTED_TRAIN_TRAJECTORIES:
                raise RuntimeError(f"{label} train trajectory count mismatch")
            if int(config["fit_trajectories"]) != EXPECTED_TRAIN_TRAJECTORIES:
                raise RuntimeError(f"{label} fit trajectory count mismatch")
            if int(config["val_trajectories"]) != EXPECTED_DEV_TRAJECTORIES:
                raise RuntimeError(f"{label} Dev trajectory count mismatch")
            if int(config["reference_fixed_stride_windows"]) != EXPECTED_FIXED_WINDOWS:
                raise RuntimeError(
                    f"{label} does not reproduce expected Train65 fixed-window count"
                )
            if int(config["train_samples_per_epoch"]) != EXPECTED_SAMPLES_PER_EPOCH:
                raise RuntimeError(
                    f"{label} consumed-samples-per-epoch mismatch after drop-last"
                )
            if bool(config["phase_counts_equalized"]):
                raise RuntimeError(f"{label} unexpectedly equalized phase counts")
            if bool(config["allow_train_dev_overlap"]):
                raise RuntimeError(f"{label} unexpectedly allows residual train/dev overlap")
            if config["resume_checkpoint"] is not None:
                raise RuntimeError(f"{label} unexpectedly resumes a residual checkpoint")

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
        control_sampling = sampling_summary(control_sampling_path)
        candidate_sampling = sampling_summary(candidate_sampling_path)
        control_draws = per_trajectory_draws(control_sampling_path)
        candidate_draws = per_trajectory_draws(candidate_sampling_path)
        per_trajectory_draws_matched = control_draws == candidate_draws

        sampling_comparison = {
            "control": control_sampling,
            "candidate": candidate_sampling,
            "same_samples_consumed": (
                int(control_sampling["samples_consumed"])
                == int(candidate_sampling["samples_consumed"])
            ),
            "per_trajectory_draws_matched": per_trajectory_draws_matched,
            "candidate_batch_trajectory_duplicates": int(
                candidate_sampling["duplicate_trajectory_batches"]
            ),
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
            raise RuntimeError("A/B total consumed sample counts are not matched")
        if not per_trajectory_draws_matched:
            raise RuntimeError("A/B per-trajectory sample exposure is not exactly matched")
        if sampling_comparison["candidate_batch_trajectory_duplicates"] != 0:
            raise RuntimeError("candidate sampler produced duplicate trajectories in a batch")
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
                "--split-manifest", str(derived_manifest_path),
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
            "primary_comparison": "candidate_final_vs_control_final_at_5055",
            "residual_stage_train_dev_disjoint": True,
            "stage1_cno_saw_dev16": True,
            "end_to_end_clean_holdout": False,
            "step_5055": final_row,
            "sampling_exposure": sampling_comparison,
            "initialization_parity": init_parity,
            "note": (
                "No automatic scientific verdict and no automatic extension. "
                "This is a residual-stage differential generalization test only."
            ),
        }
        (args.out_root / "decision_input.json").write_text(
            json.dumps(decision_input, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        evidence = {
            "execution_commit": execution_commit,
            "derived_train65_dev16_disjoint": "PASS",
            "initialization_parity": "PASS",
            "train65_fixed_windows": EXPECTED_FIXED_WINDOWS,
            "samples_per_epoch": EXPECTED_SAMPLES_PER_EPOCH,
            "full_matched_epochs": EXPECTED_FULL_EPOCHS,
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
