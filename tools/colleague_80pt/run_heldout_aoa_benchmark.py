#!/usr/bin/env python3
"""Run a clean PIV-heldout 10-degree benchmark: Control vs AoA augmentation.

Protocol:
- source split: frozen 50 Train / 16 Dev only
- train: frozen Train50 excluding all 10-degree PIV
- inner dev: frozen Dev16 excluding all 10-degree PIV
- heldout test: all 10-degree PIV from frozen Train50 + Dev16
- locked-final is never read
- both arms start from the same official sim-pretrained CNO
- heldout test is evaluated only after both arms finish training

The augmentation arm bridges 5 <-> 15 degree same-Re trajectories using Past20
mean-field interpolation.  This intentionally targets the missing 10-degree gap.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from build_heldout_aoa_split import build_split
from run_incremental_screen import require_clean_main_checkout, require_gpu, sha256


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SIM_PRETRAIN_SHA256 = "82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61"
CNO_UPDATES = 8723
RESIDUAL_UPDATES = 20000
RESIDUAL_EVAL_INTERVAL = 2500
HELDOUT_AOA = 10.0


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


def verify_files(real_root: Path, data_manifest: Path, names: set[str]) -> dict[str, object]:
    with data_manifest.open(encoding="utf-8") as handle:
        rows = {row["filename"]: row for row in csv.DictReader(handle, delimiter="\t")}
    missing = []
    failed = []
    for name in sorted(names):
        path = real_root / name
        if name not in rows or not path.is_file():
            missing.append(name)
            continue
        row = rows[name]
        if path.stat().st_size != int(row["size_bytes"]) or sha256(path) != row["sha256"]:
            failed.append(name)
    if missing:
        raise FileNotFoundError(f"benchmark files missing: {missing[:10]}")
    if failed:
        raise RuntimeError(f"benchmark files failed manifest verification: {failed[:10]}")
    return {"verified_files": len(names), "locked_final_accessed": False}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def training_manifest(split: dict[str, object]) -> dict[str, object]:
    return {
        "protocol": "heldout_aoa10_train_innerdev",
        "heldout_aoa": HELDOUT_AOA,
        "train": split["train"],
        "dev": split["dev"],
        "train_dev_overlap": False,
        "heldout_files_in_manifest": False,
    }


def heldout_eval_manifest(split: dict[str, object]) -> dict[str, object]:
    return {
        "protocol": "heldout_aoa10_final_eval_only",
        "heldout_aoa": HELDOUT_AOA,
        "train": split["train"],
        "dev": split["heldout"],
        "train_dev_overlap": False,
        "selection_allowed": False,
    }


def build_review(raw_log: Path, output: Path, command_log: Path) -> None:
    run([
        sys.executable,
        "-u",
        "-B",
        str(REPO_ROOT / "tools" / "build_training_review_log.py"),
        "--input", str(raw_log),
        "--output", str(output),
    ], output.with_suffix(output.suffix + ".builder.log"), command_log)


def inspect_bridge_audit(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in payload["rows"] if row.get("eligible")]
    unique_re = sorted({float(row["re"]) for row in rows})
    if len(rows) < 4 or len(unique_re) < 2:
        raise RuntimeError(
            "insufficient 5<->15 heldout-gap bridge coverage: "
            f"eligible_trajectories={len(rows)} unique_re={len(unique_re)}"
        )
    return {
        "eligible_trajectories": len(rows),
        "eligible_fraction": float(payload["eligible_fraction"]),
        "unique_re_groups": len(unique_re),
        "re_values": unique_re,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--frozen-50-16-manifest", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--sim-pretrain-checkpoint", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_root}")
    for path in (
        args.real_root,
        args.frozen_50_16_manifest,
        args.data_manifest,
        args.sim_pretrain_checkpoint,
        args.model_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if sha256(args.sim_pretrain_checkpoint) != SIM_PRETRAIN_SHA256:
        raise RuntimeError("official sim-pretrain checkpoint SHA mismatch")

    execution_commit = require_clean_main_checkout(REPO_ROOT)
    gpu = require_gpu()
    args.out_root.mkdir(parents=True)
    command_log = args.out_root / "commands.jsonl"
    (args.out_root / "RUNNING").touch()

    try:
        split = build_split(
            args.real_root,
            args.frozen_50_16_manifest,
            heldout_aoa=HELDOUT_AOA,
        )
        counts = split["counts"]
        expected = {"train": 40, "inner_dev": 12, "heldout": 14}
        actual = {
            "train": int(counts["train"]),
            "inner_dev": int(counts["inner_dev"]),
            "heldout": int(counts["heldout"]),
        }
        if actual != expected:
            raise RuntimeError(f"heldout split count mismatch: expected={expected} actual={actual}")

        split_audit = args.out_root / "heldout_aoa_split_audit.json"
        write_json(split_audit, split)
        train_manifest_path = args.out_root / "train_innerdev_manifest.json"
        heldout_manifest_path = args.out_root / "heldout_eval_manifest.json"
        write_json(train_manifest_path, training_manifest(split))
        write_json(heldout_manifest_path, heldout_eval_manifest(split))

        allowed_names = {
            str(row["file"])
            for key in ("train", "dev", "heldout")
            for row in split[key]
        }
        verified = verify_files(args.real_root, args.data_manifest, allowed_names)

        manifest = {
            "campaign": "piv_heldout_aoa10_control_vs_aug",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "execution_commit": execution_commit,
            "gpu": gpu,
            "sim_pretrain_checkpoint": str(args.sim_pretrain_checkpoint),
            "sim_pretrain_sha256": sha256(args.sim_pretrain_checkpoint),
            "frozen_50_16_manifest": str(args.frozen_50_16_manifest),
            "frozen_50_16_manifest_sha256": sha256(args.frozen_50_16_manifest),
            "data_manifest": str(args.data_manifest),
            "data_manifest_sha256": sha256(args.data_manifest),
            "verified": verified,
            "split_counts": actual,
            "scientific_semantics": {
                "heldout_aoa": HELDOUT_AOA,
                "real_piv_10deg_used_in_training": False,
                "real_piv_10deg_used_for_checkpoint_selection": False,
                "official_sim_pretrain_shared_by_both_arms": True,
                "augmentation_arm_bridge_pair": [5.0, 15.0],
                "augmentation_probability": 0.5,
                "lambda_range": [0.35, 0.65],
                "same_re_required": True,
                "future_used_to_construct_shift": False,
            },
            "budgets": {
                "cno_updates_per_arm": CNO_UPDATES,
                "residual_updates_per_arm": RESIDUAL_UPDATES,
                "residual_eval_interval": RESIDUAL_EVAL_INTERVAL,
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
        write_json(args.out_root / "campaign_manifest.json", manifest)

        arms = {
            "control": False,
            "aug": True,
        }
        residual_dirs: dict[str, Path] = {}

        # Phase 1: train both CNO backbones. Heldout manifest is not passed to either trainer.
        for arm, use_aug in arms.items():
            arm_root = args.out_root / arm
            arm_root.mkdir()
            cno_path = arm_root / "cno_stage1.pth"
            cmd = [
                sys.executable, "-u", "-B",
                str(SCRIPT_DIR / "train_cno_heldout_aoa.py"),
                "--real-root", str(args.real_root),
                "--split-manifest", str(train_manifest_path),
                "--init-checkpoint", str(args.sim_pretrain_checkpoint),
                "--realpdebench-root", str(args.model_root),
                "--out", str(cno_path),
                "--updates", str(CNO_UPDATES),
                "--batch-size", "8",
                "--workers", str(args.workers),
                "--lr", "0.0001",
                "--seed", "41",
            ]
            if use_aug:
                cmd += [
                    "--aoa-bridge",
                    "--bridge-low", "5",
                    "--bridge-high", "15",
                    "--aug-prob", "0.5",
                    "--lambda-min", "0.35",
                    "--lambda-max", "0.65",
                    "--max-gap-deg", "10.1",
                ]
            raw_log = arm_root / "cno_stage1.log"
            run(cmd, raw_log, command_log)
            build_review(raw_log, arm_root / "cno_stage1.train.review.log", command_log)

        aug_audit_path = (args.out_root / "aug" / "cno_stage1.aoa_audit.json")
        bridge_coverage = inspect_bridge_audit(aug_audit_path)
        manifest["bridge_coverage"] = bridge_coverage
        write_json(args.out_root / "campaign_manifest.json", manifest)

        # Phase 2: train residual correctors using only train40 and select on inner-dev12.
        for arm, use_aug in arms.items():
            arm_root = args.out_root / arm
            cno_path = arm_root / "cno_stage1.pth"
            residual_dir = arm_root / "residual"
            residual_dirs[arm] = residual_dir
            cmd = [
                sys.executable, "-u", "-B",
                str(SCRIPT_DIR / "residual_multi.py"),
                "--real-root", str(args.real_root),
                "--split-manifest", str(train_manifest_path),
                "--checkpoint", str(cno_path),
                "--realpdebench-root", str(args.model_root),
                "--base-model", "cno",
                "--out-dir", str(residual_dir),
                "--updates", str(RESIDUAL_UPDATES),
                "--eval-interval", str(RESIDUAL_EVAL_INTERVAL),
                "--batch-size", "8",
                "--test-batch-size", "32",
                "--workers", str(args.workers),
                "--lr", "0.0002",
                "--weight-decay", "0.00001",
                "--hidden", "96",
                "--blocks", "2",
                "--max-delta", "0.04",
                "--stride", "20",
                "--train-alpha", "1.0",
                "--tke", "0.06",
                "--train-window-mode", "fixed",
                "--angle-aug-max-deg", "0",
                "--bound-abs", "0.0075",
                "--bound-rel", "0.0075",
                "--seed", "41",
            ]
            if use_aug:
                cmd += [
                    "--aoa-meanfield-aug-prob", "0.5",
                    "--aoa-meanfield-lambda-min", "0.35",
                    "--aoa-meanfield-lambda-max", "0.65",
                    "--aoa-neighbor-max-gap-deg", "10.1",
                    "--aoa-min-eligible-fraction", "0",
                    "--aoa-bridge-low", "5",
                    "--aoa-bridge-high", "15",
                ]
            raw_log = arm_root / "residual.log"
            run(cmd, raw_log, command_log)
            build_review(raw_log, arm_root / "residual.train.review.log", command_log)

        # Phase 3: diagnostic comparison on inner-dev, where checkpoint selection is allowed.
        inner_compare = args.out_root / "inner_dev_comparison"
        inner_models = []
        for arm in arms:
            rd = residual_dirs[arm]
            inner_models += [
                f"{arm}_cno={rd / 'model_init.pth'}",
                f"{arm}_best={rd / 'model_best.pth'}",
                f"{arm}_final={rd / 'model_final.pth'}",
            ]
        command = [
            sys.executable, "-u", "-B",
            str(SCRIPT_DIR / "analyze_checkpoints.py"),
            "--real-root", str(args.real_root),
            "--split-manifest", str(train_manifest_path),
            "--realpdebench-root", str(args.model_root),
            "--out-root", str(inner_compare),
            "--baseline-label", "control_cno",
            "--batch-size", "32",
            "--workers", str(args.workers),
        ]
        for model in inner_models:
            command += ["--model", model]
        run(command, args.out_root / "inner_dev_comparison.log", command_log)

        # Phase 4: only now expose the heldout-10 manifest to evaluation code.
        heldout_compare = args.out_root / "heldout10_comparison"
        command = [
            sys.executable, "-u", "-B",
            str(SCRIPT_DIR / "analyze_checkpoints.py"),
            "--real-root", str(args.real_root),
            "--split-manifest", str(heldout_manifest_path),
            "--realpdebench-root", str(args.model_root),
            "--out-root", str(heldout_compare),
            "--baseline-label", "control_cno",
            "--batch-size", "32",
            "--workers", str(args.workers),
        ]
        for model in inner_models:
            command += ["--model", model]
        run(command, args.out_root / "heldout10_comparison.log", command_log)

        evidence = {
            "execution_commit": execution_commit,
            "split_counts": actual,
            "bridge_coverage": bridge_coverage,
            "control_cno_sha256": sha256(args.out_root / "control" / "cno_stage1.pth"),
            "aug_cno_sha256": sha256(args.out_root / "aug" / "cno_stage1.pth"),
            "control_best_residual_sha256": sha256(residual_dirs["control"] / "model_best.pth"),
            "aug_best_residual_sha256": sha256(residual_dirs["aug"] / "model_best.pth"),
            "heldout_evaluated_after_all_training": True,
            "locked_final_accessed": False,
            "codabench_accessed": False,
            "automatic_full_train_started": False,
        }
        write_json(args.out_root / "diagnostic_manifest.json", evidence)
        (args.out_root / "DONE").touch()
        print(json.dumps({"status": "REVIEW_REQUIRED", **evidence}, indent=2), flush=True)
    except BaseException as error:
        write_json(args.out_root / "FAILED.json", {
            "type": type(error).__name__,
            "message": str(error),
        })
        raise
    finally:
        (args.out_root / "RUNNING").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
