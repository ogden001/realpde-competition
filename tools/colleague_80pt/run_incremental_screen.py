#!/usr/bin/env python3
"""Run the frozen R0-R3 and H0-H1 colleague incremental screen sequentially."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from incremental_screen import REGISTERED_ARMS, random_phase_gate, tke_gate


SCRIPT_DIR = Path(__file__).resolve().parent
FIXED_TIME_SECONDS = 0.01762683533
UPDATES = 5000
EVAL_INTERVAL = 1000
HEAD_UPDATES = 6000
HEAD_EVAL_INTERVAL = 500
EXPECTED_BASE_SHA256 = "ff28aaf0114d57e320e5ea9cf9945d874a7482f22e3f0aeaed5ebbe286573f8a"
EXPECTED_START_SHA256 = "909fdc7f8a6a42335e4ea4ce7471fc9a507135e9a29a4927bc7b7318be9c85b2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def verify_allowed_data(real_root: Path, data_manifest: Path, split_manifest: Path) -> dict[str, object]:
    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    if split.get("protocol") != "colleague_dev16_seed41_all81":
        raise ValueError("screen requires the colleague_dev16_seed41_all81 manifest")
    if split.get("seed") != 41 or split.get("val_fraction") != 0.2 or split.get("train_dev_overlap") is not True:
        raise ValueError("colleague split manifest metadata does not match the frozen historical protocol")
    allowed = []
    for name in ("train", "dev"):
        rows = split.get(name)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"split manifest has no non-empty {name!r} list")
        allowed.extend(str(row["file"] if isinstance(row, dict) else row) for row in rows)
    if len(split["train"]) != 81 or len(split["dev"]) != 16:
        raise ValueError("colleague split must contain exactly all81 train and Dev16")
    if not set(split["dev"]).issubset(set(split["train"])):
        raise ValueError("historical Dev16 must be a subset of all81 train")
    with data_manifest.open(encoding="utf-8") as handle:
        rows = {row["filename"]: row for row in csv.DictReader(handle, delimiter="\t")}
    missing = [name for name in allowed if name not in rows or not (real_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"allowed data missing from manifest/root: {missing[:10]}")
    failures = []
    for name in allowed:
        path = real_root / name
        row = rows[name]
        if path.stat().st_size != int(row["size_bytes"]) or sha256(path) != row["sha256"]:
            failures.append(name)
    if failures:
        raise RuntimeError(f"allowed data failed manifest verification: {failures[:10]}")
    return {
        "train": len(split["train"]),
        "dev": len(split["dev"]),
        "verified_files": len(set(allowed)),
        "train_dev_overlap": True,
    }


def require_clean_main_checkout(repo: Path) -> str:
    transferred_commit = os.environ.get("REALPDE_EXECUTION_COMMIT")
    if transferred_commit:
        if len(transferred_commit) != 40 or any(char not in "0123456789abcdef" for char in transferred_commit):
            raise ValueError("REALPDE_EXECUTION_COMMIT must be a lowercase 40-character SHA-1")
        return transferred_commit
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        raise RuntimeError("campaign requires a clean Git checkout")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    origin_main = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=repo, text=True).strip()
    if head != origin_main:
        raise RuntimeError(f"campaign HEAD {head} does not equal origin/main {origin_main}")
    return head


def require_gpu() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this campaign")
    probe = torch.ones(1, device="cuda") * 2
    if float(probe.cpu()) != 2.0:
        raise RuntimeError("CUDA allocation probe failed")
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    }


def alpha_one_composite(arm_dir: Path, updates: int) -> float:
    rows = json.loads((arm_dir / f"eval_step_{updates:05d}.json").read_text(encoding="utf-8"))
    for row in rows:
        if abs(float(row["alpha"]) - 1.0) < 1e-12:
            return float(row["best_final_est"])
    raise RuntimeError(f"alpha=1.0 missing from {arm_dir}")


def build_report(out_root: Path, updates: int) -> dict[str, object]:
    residual: dict[str, dict[str, float]] = {}
    for arm in REGISTERED_ARMS:
        metrics = json.loads((out_root / arm / "final_primary_metrics.json").read_text(encoding="utf-8"))
        metrics["final_est"] = alpha_one_composite(out_root / arm, updates)
        residual[arm] = metrics
    control = residual["R0"]
    head = {
        arm: json.loads((out_root / arm / "summary.json").read_text(encoding="utf-8"))["best"]
        for arm in ("H0", "H1")
    }
    return {
        "primary_step": updates,
        "residual_metrics": residual,
        "gates": {
            "R1": tke_gate(control, residual["R1"]),
            "R2": tke_gate(control, residual["R2"]),
            "R3": random_phase_gate(control, residual["R3"]),
            "H1": {
                "pass": float(head["H1"]["sps"]) >= float(head["H0"]["sps"]) + 0.5,
                "sps_delta": float(head["H1"]["sps"]) - float(head["H0"]["sps"]),
                "threshold": 0.5,
            },
        },
        "head_best": head,
        "competition_oriented_all81_dev16_overlap": True,
        "codabench_accessed": False,
        "automatic_long_training_started": False,
    }


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

    for path in (args.real_root, args.base_checkpoint, args.start_checkpoint, args.model_root, args.data_manifest, args.split_manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    git_commit = require_clean_main_checkout(SCRIPT_DIR.parents[1])
    gpu = require_gpu()
    if sha256(args.base_checkpoint) != EXPECTED_BASE_SHA256:
        raise RuntimeError("Stage 1 base checkpoint SHA-256 mismatch")
    if sha256(args.start_checkpoint) != EXPECTED_START_SHA256:
        raise RuntimeError("Stage 2 starting checkpoint SHA-256 mismatch")
    verified_data = verify_allowed_data(args.real_root, args.data_manifest, args.split_manifest)
    args.out_root.mkdir(parents=True, exist_ok=False)
    command_log = args.out_root / "commands.jsonl"
    manifest = {
        "run_id": args.out_root.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": sys.version,
        "git_commit": git_commit,
        "real_root": str(args.real_root),
        "data_manifest": str(args.data_manifest),
        "data_manifest_sha256": sha256(args.data_manifest),
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": sha256(args.split_manifest),
        "verified_data": verified_data,
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "start_checkpoint": str(args.start_checkpoint),
        "start_checkpoint_sha256": sha256(args.start_checkpoint),
        "optimizer_policy": "reset AdamW and cosine scheduler identically in every residual arm",
        "fixed_time_seconds": FIXED_TIME_SECONDS,
        "arms": REGISTERED_ARMS,
        "updates": UPDATES,
        "eval_interval": EVAL_INTERVAL,
        "head_updates": HEAD_UPDATES,
        "head_eval_interval": HEAD_EVAL_INTERVAL,
        "gpu": gpu,
        "environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES",) if os.environ.get(key)},
        "competition_oriented_all81_dev16_overlap": True,
        "codabench_accessed": False,
    }
    (args.out_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_root / "RUNNING").touch()

    try:
        for arm, config in REGISTERED_ARMS.items():
            command = [
                sys.executable, "-u", "-B", str(SCRIPT_DIR / "residual_multi.py"),
                "--real-root", str(args.real_root),
                "--split-manifest", str(args.split_manifest),
                "--allow-train-dev-overlap",
                "--checkpoint", str(args.base_checkpoint),
                "--resume-checkpoint", str(args.start_checkpoint),
                "--realpdebench-root", str(args.model_root),
                "--base-model", "cno",
                "--out-dir", str(args.out_root / arm),
                "--updates", str(UPDATES),
                "--eval-interval", str(EVAL_INTERVAL),
                "--batch-size", "8",
                "--test-batch-size", "32",
                "--workers", "2",
                "--lr", "0.0002",
                "--weight-decay", "0.00001",
                "--hidden", "96",
                "--blocks", "2",
                "--max-delta", "0.04",
                "--stride", "20",
                "--train-alpha", "1.0",
                "--tke", str(config["tke"]),
                "--train-window-mode", str(config["window_mode"]),
                "--bound-abs", "0.0075",
                "--bound-rel", "0.0075",
                "--fixed-time-seconds", str(FIXED_TIME_SECONDS),
                "--seed", "41",
            ]
            run(command, args.out_root / f"{arm}.log", command_log)

        cache = args.out_root / "head_cache"
        run([
            sys.executable, "-u", "-B", str(SCRIPT_DIR / "cache_frozen.py"),
            "--real-root", str(args.real_root),
            "--split-manifest", str(args.split_manifest),
            "--allow-train-dev-overlap",
            "--realpdebench-root", str(args.model_root),
            "--stride", "5",
            "--champion", str(args.start_checkpoint),
            "--out", str(cache),
        ], args.out_root / "cache.log", command_log)
        for arm, include_delta in (("H0", False), ("H1", True)):
            command = [
                sys.executable, "-u", "-B", str(SCRIPT_DIR / "train_head_fast.py"),
                "--cache", str(cache), "--hidden", "64", "--blocks", "2",
                "--loss", "logmae", "--updates", str(HEAD_UPDATES),
                "--eval-every", str(HEAD_EVAL_INTERVAL), "--batch", "16",
                "--seed", "41", "--out", str(args.out_root / arm),
            ]
            if include_delta:
                command.append("--include-delta")
            run(command, args.out_root / f"{arm}.log", command_log)

        report = build_report(args.out_root, UPDATES)
        (args.out_root / "screen_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.out_root / "DONE").touch()
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
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
