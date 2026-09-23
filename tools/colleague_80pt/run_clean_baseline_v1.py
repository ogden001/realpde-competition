#!/usr/bin/env python3
"""Run REALPDE_CLEAN_BASELINE_V1 under the frozen research protocol.

Protocol invariants:
- released real PIV only; BAD_TRAIN_FILES excluded;
- Train51 / Seen-Dev12 / unseen-AoA10 Holdout18;
- all 10-degree trajectories are holdout-only;
- Stage1 and Stage2 train on all legal stride-1 windows with deterministic
  global shuffle; Dev/Holdout use stride=20, start=0, shuffle=False;
- colleague-80 architecture and losses are preserved;
- Stage1 CNO selected on point-only dev score;
- Stage2 frozen-CNO residual h96/b2 selected on point-only dev score;
- holdout is evaluated only after all training is complete;
- no locked-final, Codabench, full-data refit, packaging, or submission access.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py

from realpde_h5_feature_adapter_train import BAD_TRAIN_FILES, h5_field, list_h5
from run_incremental_screen import require_clean_main_checkout, require_gpu, sha256

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SPLIT_MANIFEST = REPO_ROOT / "configs" / "clean_baseline_v1_split.json"
SIM_PRETRAIN_SHA256 = "82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61"
PROTOCOL = "REALPDE_CLEAN_BASELINE_V1"
SEED = 41
CNO_UPDATES = 8723
RESIDUAL_UPDATES = 38400
EVAL_INTERVAL = 1000
BATCH_SIZE = 8
PREFETCH_FACTOR = 4


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(command: list[str], log_path: Path, command_log: Path) -> None:
    with command_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "started_at": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "log": str(log_path),
        }) + "\n")
    print("RUN " + " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {command}")


def read_aoa(path: Path) -> float:
    with h5py.File(path, "r") as handle:
        return float(h5_field(handle, "aoa")[()])


def validate_split_payload(payload: dict[str, object]) -> None:
    expected_counts = {"train": 51, "dev": 12, "holdout": 18}
    if payload.get("protocol") != PROTOCOL:
        raise ValueError(f"unexpected protocol: {payload.get('protocol')}")
    groups: dict[str, list[str]] = {}
    for key, expected in expected_counts.items():
        rows = payload.get(key)
        if not isinstance(rows, list) or len(rows) != expected:
            raise ValueError(f"{key} must contain exactly {expected} trajectories")
        names = [str(row["file"] if isinstance(row, dict) else row) for row in rows]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate filenames in {key}")
        groups[key] = names
    if set(groups["train"]) & set(groups["dev"]):
        raise ValueError("train/dev overlap")
    if set(groups["train"]) & set(groups["holdout"]):
        raise ValueError("train/holdout overlap")
    if set(groups["dev"]) & set(groups["holdout"]):
        raise ValueError("dev/holdout overlap")
    if len(set(groups["train"] + groups["dev"] + groups["holdout"])) != 81:
        raise ValueError("split must cover exactly 81 unique usable trajectories")
    if BAD_TRAIN_FILES & set(groups["train"] + groups["dev"] + groups["holdout"]):
        raise ValueError("BAD_TRAIN_FILES must never appear in clean baseline split")


def verify_split_against_data(real_root: Path, payload: dict[str, object]) -> dict[str, object]:
    validate_split_payload(payload)
    groups = {
        key: [str(row["file"] if isinstance(row, dict) else row) for row in payload[key]]  # type: ignore[index]
        for key in ("train", "dev", "holdout")
    }
    usable = list_h5(real_root, BAD_TRAIN_FILES)
    usable_names = {path.name for path in usable}
    split_names = set(groups["train"] + groups["dev"] + groups["holdout"])
    if split_names != usable_names:
        missing = sorted(usable_names - split_names)
        extra = sorted(split_names - usable_names)
        raise ValueError(f"split/data mismatch missing={missing[:10]} extra={extra[:10]}")

    aoa_by_name = {name: read_aoa(real_root / name) for name in sorted(split_names)}
    train_aoa = [aoa_by_name[name] for name in groups["train"]]
    dev_aoa = [aoa_by_name[name] for name in groups["dev"]]
    holdout_aoa = [aoa_by_name[name] for name in groups["holdout"]]
    if any(abs(value - 10.0) < 1e-6 for value in train_aoa + dev_aoa):
        raise ValueError("10-degree trajectory leaked into train/dev")
    if any(abs(value - 10.0) >= 1e-6 for value in holdout_aoa):
        raise ValueError("holdout must contain only AoA=10 trajectories")
    all_10 = {name for name, value in aoa_by_name.items() if abs(value - 10.0) < 1e-6}
    if set(groups["holdout"]) != all_10:
        raise ValueError("holdout must contain every available AoA=10 trajectory")
    dev_counts = Counter(int(round(value)) for value in dev_aoa)
    if dev_counts != Counter({0: 3, 5: 3, 15: 3, 20: 3}):
        raise ValueError(f"Seen-Dev12 AoA balance mismatch: {dict(dev_counts)}")
    if set(int(round(value)) for value in train_aoa) - {0, 5, 15, 20}:
        raise ValueError("Train51 contains an unexpected AoA")
    return {
        "usable_trajectories": len(usable_names),
        "train_trajectories": len(groups["train"]),
        "dev_trajectories": len(groups["dev"]),
        "holdout_trajectories": len(groups["holdout"]),
        "dev_aoa_counts": dict(sorted(dev_counts.items())),
        "holdout_aoa": 10,
        "all_10deg_held_out": True,
        "train_dev_holdout_overlap": False,
    }


def training_manifest(payload: dict[str, object]) -> dict[str, object]:
    return {
        "protocol": PROTOCOL + "_TRAIN_DEV",
        "train": payload["train"],
        "dev": payload["dev"],
        "train_dev_overlap": False,
        "holdout_exposed_to_training": False,
    }


def holdout_eval_manifest(payload: dict[str, object]) -> dict[str, object]:
    return {
        "protocol": PROTOCOL + "_HOLDOUT_EVAL_ONLY",
        "train": payload["train"],
        "dev": payload["holdout"],
        "train_dev_overlap": False,
        "selection_allowed": False,
        "heldout_aoa": 10,
    }


def build_stage1_command(
    *,
    python: str,
    real_root: Path,
    train_manifest: Path,
    sim_pretrain: Path,
    model_root: Path,
    out_dir: Path,
    workers: int,
) -> list[str]:
    return [
        python, "-u", "-B", str(SCRIPT_DIR / "train_clean_baseline_cno.py"),
        "--real-root", str(real_root),
        "--split-manifest", str(train_manifest),
        "--init-checkpoint", str(sim_pretrain),
        "--realpdebench-root", str(model_root),
        "--out-dir", str(out_dir),
        "--updates", str(CNO_UPDATES),
        "--eval-interval", str(EVAL_INTERVAL),
        "--batch-size", str(BATCH_SIZE),
        "--test-batch-size", "32",
        "--workers", str(workers),
        "--preload-to-ram",
        "--prefetch-factor", str(PREFETCH_FACTOR),
        "--lr", "0.0001",
        "--seed", str(SEED),
    ]


def build_stage2_command(
    *,
    python: str,
    real_root: Path,
    train_manifest: Path,
    cno_checkpoint: Path,
    model_root: Path,
    out_dir: Path,
    workers: int,
) -> list[str]:
    return [
        python, "-u", "-B", str(SCRIPT_DIR / "residual_multi.py"),
        "--real-root", str(real_root),
        "--split-manifest", str(train_manifest),
        "--checkpoint", str(cno_checkpoint),
        "--realpdebench-root", str(model_root),
        "--base-model", "cno",
        "--out-dir", str(out_dir),
        "--updates", str(RESIDUAL_UPDATES),
        "--eval-interval", str(EVAL_INTERVAL),
        "--batch-size", str(BATCH_SIZE),
        "--test-batch-size", "32",
        "--workers", str(workers),
        "--preload-to-ram",
        "--prefetch-factor", str(PREFETCH_FACTOR),
        "--lr", "0.0002",
        "--weight-decay", "0.00001",
        "--hidden", "96",
        "--blocks", "2",
        "--dropout", "0",
        "--max-delta", "0.04",
        "--stride", "1",
        "--eval-stride", "20",
        "--train-window-mode", "fixed",
        "--train-alpha", "1.0",
        "--eval-alphas", "1.0",
        "--point", "1.0",
        "--mse", "0.05",
        "--tke", "0.06",
        "--temporal", "0.03",
        "--grad", "0.015",
        "--p-zero", "0.01",
        "--residual-mse", "0.25",
        "--delta-penalty", "0.02",
        "--clip-grad", "1.0",
        "--bound-abs", "0.0075",
        "--bound-rel", "0.0075",
        "--selection-metric", "point_score",
        "--seed", str(SEED),
    ]


def build_review(raw_log: Path, output: Path, command_log: Path) -> None:
    run([
        sys.executable, "-u", "-B",
        str(REPO_ROOT / "tools" / "build_training_review_log.py"),
        "--input", str(raw_log),
        "--output", str(output),
    ], output.with_suffix(output.suffix + ".builder.log"), command_log)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--sim-pretrain-checkpoint", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--runtime-profile",
        action="store_true",
        help=(
            "Run disposable b8/b16 throughput+VRAM benchmarks for Stage1 and Stage2. "
            "The current baseline remains fixed at batch=8 regardless of recommendation."
        ),
    )
    args = parser.parse_args()

    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_root}")
    for path in (args.real_root, args.sim_pretrain_checkpoint, args.model_root, SPLIT_MANIFEST):
        if not path.exists():
            raise FileNotFoundError(path)
    if sha256(args.sim_pretrain_checkpoint) != SIM_PRETRAIN_SHA256:
        raise RuntimeError("official sim_real CNO checkpoint SHA mismatch")

    execution_commit = require_clean_main_checkout(REPO_ROOT)
    gpu = require_gpu()
    split_payload = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    split_audit = verify_split_against_data(args.real_root, split_payload)

    args.out_root.mkdir(parents=True)
    (args.out_root / "RUNNING").touch()
    command_log = args.out_root / "commands.jsonl"
    train_manifest = args.out_root / "train_dev_manifest.json"
    holdout_manifest = args.out_root / "holdout_eval_manifest.json"
    write_json(train_manifest, training_manifest(split_payload))
    write_json(holdout_manifest, holdout_eval_manifest(split_payload))

    campaign_manifest = {
        "status": "REVIEW_REQUIRED",
        "protocol": PROTOCOL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_commit": execution_commit,
        "gpu": gpu,
        "canonical_split_manifest": str(SPLIT_MANIFEST),
        "canonical_split_manifest_sha256": sha256(SPLIT_MANIFEST),
        "split_audit": split_audit,
        "sim_pretrain_checkpoint": str(args.sim_pretrain_checkpoint),
        "sim_pretrain_sha256": sha256(args.sim_pretrain_checkpoint),
        "scientific_protocol": {
            "stage1_architecture_loss": "colleague80_exact",
            "stage2_architecture_loss": "colleague80_exact",
            "train_stride": 1,
            "train_sampling": "all legal stride1 windows, deterministic global shuffle, without replacement per epoch",
            "eval_stride": 20,
            "eval_start": 0,
            "seen_dev_for_checkpoint_selection": True,
            "holdout_for_checkpoint_selection": False,
            "selection_metric": "mean official-v9 Rel-L2/TKE/MVPE subscores",
            "sps_in_training_or_selection": False,
            "runtime_in_training_or_selection": False,
            "ram_preload": True,
            "persistent_workers": True,
            "prefetch_factor": PREFETCH_FACTOR,
            "runtime_profile_requested": bool(args.runtime_profile),
            "runtime_profile_can_override_baseline": False,
        },
        "budgets": {
            "stage1_cno_updates": CNO_UPDATES,
            "stage2_residual_updates": RESIDUAL_UPDATES,
            "eval_interval": EVAL_INTERVAL,
            "batch_size": BATCH_SIZE,
        },
        "locked_final_accessed": False,
        "codabench_accessed": False,
        "automatic_full_data_refit_started": False,
        "submission_packaging_started": False,
        "environment": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES",)
            if os.environ.get(key)
        },
    }
    write_json(args.out_root / "campaign_manifest.json", campaign_manifest)

    try:
        if args.runtime_profile:
            profile_root = args.out_root / "runtime_profile_stage1"
            profile_cmd = [
                sys.executable, "-u", "-B",
                str(SCRIPT_DIR / "benchmark_clean_runtime_profiles.py"),
                "--stage", "cno",
                "--real-root", str(args.real_root),
                "--split-manifest", str(train_manifest),
                "--checkpoint", str(args.sim_pretrain_checkpoint),
                "--model-root", str(args.model_root),
                "--out-root", str(profile_root),
                "--workers", str(args.workers),
            ]
            run(profile_cmd, args.out_root / "runtime_profile_stage1.log", command_log)

        stage1 = args.out_root / "stage1_cno"
        cmd = build_stage1_command(
            python=sys.executable,
            real_root=args.real_root,
            train_manifest=train_manifest,
            sim_pretrain=args.sim_pretrain_checkpoint,
            model_root=args.model_root,
            out_dir=stage1,
            workers=args.workers,
        )
        raw_log = args.out_root / "stage1_cno.log"
        run(cmd, raw_log, command_log)
        build_review(raw_log, args.out_root / "stage1_cno.train.review.log", command_log)

        if args.runtime_profile:
            profile_root = args.out_root / "runtime_profile_stage2"
            profile_cmd = [
                sys.executable, "-u", "-B",
                str(SCRIPT_DIR / "benchmark_clean_runtime_profiles.py"),
                "--stage", "residual",
                "--real-root", str(args.real_root),
                "--split-manifest", str(train_manifest),
                "--checkpoint", str(stage1 / "model_best.pth"),
                "--model-root", str(args.model_root),
                "--out-root", str(profile_root),
                "--workers", str(args.workers),
            ]
            run(profile_cmd, args.out_root / "runtime_profile_stage2.log", command_log)

        stage2 = args.out_root / "stage2_residual"
        cmd = build_stage2_command(
            python=sys.executable,
            real_root=args.real_root,
            train_manifest=train_manifest,
            cno_checkpoint=stage1 / "model_best.pth",
            model_root=args.model_root,
            out_dir=stage2,
            workers=args.workers,
        )
        raw_log = args.out_root / "stage2_residual.log"
        run(cmd, raw_log, command_log)
        build_review(raw_log, args.out_root / "stage2_residual.train.review.log", command_log)

        holdout_root = args.out_root / "holdout10_final_diagnostics"
        holdout_cmd = [
            sys.executable, "-u", "-B", str(SCRIPT_DIR / "analyze_checkpoints.py"),
            "--real-root", str(args.real_root),
            "--split-manifest", str(holdout_manifest),
            "--realpdebench-root", str(args.model_root),
            "--out-root", str(holdout_root),
            "--baseline-label", "stage2_init",
            "--batch-size", "32",
            "--workers", str(args.workers),
            "--model", f"stage2_init={stage2 / 'model_init.pth'}",
            "--model", f"stage2_best={stage2 / 'model_best.pth'}",
            "--model", f"stage2_final={stage2 / 'model_final.pth'}",
        ]
        run(holdout_cmd, args.out_root / "holdout10_final_diagnostics.log", command_log)

        evidence = {
            "status": "REVIEW_REQUIRED",
            "execution_commit": execution_commit,
            "split_manifest_sha256": sha256(SPLIT_MANIFEST),
            "stage1_best_sha256": sha256(stage1 / "model_best.pth"),
            "stage1_final_sha256": sha256(stage1 / "model_final.pth"),
            "stage2_init_sha256": sha256(stage2 / "model_init.pth"),
            "stage2_best_sha256": sha256(stage2 / "model_best.pth"),
            "stage2_final_sha256": sha256(stage2 / "model_final.pth"),
            "holdout_evaluated_only_after_stage2_training": True,
            "locked_final_accessed": False,
            "codabench_accessed": False,
            "automatic_full_data_refit_started": False,
            "submission_packaging_started": False,
        }
        write_json(args.out_root / "diagnostic_manifest.json", evidence)
        (args.out_root / "DONE").touch()
        print(json.dumps(evidence, indent=2, sort_keys=True), flush=True)
    except BaseException as error:
        write_json(args.out_root / "FAILED.json", {
            "status": "REVIEW_REQUIRED",
            "type": type(error).__name__,
            "message": str(error),
        })
        raise
    finally:
        (args.out_root / "RUNNING").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
