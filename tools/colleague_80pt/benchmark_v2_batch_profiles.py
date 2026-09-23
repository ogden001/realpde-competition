#!/usr/bin/env python3
"""Benchmark b8 vs b16 on the actual three colleague80 V2 training paths.

Exactly 200 training updates are timed for each arm/profile. Selection uses only
throughput and VRAM headroom, never validation metrics.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from batch_profiles import ARM_PROFILES, select_profile
from run_incremental_screen import (
    EXPECTED_BASE_SHA256,
    EXPECTED_START_SHA256,
    require_clean_main_checkout,
    require_gpu,
    sha256,
    verify_allowed_data,
)
from run_v2_campaign import APPROVED_STRONG_BACKBONE_SHA256

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
BENCHMARK_UPDATES = 200
ARM_NAMES = (
    "A_pareto_tke",
    "B_strong_backbone",
    "C_aoa_meanfield",
)


def parse_arms(value: str | None) -> tuple[str, ...]:
    """Parse a bounded comma-separated arm selection for independent runs."""
    if value is None or not value.strip():
        return ARM_NAMES
    arms = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(arms) - set(ARM_NAMES))
    if unknown:
        raise ValueError(f"unknown arm(s): {', '.join(unknown)}")
    if not arms:
        raise ValueError("at least one arm is required")
    if len(set(arms)) != len(arms):
        raise ValueError("duplicate arm selection")
    return arms


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def common_command(
    args: argparse.Namespace,
    *,
    arm: str,
    profile_name: str,
    out_dir: Path,
) -> list[str]:
    profile = ARM_PROFILES[arm][profile_name]
    base_checkpoint = (
        args.strong_backbone_checkpoint
        if arm == "B_strong_backbone"
        else args.colleague_base_checkpoint
    )
    model_root = (
        args.strong_kit_root
        if arm == "B_strong_backbone"
        else args.colleague_model_root
    )
    base_model = "sota_v2_mf" if arm == "B_strong_backbone" else "cno"
    command = [
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
        str(base_checkpoint),
        "--realpdebench-root",
        str(model_root),
        "--base-model",
        base_model,
        "--out-dir",
        str(out_dir),
        "--updates",
        str(BENCHMARK_UPDATES),
        "--eval-interval",
        str(BENCHMARK_UPDATES),
        "--batch-size",
        str(profile.batch_size),
        "--test-batch-size",
        "16",
        "--workers",
        str(args.workers),
        "--lr",
        str(profile.lr),
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
        "--train-window-mode",
        "fixed",
        "--train-on-all",
        "--bound-abs",
        "0.0075",
        "--bound-rel",
        "0.0075",
        "--max-eval-batches",
        "1",
        "--benchmark-mode",
        "--seed",
        "41",
    ]
    if arm == "A_pareto_tke":
        command += [
            "--resume-checkpoint",
            str(args.colleague_residual_checkpoint),
            "--tke",
            "0.12",
            "--gradient-mode",
            "project_tke",
        ]
    elif arm == "B_strong_backbone":
        command += ["--tke", "0.06", "--gradient-mode", "scalar"]
    elif arm == "C_aoa_meanfield":
        command += [
            "--resume-checkpoint",
            str(args.colleague_residual_checkpoint),
            "--tke",
            "0.06",
            "--gradient-mode",
            "scalar",
            "--aoa-meanfield-aug-prob",
            "0.5",
            "--aoa-meanfield-lambda-min",
            "0.2",
            "--aoa-meanfield-lambda-max",
            "0.5",
            "--aoa-neighbor-max-gap-deg",
            "5.1",
            "--aoa-min-eligible-fraction",
            "0.8",
        ]
    else:
        raise KeyError(arm)
    return command


def run_one(
    args: argparse.Namespace,
    *,
    arm: str,
    profile_name: str,
    out_dir: Path,
) -> dict[str, object]:
    command = common_command(
        args,
        arm=arm,
        profile_name=profile_name,
        out_dir=out_dir,
    )
    log_path = out_dir.parent / f"{profile_name}.log"
    record = {
        "arm": arm,
        "profile": profile_name,
        "command": command,
        "log": str(log_path),
    }
    print("BENCH " + " ".join(command), flush=True)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    record["returncode"] = completed.returncode
    runtime_path = out_dir / "runtime.json"
    runtime_ok = completed.returncode == 0 and runtime_path.is_file()
    record["success"] = runtime_ok
    if runtime_ok:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        record.update(runtime)
    else:
        record["runtime_missing"] = not runtime_path.is_file()
    return record


def preflight(
    args: argparse.Namespace,
    arms: tuple[str, ...],
) -> dict[str, object]:
    required_paths = [
        args.real_root,
        args.split_manifest,
        args.data_manifest,
        args.colleague_base_checkpoint,
        args.colleague_residual_checkpoint,
        args.colleague_model_root,
    ]
    if "B_strong_backbone" in arms:
        required_paths.extend(
            [args.strong_backbone_checkpoint, args.strong_kit_root]
        )
    for path in required_paths:
        if path is None:
            raise FileNotFoundError("missing required path for selected arms")
        if not path.exists():
            raise FileNotFoundError(path)
    if sha256(args.colleague_base_checkpoint) != EXPECTED_BASE_SHA256:
        raise RuntimeError("colleague base checkpoint SHA mismatch")
    if sha256(args.colleague_residual_checkpoint) != EXPECTED_START_SHA256:
        raise RuntimeError("colleague residual checkpoint SHA mismatch")
    if "B_strong_backbone" in arms:
        strong_sha = sha256(args.strong_backbone_checkpoint)
        if strong_sha not in APPROVED_STRONG_BACKBONE_SHA256:
            raise RuntimeError("strong SOTA-V2 checkpoint SHA is not approved")

    return {
        "execution_commit": require_clean_main_checkout(REPO_ROOT),
        "gpu": require_gpu(),
        "verified_data": verify_allowed_data(
            args.real_root,
            args.data_manifest,
            args.split_manifest,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--colleague-base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--colleague-residual-checkpoint", type=Path, required=True
    )
    parser.add_argument("--colleague-model-root", type=Path, required=True)
    parser.add_argument(
        "--strong-backbone-checkpoint", type=Path, default=None
    )
    parser.add_argument("--strong-kit-root", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--arms",
        default=None,
        help="comma-separated subset of A_pareto_tke,B_strong_backbone,C_aoa_meanfield",
    )
    args = parser.parse_args()

    if args.out_root.exists():
        raise FileExistsError(args.out_root)
    arms = parse_arms(args.arms)
    meta = preflight(args, arms)
    args.out_root.mkdir(parents=True)

    results: dict[str, object] = {
        "benchmark": "colleague80_v2_batch_profile",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_updates": BENCHMARK_UPDATES,
        "arms_requested": list(arms),
        "selection_uses_validation_metrics": False,
        **meta,
        "arms": {},
    }
    selected: dict[str, object] = {
        "version": 1,
        "selection_rule": {
            "b16_min_throughput_gain": 1.20,
            "b16_max_peak_allocated_fraction": 0.92,
            "b16_max_peak_reserved_fraction": 0.96,
            "validation_metrics_used": False,
        },
        "arms": {},
    }

    for arm in arms:
        arm_dir = args.out_root / arm
        b8 = run_one(
            args,
            arm=arm,
            profile_name="b8",
            out_dir=arm_dir / "b8",
        )
        if not bool(b8["success"]):
            dump(args.out_root / "benchmark_results.json", results)
            raise RuntimeError(f"{arm} b8 reference benchmark failed")

        b16 = run_one(
            args,
            arm=arm,
            profile_name="b16",
            out_dir=arm_dir / "b16",
        )
        decision = select_profile(b8, b16)
        chosen = str(decision["selected"])
        chosen_profile = ARM_PROFILES[arm][chosen]
        results["arms"][arm] = {
            "b8": b8,
            "b16": b16,
            "decision": decision,
        }
        selected["arms"][arm] = {
            **chosen_profile.to_dict(),
            "decision": decision,
        }
        dump(args.out_root / "benchmark_results.json", results)
        dump(args.out_root / "selected_profiles.json", selected)

    print(json.dumps(selected, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
