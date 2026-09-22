#!/usr/bin/env python3
"""Unified three-arm campaign on top of the colleague 80-point baseline.

Arm A: 12k conflict-aware Pareto-TKE continuation from the mature residual.
Arm B: fresh h96/b2 colleague residual on the frozen SOTA-V2 P0-A/MF backbone.
Arm C: existing 20k adjacent-AoA mean-field continuation.

The arms are independent. No arm inherits another arm's trained weights.
No combo/full merge/package/Codabench/locked-final path is implemented here.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from batch_profiles import ARM_PROFILES, TrainProfile
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

BASELINE = {
    "rel_l2_raw": 0.08042041957378387,
    "tke_raw": 0.449158251285553,
    "mvpe_raw": 0.07111691683530807,
}

ARM_A_UPDATES = 12_000
ARM_A_EVAL = 2_000
ARM_B_UPDATES = 38_400
ARM_B_EVAL = 4_800
ARM_C_UPDATES = 20_000
EXPECTED_STRONG_BACKBONE_SHA256 = (
    "f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce8"
)


def load_training_profiles(path: Path | None) -> dict[str, TrainProfile]:
    selected_names = {
        "A_pareto_tke": "b8",
        "B_strong_backbone": "b8",
        "C_aoa_meanfield": "b8",
    }
    if path is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_arms = payload.get("arms")
        if not isinstance(raw_arms, dict):
            raise ValueError("profile JSON missing arms")
        for arm in selected_names:
            raw = raw_arms.get(arm)
            if not isinstance(raw, dict):
                raise ValueError(f"profile JSON missing {arm}")
            name = str(raw.get("name", raw.get("selected", "")))
            if name not in ("b8", "b16"):
                raise ValueError(f"{arm}: invalid frozen profile {name!r}")
            expected = ARM_PROFILES[arm][name]
            for key, expected_value in expected.to_dict().items():
                if key in raw and raw[key] != expected_value:
                    raise ValueError(
                        f"{arm}: profile field {key}={raw[key]!r} "
                        f"!= frozen {expected_value!r}"
                    )
            selected_names[arm] = name
    return {
        arm: ARM_PROFILES[arm][name]
        for arm, name in selected_names.items()
    }


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
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


def pct_delta(value: float, baseline: float) -> float:
    return 100.0 * (float(value) - float(baseline)) / max(abs(float(baseline)), 1e-12)


def load_metrics(path: Path) -> dict[str, float]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: float(value[key])
        for key in ("rel_l2_raw", "tke_raw", "mvpe_raw")
    }


def compare_to_current80(metrics: dict[str, float]) -> dict[str, float]:
    result = dict(metrics)
    for key, baseline in BASELINE.items():
        result[f"delta_{key}_pct_vs_current80"] = pct_delta(
            metrics[key], baseline
        )
    return result


def pareto_mechanical_gate(metrics: dict[str, float]) -> dict[str, object]:
    rel_deg = pct_delta(metrics["rel_l2_raw"], BASELINE["rel_l2_raw"])
    tke_delta = pct_delta(metrics["tke_raw"], BASELINE["tke_raw"])
    mvpe_deg = pct_delta(metrics["mvpe_raw"], BASELINE["mvpe_raw"])
    checks = {
        "tke_improvement_ge_3pct": tke_delta <= -3.0,
        "rel_degradation_le_0p5pct": rel_deg <= 0.5,
        "mvpe_degradation_le_0p3pct": mvpe_deg <= 0.3,
    }
    return {"pass": all(checks.values()), "checks": checks}


def backbone_mechanical_gate(metrics: dict[str, float]) -> dict[str, object]:
    rel_deg = pct_delta(metrics["rel_l2_raw"], BASELINE["rel_l2_raw"])
    tke_delta = pct_delta(metrics["tke_raw"], BASELINE["tke_raw"])
    mvpe_deg = pct_delta(metrics["mvpe_raw"], BASELINE["mvpe_raw"])
    checks = {
        "tke_improvement_ge_3pct": tke_delta <= -3.0,
        "rel_degradation_le_1pct": rel_deg <= 1.0,
        "mvpe_degradation_le_1pct": mvpe_deg <= 1.0,
    }
    return {"pass": all(checks.values()), "checks": checks}


def residual_common(
    args: argparse.Namespace,
    *,
    base_checkpoint: Path,
    model_root: Path,
    out_dir: Path,
    updates: int,
    eval_interval: int,
    base_model: str,
    batch_size: int,
    lr: float,
) -> list[str]:
    return [
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
        str(updates),
        "--eval-interval",
        str(eval_interval),
        "--batch-size",
        str(batch_size),
        "--test-batch-size",
        "32",
        "--workers",
        str(args.workers),
        "--lr",
        str(lr),
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
        "--seed",
        "41",
    ]


def run_campaign(args: argparse.Namespace) -> dict[str, object]:
    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_root}")
    required = (
        args.real_root,
        args.split_manifest,
        args.data_manifest,
        args.colleague_base_checkpoint,
        args.colleague_residual_checkpoint,
        args.colleague_model_root,
        args.strong_backbone_checkpoint,
        args.strong_kit_root,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    if sha256(args.colleague_base_checkpoint) != EXPECTED_BASE_SHA256:
        raise RuntimeError("colleague Stage-1 CNO checkpoint SHA-256 mismatch")
    if sha256(args.colleague_residual_checkpoint) != EXPECTED_START_SHA256:
        raise RuntimeError("colleague 80-point residual checkpoint SHA-256 mismatch")
    if sha256(args.strong_backbone_checkpoint) != EXPECTED_STRONG_BACKBONE_SHA256:
        raise RuntimeError("SOTA-V2 full@53582 checkpoint SHA-256 mismatch")

    execution_commit = require_clean_main_checkout(REPO_ROOT)
    gpu = require_gpu()
    profiles = load_training_profiles(args.profile_json)
    verified_data = verify_allowed_data(
        args.real_root,
        args.data_manifest,
        args.split_manifest,
    )

    args.out_root.mkdir(parents=True)
    command_log = args.out_root / "commands.jsonl"
    dump(
        args.out_root / "campaign_manifest.json",
        {
            "campaign": "colleague80_v2_three_arm_campaign_20260922",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "execution_commit": execution_commit,
            "gpu": gpu,
            "verified_data": verified_data,
            "checkpoints": {
                "colleague_base": str(args.colleague_base_checkpoint),
                "colleague_base_sha256": sha256(args.colleague_base_checkpoint),
                "colleague_residual": str(args.colleague_residual_checkpoint),
                "colleague_residual_sha256": sha256(
                    args.colleague_residual_checkpoint
                ),
                "strong_backbone": str(args.strong_backbone_checkpoint),
                "strong_backbone_sha256": sha256(args.strong_backbone_checkpoint),
            },
            "batch_profile_source": (
                str(args.profile_json) if args.profile_json is not None else "default_b8"
            ),
            "arms": {
                "A_pareto_tke": {
                    **profiles["A_pareto_tke"].to_dict(),
                    "start": "mature colleague 80pt residual",
                    "tke_weight": 0.12,
                    "gradient_mode": "project_tke",
                    "primary_gradient_modified": False,
                },
                "B_strong_backbone": {
                    **profiles["B_strong_backbone"].to_dict(),
                    "start": "fresh zero-init h96/b2 residual",
                    "base": "frozen SOTA-V2 P0-A/MF checkpoint",
                    "loss": "original colleague residual scalar objective",
                },
                "C_aoa_meanfield": {
                    **profiles["C_aoa_meanfield"].to_dict(),
                    "runner": "run_aoa_meanfield_long_screen.py",
                    "start": "mature colleague 80pt residual",
                },
            },
            "automatic_combo": False,
            "automatic_full_merge": False,
            "locked_final_accessed": False,
            "codabench_accessed": False,
        },
    )

    arm_a = args.out_root / "A_pareto_tke"
    profile_a = profiles["A_pareto_tke"]
    cmd_a = residual_common(
        args,
        base_checkpoint=args.colleague_base_checkpoint,
        model_root=args.colleague_model_root,
        out_dir=arm_a,
        updates=profile_a.updates,
        eval_interval=profile_a.eval_interval,
        base_model="cno",
        batch_size=profile_a.batch_size,
        lr=profile_a.lr,
    )
    cmd_a += [
        "--resume-checkpoint",
        str(args.colleague_residual_checkpoint),
        "--tke",
        "0.12",
        "--gradient-mode",
        "project_tke",
        "--fixed-time-seconds",
        str(FIXED_TIME_SECONDS),
    ]
    run(cmd_a, args.out_root / "A_pareto_tke.log", command_log)
    run(
        [
            sys.executable,
            "-u",
            "-B",
            str(REPO_ROOT / "tools" / "build_training_review_log.py"),
            "--input",
            str(args.out_root / "A_pareto_tke.log"),
            "--output",
            str(args.out_root / "A_pareto_tke.train.review.log"),
        ],
        args.out_root / "A_review_builder.log",
        command_log,
    )

    arm_b = args.out_root / "B_strong_backbone"
    profile_b = profiles["B_strong_backbone"]
    cmd_b = residual_common(
        args,
        base_checkpoint=args.strong_backbone_checkpoint,
        model_root=args.strong_kit_root,
        out_dir=arm_b,
        updates=profile_b.updates,
        eval_interval=profile_b.eval_interval,
        base_model="sota_v2_mf",
        batch_size=profile_b.batch_size,
        lr=profile_b.lr,
    )
    cmd_b += ["--tke", "0.06", "--gradient-mode", "scalar"]
    run(cmd_b, args.out_root / "B_strong_backbone.log", command_log)
    run(
        [
            sys.executable,
            "-u",
            "-B",
            str(REPO_ROOT / "tools" / "build_training_review_log.py"),
            "--input",
            str(args.out_root / "B_strong_backbone.log"),
            "--output",
            str(args.out_root / "B_strong_backbone.train.review.log"),
        ],
        args.out_root / "B_review_builder.log",
        command_log,
    )

    arm_c = args.out_root / "C_aoa_meanfield"
    profile_c = profiles["C_aoa_meanfield"]
    if profile_c.matched_control_step is None:
        raise ValueError("AoA profile missing matched_control_step")
    run(
        [
            sys.executable,
            "-u",
            "-B",
            str(SCRIPT_DIR / "run_aoa_meanfield_long_screen.py"),
            "--real-root",
            str(args.real_root),
            "--base-checkpoint",
            str(args.colleague_base_checkpoint),
            "--start-checkpoint",
            str(args.colleague_residual_checkpoint),
            "--model-root",
            str(args.colleague_model_root),
            "--data-manifest",
            str(args.data_manifest),
            "--split-manifest",
            str(args.split_manifest),
            "--out-root",
            str(arm_c),
            "--workers",
            str(args.workers),
            "--updates",
            str(profile_c.updates),
            "--eval-interval",
            str(profile_c.eval_interval),
            "--matched-control-step",
            str(profile_c.matched_control_step),
            "--batch-size",
            str(profile_c.batch_size),
            "--lr",
            str(profile_c.lr),
        ],
        args.out_root / "C_aoa_meanfield.wrapper.log",
        command_log,
    )

    metrics_a = load_metrics(arm_a / "final_primary_metrics.json")
    metrics_b = load_metrics(arm_b / "final_primary_metrics.json")
    metrics_c = load_metrics(
        arm_c / "A1_aoa_meanfield" / "final_primary_metrics.json"
    )
    summary = {
        "status": "REVIEW_REQUIRED",
        "execution_commit": execution_commit,
        "baseline_current80": BASELINE,
        "training_profiles": {
            arm: profile.to_dict() for arm, profile in profiles.items()
        },
        "A_pareto_tke": {
            **compare_to_current80(metrics_a),
            "mechanical_gate": pareto_mechanical_gate(metrics_a),
            "final_checkpoint": str(arm_a / "model_final.pth"),
            "best_checkpoint": str(arm_a / "model_best.pth"),
        },
        "B_strong_backbone": {
            **compare_to_current80(metrics_b),
            "mechanical_gate": backbone_mechanical_gate(metrics_b),
            "final_checkpoint": str(arm_b / "model_final.pth"),
            "best_checkpoint": str(arm_b / "model_best.pth"),
            "base_vs_final": json.loads(
                (arm_b / "final_primary_metrics.json").read_text(
                    encoding="utf-8"
                )
            ),
        },
        "C_aoa_meanfield": {
            **compare_to_current80(metrics_c),
            "run_root": str(arm_c),
        },
        "automatic_combo_started": False,
        "automatic_full_merge_started": False,
        "locked_final_accessed": False,
        "codabench_accessed": False,
    }
    dump(args.out_root / "summary.json", summary)
    (args.out_root / "DONE").touch()
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument(
        "--colleague-base-checkpoint", type=Path, required=True
    )
    parser.add_argument(
        "--colleague-residual-checkpoint", type=Path, required=True
    )
    parser.add_argument("--colleague-model-root", type=Path, required=True)
    parser.add_argument(
        "--strong-backbone-checkpoint", type=Path, required=True
    )
    parser.add_argument("--strong-kit-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--profile-json",
        type=Path,
        default=None,
        help=(
            "selected_profiles.json from benchmark_v2_batch_profiles.py. "
            "If omitted, the original b8 protocol is used."
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run_campaign(args)


if __name__ == "__main__":
    main()
