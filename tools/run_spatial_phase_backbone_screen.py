#!/usr/bin/env python3
"""One-shot clean screen of spatial-phase augmentation in the original Stage-1 CNO backbone.

This runner deliberately reuses the existing clean-baseline Stage-1 trainer.
The historical clean baseline is the frozen control. The only scientific
change in the candidate is the 64x128 -> 32x64 spatial sampling phase used for
training windows. Seen-Dev remains the historical P00 sampling phase.

No residual training, holdout evaluation, Codabench access, full-data refit,
or submission work exists in this runner.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
COLLEAGUE = SCRIPT_DIR / "colleague_80pt"
sys.path.insert(0, str(COLLEAGUE))

from run_clean_baseline_v1 import (  # noqa: E402
    SIM_PRETRAIN_SHA256,
    SPLIT_MANIFEST,
    training_manifest,
    validate_split_payload,
)
from run_incremental_screen import require_gpu, sha256  # noqa: E402

PROTOCOL = "REALPDE_SPATIAL_PHASE_BACKBONE_SCREEN_V1"
STATUS = "REVIEW_REQUIRED"

SEED = 41
SPATIAL_PHASE_SEED = 20260925
UPDATES = 8723
EVAL_INTERVAL = 1000
BATCH_SIZE = 8
LR = 1e-4
SPATIAL_PHASE_MIX_PROB = 0.5

# Frozen historical control from:
# docs/clean_baseline_v1/results/20260924_run1/RUN_REPORT.md
BASELINE_EXECUTION_COMMIT = "65e4f0f1d029eea47c6ba96364ce889e22d437d8"
BASELINE = {
    8000: {
        "rel_l2_raw": 0.099606201,
        "tke_raw": 0.778031290,
        "mvpe_raw": 0.080289602,
    },
    8723: {
        "rel_l2_raw": 0.099932298,
        "tke_raw": 0.792903721,
        "mvpe_raw": 0.080299616,
    },
}

GATE_MEAN_DELTA_PCT = -0.50
GATE_MAX_SINGLE_DEGRADE_PCT = 1.00


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}; log={log_path}"
        )


def score_error(error: float) -> float:
    return 100.0 / (1.0 + 0.5 * max(float(error), 0.0))


def point_score(metrics: dict[str, float]) -> float:
    return sum(
        score_error(float(metrics[key]))
        for key in ("rel_l2_raw", "tke_raw", "mvpe_raw")
    ) / 3.0


def pct(candidate: float, control: float) -> float:
    return 100.0 * (float(candidate) - float(control)) / max(abs(float(control)), 1e-12)


def raw_metrics(row: dict[str, object]) -> dict[str, float]:
    return {
        key: float(row[key])
        for key in ("rel_l2_raw", "tke_raw", "mvpe_raw")
    }


def compare(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, object]:
    delta = {key: pct(candidate[key], baseline[key]) for key in baseline}
    mean_delta = sum(delta.values()) / 3.0
    return {
        "delta_pct": delta,
        "mean_delta_pct": mean_delta,
        "baseline_point_score": point_score(baseline),
        "candidate_point_score": point_score(candidate),
        "point_score_delta": point_score(candidate) - point_score(baseline),
    }


def decision(candidate_8000: dict[str, float]) -> dict[str, object]:
    baseline = BASELINE[8000]
    comparison = compare(candidate_8000, baseline)
    delta = comparison["delta_pct"]
    assert isinstance(delta, dict)
    checks = {
        "mean_raw_error_improves_ge_0p5pct": float(comparison["mean_delta_pct"]) <= GATE_MEAN_DELTA_PCT,
        "at_least_two_metrics_improve": sum(float(v) < 0.0 for v in delta.values()) >= 2,
        "no_metric_degrades_gt_1pct": max(float(v) for v in delta.values()) <= GATE_MAX_SINGLE_DEGRADE_PCT,
        "point_score_improves": float(comparison["point_score_delta"]) > 0.0,
    }
    return {
        **comparison,
        "checks": checks,
        "status": "GO" if all(checks.values()) else "NO_GO",
    }


def build_candidate_command(
    args: argparse.Namespace,
    *,
    train_manifest: Path,
    out_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-B",
        str(COLLEAGUE / "train_clean_baseline_cno.py"),
        "--real-root",
        str(args.real_root),
        "--split-manifest",
        str(train_manifest),
        "--init-checkpoint",
        str(args.sim_pretrain),
        "--realpdebench-root",
        str(args.kit_root),
        "--out-dir",
        str(out_dir),
        "--updates",
        str(UPDATES),
        "--eval-interval",
        str(EVAL_INTERVAL),
        "--batch-size",
        str(BATCH_SIZE),
        "--test-batch-size",
        "32",
        "--workers",
        str(args.workers),
        "--preload-to-ram",
        "--prefetch-factor",
        "4",
        "--lr",
        str(LR),
        "--seed",
        str(SEED),
        "--spatial-phase-mix-prob",
        str(SPATIAL_PHASE_MIX_PROB),
        "--spatial-phase-seed",
        str(SPATIAL_PHASE_SEED),
        "--protocol-label",
        PROTOCOL,
    ]


def prepare(args: argparse.Namespace) -> None:
    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_root}")
    for path in (args.real_root, args.sim_pretrain, args.kit_root, SPLIT_MANIFEST):
        if not path.exists():
            raise FileNotFoundError(path)
    if sha256(args.sim_pretrain) != SIM_PRETRAIN_SHA256:
        raise RuntimeError("official sim_real CNO checkpoint SHA mismatch")

    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    validate_split_payload(split)
    manifest = training_manifest(split)

    train_names = [
        row["file"] if isinstance(row, dict) else row
        for row in manifest["train"]
    ]
    dev_names = [
        row["file"] if isinstance(row, dict) else row
        for row in manifest["dev"]
    ]
    missing = [
        str(name)
        for name in train_names + dev_names
        if not (args.real_root / str(name)).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing train/dev files: {missing[:10]}")

    args.out_root.mkdir(parents=True)
    dump(args.out_root / "train_dev_manifest.json", manifest)
    dump(
        args.out_root / "preflight.json",
        {
            "status": STATUS,
            "protocol": PROTOCOL,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "execution_commit": git_head(),
            "gpu": require_gpu(),
            "sim_pretrain_sha256": sha256(args.sim_pretrain),
            "canonical_split_manifest": str(SPLIT_MANIFEST),
            "canonical_split_manifest_sha256": sha256(SPLIT_MANIFEST),
            "train_trajectories": len(train_names),
            "seen_dev_trajectories": len(dev_names),
            "historical_control_execution_commit": BASELINE_EXECUTION_COMMIT,
            "updates": UPDATES,
            "eval_interval": EVAL_INTERVAL,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "seed": SEED,
            "candidate_spatial_phase_mix_prob": SPATIAL_PHASE_MIX_PROB,
            "candidate_spatial_phase_seed": SPATIAL_PHASE_SEED,
            "candidate_target_distribution": {
                "P00": 0.5,
                "P01": 1.0 / 6.0,
                "P10": 1.0 / 6.0,
                "P11": 1.0 / 6.0,
            },
            "validation_phase": "P00",
            "residual_training_started": False,
            "holdout_accessed": False,
            "locked_final_accessed": False,
            "codabench_accessed": False,
            "full_data_refit_started": False,
            "submission_packaging_started": False,
        },
    )


def run_candidate(args: argparse.Namespace) -> None:
    root = args.out_root / "candidate_backbone"
    if root.exists():
        raise FileExistsError(root)
    command = build_candidate_command(
        args,
        train_manifest=args.out_root / "train_dev_manifest.json",
        out_dir=root,
    )
    run(command, args.out_root / "candidate_backbone.log")
    for required in (
        root / "model_final.pth",
        root / "summary.json",
        root / "run_config.json",
        root / "sampling_audit.json",
        root / "eval_step_08000" / "metrics.json",
        root / "eval_step_08723" / "metrics.json",
    ):
        if not required.exists():
            raise FileNotFoundError(f"candidate output missing: {required}")
    (root / "DONE").touch()


def review(args: argparse.Namespace) -> None:
    root = args.out_root / "candidate_backbone"
    if not (root / "DONE").is_file():
        raise FileNotFoundError("candidate backbone is not complete")

    metrics_8000 = raw_metrics(
        json.loads((root / "eval_step_08000" / "metrics.json").read_text(encoding="utf-8"))
    )
    metrics_8723 = raw_metrics(
        json.loads((root / "eval_step_08723" / "metrics.json").read_text(encoding="utf-8"))
    )
    run_config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    phase = run_config["spatial_phase_augmentation"]

    expected = {
        "updates": UPDATES,
        "eval_interval": EVAL_INTERVAL,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "seed": SEED,
    }
    for key, value in expected.items():
        if run_config.get(key) != value:
            raise RuntimeError(f"candidate config mismatch for {key}: {run_config.get(key)} != {value}")
    if abs(float(phase["non_p00_probability"]) - SPATIAL_PHASE_MIX_PROB) > 1e-12:
        raise RuntimeError("candidate spatial phase mix probability mismatch")
    if phase.get("validation_phase") != "P00":
        raise RuntimeError("candidate validation phase must remain P00")

    primary = decision(metrics_8000)
    final_comparison = compare(metrics_8723, BASELINE[8723])
    training_summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))

    summary = {
        "status": STATUS,
        "protocol": PROTOCOL,
        "historical_control": {
            "source": "docs/clean_baseline_v1/results/20260924_run1/RUN_REPORT.md",
            "execution_commit": BASELINE_EXECUTION_COMMIT,
            "metrics": BASELINE,
        },
        "candidate_at_8000": metrics_8000,
        "candidate_at_8723": metrics_8723,
        "primary_matched_8000": primary,
        "matched_final_8723": final_comparison,
        "candidate_training_summary": training_summary,
        "candidate_spatial_phase": phase,
        "gate": primary["status"],
        "gate_uses_candidate_best_checkpoint": False,
        "gate_uses_matched_update": 8000,
        "residual_training_started": False,
        "holdout_accessed": False,
        "locked_final_accessed": False,
        "codabench_accessed": False,
        "full_data_refit_started": False,
        "submission_packaging_started": False,
        "automatic_followup_started": False,
    }
    dump(args.out_root / "summary.json", summary)

    lines = [
        "# REALPDE Spatial Phase Backbone Screen V1",
        "",
        "Status: `REVIEW_REQUIRED`",
        "",
        "Historical Clean Baseline Stage-1 is the frozen control. The candidate",
        "reuses the original Stage-1 trainer and changes only training spatial sampling phase.",
        "",
        f"- Gate: **{primary['status']}**",
        f"- Baseline @8000: `{BASELINE[8000]}`",
        f"- Candidate @8000: `{metrics_8000}`",
        f"- @8000 relative deltas: `{primary['delta_pct']}`",
        f"- @8000 mean raw-error delta: `{float(primary['mean_delta_pct']):.4f}%`",
        f"- @8000 point-score delta: `{float(primary['point_score_delta']):.6f}`",
        f"- Baseline @8723: `{BASELINE[8723]}`",
        f"- Candidate @8723: `{metrics_8723}`",
        f"- @8723 relative deltas: `{final_comparison['delta_pct']}`",
        f"- Candidate assigned spatial phase: `{phase.get('assigned_legal_window_fractions')}`",
        f"- Candidate best iteration (diagnostic only): `{training_summary.get('best_iteration')}`",
        "",
        "Decision is based on matched update 8000, not best-vs-best checkpoint selection.",
        "STOP here. No residual or follow-up training is authorized.",
        "",
    ]
    (args.out_root / "SPATIAL_PHASE_BACKBONE_SCREEN_REVIEW.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (args.out_root / "DONE").touch()


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--sim-pretrain", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "candidate", "review", "run-all"):
        add_common(sub.add_parser(name))
    args = parser.parse_args()

    if args.cmd == "preflight":
        prepare(args)
        return
    if args.cmd == "run-all":
        prepare(args)
        run_candidate(args)
        review(args)
        return
    if not args.out_root.exists():
        raise FileNotFoundError("run preflight first")
    if args.cmd == "candidate":
        run_candidate(args)
    elif args.cmd == "review":
        review(args)


if __name__ == "__main__":
    main()
