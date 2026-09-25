#!/usr/bin/env python3
"""Matched one-shot gate for 2x spatial sampling phase augmentation.

The gate changes only the training observation phase. Both arms:
- start from the same selected Strong Backbone + residual checkpoint;
- use the same Train51 temporal windows, order, optimizer reset, budget and loss;
- evaluate only on official P00 Seen-Dev12;
- never read AoA10 holdout, locked-final/private data, Codabench, or submission code.

Candidate exposure is frozen to 50% P00 and 50% balanced P01/P10/P11.
All outputs are REVIEW_REQUIRED. No automatic follow-up experiment exists here.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
COLLEAGUE = TOOLS / "colleague_80pt"
sys.path.insert(0, str(COLLEAGUE))

from run_clean_baseline_v1 import (  # noqa: E402
    SPLIT_MANIFEST,
    training_manifest,
    validate_split_payload,
)
from run_incremental_screen import require_clean_main_checkout, require_gpu, sha256  # noqa: E402

PROTOCOL = "REALPDE_SPATIAL_PHASE_GATE_V1"
STATUS = "REVIEW_REQUIRED"
SEED = 41
SPATIAL_PHASE_SEED = 20260925
UPDATES = 5_000
EVAL_INTERVAL = 1_000
BATCH_SIZE = 8
LR = 1e-5
CANDIDATE_MIX_PROB = 0.5

STRONG_BACKBONE_SHA256 = "d340effd68031aba1cd2bc610c676878fb9105dd7f81b9ce15767280e665a5c0"
STRONG_RESIDUAL_SHA256 = "d2b4ddd0df6064d053ec918c2671a93b7d9c2a40a15afe854bb3a5badf1438fc"

GATE_MEAN_DELTA_PCT = -0.25
GATE_MAX_SINGLE_DEGRADE_PCT = 0.50


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run(cmd: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(cmd)}; log={log}")


def pct(candidate: float, control: float) -> float:
    return 100.0 * (float(candidate) - float(control)) / max(abs(float(control)), 1e-12)


def score_error(error: float) -> float:
    return 100.0 / (1.0 + 0.5 * max(float(error), 0.0))


def point_score(metrics: dict[str, float]) -> float:
    return sum(score_error(metrics[k]) for k in ("rel_l2_raw", "tke_raw", "mvpe_raw")) / 3.0


def spatial_gate(candidate: dict[str, float], control: dict[str, float]) -> dict[str, object]:
    delta = {key: pct(candidate[key], control[key]) for key in control}
    mean_delta = sum(delta.values()) / 3.0
    checks = {
        "mean_raw_error_improves_ge_0p25pct": mean_delta <= GATE_MEAN_DELTA_PCT,
        "at_least_two_metrics_improve": sum(value < 0.0 for value in delta.values()) >= 2,
        "no_metric_degrades_gt_0p5pct": max(delta.values()) <= GATE_MAX_SINGLE_DEGRADE_PCT,
        "point_score_improves": point_score(candidate) > point_score(control),
    }
    return {
        "status": "GO" if all(checks.values()) else "NO_GO",
        "delta_pct": delta,
        "mean_delta_pct": mean_delta,
        "control_point_score": point_score(control),
        "candidate_point_score": point_score(candidate),
        "point_score_delta": point_score(candidate) - point_score(control),
        "checks": checks,
    }


def build_arm_command(args: argparse.Namespace, out_dir: Path, *, spatial_mix_prob: float) -> list[str]:
    return [
        sys.executable, "-u", "-B", str(COLLEAGUE / "residual_multi.py"),
        "--real-root", str(args.real_root),
        "--split-manifest", str(args.out_root / "train_dev_manifest.json"),
        "--checkpoint", str(args.strong_backbone),
        "--resume-checkpoint", str(args.strong_residual),
        "--realpdebench-root", str(args.kit_root),
        "--base-model", "sota_v2_mf",
        "--out-dir", str(out_dir),
        "--updates", str(UPDATES),
        "--eval-interval", str(EVAL_INTERVAL),
        "--batch-size", str(BATCH_SIZE),
        "--test-batch-size", "32",
        "--workers", str(args.workers),
        "--preload-to-ram",
        "--prefetch-factor", "4",
        "--lr", str(LR),
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
        "--selection-metric", "point_score",
        "--gradient-mode", "scalar",
        "--seed", str(SEED),
        "--spatial-phase-mix-prob", str(spatial_mix_prob),
        "--spatial-phase-seed", str(SPATIAL_PHASE_SEED),
    ]


def build_eval_command(args: argparse.Namespace, residual: Path, out_dir: Path) -> list[str]:
    return [
        sys.executable, "-u", "-B", str(TOOLS / "evaluate_clean_residual_checkpoint.py"),
        "--real-root", str(args.real_root),
        "--split-manifest", str(args.out_root / "train_dev_manifest.json"),
        "--backbone-checkpoint", str(args.strong_backbone),
        "--residual-checkpoint", str(residual),
        "--model-root", str(args.kit_root),
        "--base-model", "sota_v2_mf",
        "--out-dir", str(out_dir),
        "--workers", str(args.workers),
        "--require-cuda",
    ]


def prepare(args: argparse.Namespace) -> None:
    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_root}")
    for path in (args.real_root, args.strong_backbone, args.strong_residual, args.kit_root, SPLIT_MANIFEST):
        if not path.exists():
            raise FileNotFoundError(path)
    if sha256(args.strong_backbone) != STRONG_BACKBONE_SHA256:
        raise RuntimeError("Strong Backbone checkpoint SHA mismatch")
    if sha256(args.strong_residual) != STRONG_RESIDUAL_SHA256:
        raise RuntimeError("Strong residual checkpoint SHA mismatch")

    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    validate_split_payload(split)
    manifest = training_manifest(split)

    # This gate never resolves or opens holdout files.
    train_names = [row["file"] if isinstance(row, dict) else row for row in manifest["train"]]
    dev_names = [row["file"] if isinstance(row, dict) else row for row in manifest["dev"]]
    missing = [name for name in train_names + dev_names if not (args.real_root / str(name)).is_file()]
    if missing:
        raise FileNotFoundError(f"missing train/dev files: {missing[:10]}")

    args.out_root.mkdir(parents=True)
    dump(args.out_root / "train_dev_manifest.json", manifest)
    dump(args.out_root / "preflight.json", {
        "status": STATUS,
        "protocol": PROTOCOL,
        "execution_commit": require_clean_main_checkout(REPO),
        "gpu": require_gpu(),
        "canonical_split_manifest": str(SPLIT_MANIFEST),
        "canonical_split_manifest_sha256": sha256(SPLIT_MANIFEST),
        "train_trajectories": len(train_names),
        "seen_dev_trajectories": len(dev_names),
        "holdout_accessed": False,
        "strong_backbone_sha256": sha256(args.strong_backbone),
        "strong_residual_sha256": sha256(args.strong_residual),
        "updates_per_arm": UPDATES,
        "batch_size": BATCH_SIZE,
        "samples_per_arm": UPDATES * BATCH_SIZE,
        "lr": LR,
        "candidate_spatial_phase_mix_prob": CANDIDATE_MIX_PROB,
        "candidate_target_phase_distribution": {
            "P00": 0.50, "P01": 1.0 / 6.0, "P10": 1.0 / 6.0, "P11": 1.0 / 6.0,
        },
        "evaluation_phase": "P00",
        "codabench_accessed": False,
        "locked_final_accessed": False,
        "full_data_refit_started": False,
        "submission_packaging_started": False,
    })


def run_arm(args: argparse.Namespace, name: str, mix_prob: float) -> None:
    root = args.out_root / name
    if root.exists():
        raise FileExistsError(root)
    train_dir = root / "train"
    run(build_arm_command(args, train_dir, spatial_mix_prob=mix_prob), root / "train.log")
    eval_dir = root / "seen_dev_best_p00"
    run(build_eval_command(args, train_dir / "model_best.pth", eval_dir), root / "seen_dev_best_p00.log")
    dump(root / "arm_manifest.json", {
        "status": STATUS,
        "arm": name,
        "spatial_phase_mix_prob": mix_prob,
        "training_phase_policy": "P00 only" if mix_prob == 0.0 else "50% P00 + 50% balanced P01/P10/P11",
        "evaluation_phase": "P00",
        "holdout_accessed": False,
        "codabench_accessed": False,
        "locked_final_accessed": False,
    })
    (root / "DONE").touch()


def _metrics(path: Path) -> dict[str, float]:
    row = json.loads(path.read_text(encoding="utf-8"))
    return {key: float(row[key]) for key in ("rel_l2_raw", "tke_raw", "mvpe_raw")}


def _state_parity(left: Path, right: Path) -> float:
    a = torch.load(left, map_location="cpu", weights_only=False)["model_state_dict"]
    b = torch.load(right, map_location="cpu", weights_only=False)["model_state_dict"]
    if set(a) != set(b):
        raise RuntimeError("arm initialization state keys differ")
    worst = 0.0
    for key in a:
        left_value, right_value = a[key], b[key]
        if not (torch.is_tensor(left_value) and torch.is_tensor(right_value)):
            if left_value != right_value:
                raise RuntimeError(f"non-tensor initialization state differs at {key}")
            continue
        if left_value.shape != right_value.shape or left_value.dtype != right_value.dtype:
            raise RuntimeError(
                f"initialization tensor metadata differs at {key}: "
                f"{tuple(left_value.shape)}/{left_value.dtype} vs "
                f"{tuple(right_value.shape)}/{right_value.dtype}"
            )
        if left_value.numel() == 0:
            # Some modules legitimately persist empty buffers. Equal shape/dtype
            # means there is no value payload to compare and max() is undefined.
            continue
        if left_value.is_floating_point() or left_value.is_complex():
            worst = max(
                worst,
                float((left_value - right_value).abs().max().item()),
            )
        elif not torch.equal(left_value, right_value):
            raise RuntimeError(f"non-floating initialization state differs at {key}")
    return worst


def review(args: argparse.Namespace) -> None:
    control_root = args.out_root / "control"
    candidate_root = args.out_root / "candidate"
    for root in (control_root, candidate_root):
        if not (root / "DONE").is_file():
            raise FileNotFoundError(f"arm not complete: {root}")

    control_train = control_root / "train"
    candidate_train = candidate_root / "train"
    init_diff = _state_parity(control_train / "model_init.pth", candidate_train / "model_init.pth")
    temporal_sampling_equal = sha256(control_train / "window_audit.jsonl") == sha256(candidate_train / "window_audit.jsonl")
    if init_diff != 0.0:
        raise RuntimeError(f"matched initialization failed: max_abs_diff={init_diff}")
    if not temporal_sampling_equal:
        raise RuntimeError("matched temporal sampling failed: window_audit differs")

    control_audit = json.loads((control_train / "sampling_audit.json").read_text(encoding="utf-8"))
    candidate_audit = json.loads((candidate_train / "sampling_audit.json").read_text(encoding="utf-8"))
    control_phase = control_audit["spatial_phase_fractions"]
    candidate_phase = candidate_audit["spatial_phase_fractions"]
    phase_checks = {
        "control_is_all_p00": abs(float(control_phase["P00"]) - 1.0) <= 1e-12,
        "candidate_p00_near_half": abs(float(candidate_phase["P00"]) - 0.5) <= 0.02,
        "candidate_all_alt_phases_present": all(float(candidate_phase[key]) > 0.14 for key in ("P01", "P10", "P11")),
    }
    if not all(phase_checks.values()):
        raise RuntimeError(f"spatial phase exposure audit failed: {phase_checks}")

    control = _metrics(control_root / "seen_dev_best_p00/final_primary_metrics.json")
    candidate = _metrics(candidate_root / "seen_dev_best_p00/final_primary_metrics.json")
    gate = spatial_gate(candidate, control)

    summary = {
        "status": STATUS,
        "protocol": PROTOCOL,
        "control": control,
        "candidate": candidate,
        "gate": gate,
        "init_max_abs_diff": init_diff,
        "temporal_sampling_equal": temporal_sampling_equal,
        "control_window_audit_sha256": sha256(control_train / "window_audit.jsonl"),
        "candidate_window_audit_sha256": sha256(candidate_train / "window_audit.jsonl"),
        "phase_exposure_checks": phase_checks,
        "control_spatial_phase_fractions": control_phase,
        "candidate_spatial_phase_fractions": candidate_phase,
        "control_best_checkpoint_sha256": sha256(control_train / "model_best.pth"),
        "candidate_best_checkpoint_sha256": sha256(candidate_train / "model_best.pth"),
        "holdout_accessed": False,
        "codabench_accessed": False,
        "locked_final_accessed": False,
        "full_data_refit_started": False,
        "submission_packaging_started": False,
        "automatic_followup_experiment_started": False,
    }
    dump(args.out_root / "summary.json", summary)

    lines = [
        "# REALPDE Spatial Phase Gate V1",
        "",
        "Status: `REVIEW_REQUIRED`",
        "",
        "One matched screen only. Evaluation is official P00 Seen-Dev12.",
        "",
        f"- Gate: **{gate['status']}**",
        f"- Control: `{control}`",
        f"- Candidate: `{candidate}`",
        f"- Relative deltas: `{gate['delta_pct']}`",
        f"- Mean raw-error delta: `{gate['mean_delta_pct']:.4f}%`",
        f"- Point-score delta: `{gate['point_score_delta']:.6f}`",
        f"- Candidate spatial exposure: `{candidate_phase}`",
        f"- Init parity max abs: `{init_diff}`",
        f"- Temporal sampling identical: `{temporal_sampling_equal}`",
        "",
        "Safety: holdout/Codabench/locked-final/full-data/submission were not accessed or started.",
        "",
        "STOP here. Sol reviews GO/NO_GO; Codex must not start another experiment automatically.",
        "",
    ]
    (args.out_root / "SPATIAL_PHASE_GATE_REVIEW.md").write_text("\n".join(lines), encoding="utf-8")
    (args.out_root / "DONE").touch()


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--strong-backbone", type=Path, required=True)
    parser.add_argument("--strong-residual", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "control", "candidate", "review", "run-all"):
        p = sub.add_parser(name)
        add_common(p)
    args = parser.parse_args()

    if args.cmd == "preflight":
        prepare(args)
        return
    if args.cmd == "run-all":
        prepare(args)
        run_arm(args, "control", 0.0)
        run_arm(args, "candidate", CANDIDATE_MIX_PROB)
        review(args)
        return
    if not args.out_root.exists():
        raise FileNotFoundError("run preflight first")
    if args.cmd == "control":
        run_arm(args, "control", 0.0)
    elif args.cmd == "candidate":
        run_arm(args, "candidate", CANDIDATE_MIX_PROB)
    elif args.cmd == "review":
        review(args)


if __name__ == "__main__":
    main()
