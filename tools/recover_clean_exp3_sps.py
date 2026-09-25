#!/usr/bin/env python3
"""Review-only recovery for CLEAN_BASELINE_FINAL_CAMPAIGN Experiment 3.

This script fixes only the calibration-selection interpretation of an already
completed Exp3 run. It performs zero optimizer steps and never retrains the
uncertainty head. The existing head checkpoint is usable only when the
constrained Seen-Dev optimum occurs at the checkpoint's saved selected step.

AoA10 holdout is accessed once, without recalibration, only after the corrected
Seen-Dev gate is verified GO.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from colleague_80pt.realpde_h5_feature_adapter_train import H5WindowDataset, paths_from_split_manifest
from colleague_80pt.residual_multi import load_full_residual_model
from colleague_80pt.train_head_fast import Head3D, HeadConfig
from colleague_80pt import realpde_sps_scoring as SPS

MIN_DEV_SPS_GAIN = 1.0
MAX_DEV_WIDTH_RATIO = 1.20
POINT_PARITY_TOL = 1e-7
EXPECTED_HEAD_SHA256 = "1cde13a23fe15547dc2ee16cede9edd89fe8c12bc88f8e9ad059641f0a6c90b8"


def dataset(paths: list[Path], stride: int) -> H5WindowDataset:
    return H5WindowDataset(
        paths,
        in_steps=20,
        out_steps=20,
        stride=stride,
        sub_sample=2,
        include_pressure=False,
        window_mode="fixed",
    )


@torch.no_grad()
def collect(model, head, paths: list[Path], device: torch.device, workers: int):
    ds = dataset(paths, 20)
    loader = DataLoader(
        ds,
        batch_size=32,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    preds, bases, sigmas, targets = [], [], [], []
    model.eval()
    head.eval()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        base = model.base_predict(x)
        delta = model.predict_delta(x, base)
        final = model.combine(base, delta, 1.0)
        log_std = head(x, base)
        preds.append(final.cpu().numpy().astype(np.float32))
        bases.append(base.cpu().numpy().astype(np.float32))
        sigmas.append(torch.exp(log_std).cpu().numpy().astype(np.float32))
        targets.append(y.numpy().astype(np.float32))
    return (
        ds,
        np.concatenate(preds),
        np.concatenate(bases),
        np.concatenate(sigmas),
        np.concatenate(targets),
    )


def adaptive_score(pred, target, sigma, floor: float, mu: float, mv: float, rel: float) -> dict[str, float]:
    half = np.zeros_like(pred, dtype=np.float32)
    half[..., 0] = floor + mu * sigma[..., 0] + rel * np.abs(pred[..., 0])
    half[..., 1] = floor + mv * sigma[..., 1] + rel * np.abs(pred[..., 1])
    c = SPS.measured_channels(target)
    raw, cov = SPS.aggregate_sps(pred, target, c, pred - half, pred + half)
    return {
        "sps": 100.0 * raw,
        "coverage": cov,
        "mean_width_uv": float(np.mean(2.0 * half[..., :2])),
    }


def static_score(pred, target, abs_w: float, rel_w: float) -> dict[str, float]:
    half = (abs_w + rel_w * np.abs(pred)).astype(np.float32)
    half[..., 2] = 0.0
    c = SPS.measured_channels(target)
    raw, cov = SPS.aggregate_sps(pred, target, c, pred - half, pred + half)
    return {
        "sps": 100.0 * raw,
        "coverage": cov,
        "mean_width_uv": float(np.mean(2.0 * half[..., :2])),
    }


def select_adaptive_under_width_cap(
    rows: list[dict[str, float]],
    static_best: dict[str, float],
    max_width_ratio: float = MAX_DEV_WIDTH_RATIO,
) -> dict[str, float]:
    cap = max_width_ratio * float(static_best["mean_width_uv"])
    feasible = [row for row in rows if float(row["mean_width_uv"]) <= cap + 1e-12]
    if not feasible:
        raise RuntimeError(f"no adaptive calibration satisfies width cap {cap:.12g}")
    return max(feasible, key=lambda row: float(row["sps"]))


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_close(name: str, actual: float, expected: float, tol: float = 1e-5) -> None:
    if abs(float(actual) - float(expected)) > tol:
        raise RuntimeError(f"{name} mismatch: actual={actual} expected={expected} tol={tol}")


@torch.no_grad()
def collect_point_only(model, paths: list[Path], device: torch.device, workers: int):
    ds = dataset(paths, 20)
    loader = DataLoader(
        ds,
        batch_size=32,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    preds, targets = [], []
    model.eval()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        base = model.base_predict(x)
        delta = model.predict_delta(x, base)
        final = model.combine(base, delta, 1.0)
        preds.append(final.cpu().numpy().astype(np.float32))
        targets.append(y.numpy().astype(np.float32))
    return np.concatenate(preds), np.concatenate(targets)


def recover_selection(source_run: Path) -> tuple[dict, dict, list[dict]]:
    static_rows = load_json(source_run / "static_calibration_grid.json")
    if not static_rows:
        raise RuntimeError("empty static calibration grid")
    static_best = max(static_rows, key=lambda row: float(row["sps"]))

    records = []
    grid_files = sorted(source_run.glob("calibration_grid_*.json"))
    if not grid_files:
        raise RuntimeError("no adaptive calibration grids found")
    for path in grid_files:
        try:
            step = int(path.stem.rsplit("_", 1)[1])
        except Exception as exc:
            raise RuntimeError(f"cannot parse calibration step from {path.name}") from exc
        rows = load_json(path)
        if not rows:
            raise RuntimeError(f"empty calibration grid: {path}")
        unconstrained = max(rows, key=lambda row: float(row["sps"]))
        constrained = select_adaptive_under_width_cap(rows, static_best, MAX_DEV_WIDTH_RATIO)
        records.append({
            "step": step,
            "constrained_best": constrained,
            "unconstrained_best": unconstrained,
        })

    selected = max(records, key=lambda row: float(row["constrained_best"]["sps"]))
    return static_best, selected, records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--real-root", type=Path, required=True)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--holdout-manifest", type=Path, required=True)
    p.add_argument("--residual-checkpoint", type=Path, required=True)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--expected-head-sha256", default=EXPECTED_HEAD_SHA256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--require-cuda", action="store_true")
    args = p.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if not args.source_run.is_dir():
        raise FileNotFoundError(args.source_run)

    head_path = args.source_run / "head_best.pth"
    summary_path = args.source_run / "summary.json"
    if not head_path.is_file():
        raise FileNotFoundError(
            f"source head checkpoint missing: {head_path}; recovery must stop, never retrain"
        )
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)

    head_hash = sha256(head_path)
    if head_hash != args.expected_head_sha256:
        raise RuntimeError(
            f"source head SHA mismatch: got {head_hash}, expected {args.expected_head_sha256}"
        )

    static_best, selected_record, records = recover_selection(args.source_run)
    selected_step = int(selected_record["step"])
    selected = selected_record["constrained_best"]

    checkpoint = torch.load(head_path, map_location="cpu", weights_only=False)
    checkpoint_step = int(checkpoint["selected_step"])
    if selected_step != checkpoint_step:
        raise RuntimeError(
            "constrained optimum requires a different head checkpoint "
            f"(grid step={selected_step}, saved head step={checkpoint_step}); "
            "review-only recovery cannot continue and must not retrain"
        )

    source_summary = load_json(summary_path)
    if int(source_summary.get("selected_step", -1)) != checkpoint_step:
        raise RuntimeError("source summary/head selected_step mismatch")

    train_paths, dev_paths = paths_from_split_manifest(args.real_root, args.train_manifest)
    _, holdout_paths = paths_from_split_manifest(args.real_root, args.holdout_manifest)
    if (len(train_paths), len(dev_paths), len(holdout_paths)) != (51, 12, 18):
        raise ValueError("expected clean 51/12/18 split")
    train_names = {p.name for p in train_paths}
    dev_names = {p.name for p in dev_paths}
    holdout_names = {p.name for p in holdout_paths}
    if train_names & dev_names or holdout_names & (train_names | dev_names):
        raise ValueError("train/dev/AoA10 holdout overlap")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")

    model, _ = load_full_residual_model(args.residual_checkpoint, args.model_root, device)
    model.eval()
    cfg = HeadConfig(**checkpoint["head_config"])
    head = Head3D(cfg).to(device)
    head.load_state_dict(checkpoint["head_state_dict"], strict=True)
    head.eval()

    point_only, point_targets = collect_point_only(model, dev_paths, device, args.workers)
    _, pred_dev, _, sigma_dev, target_dev = collect(model, head, dev_paths, device, args.workers)
    if not np.array_equal(point_targets, target_dev):
        raise RuntimeError("Seen-Dev target parity failed")
    parity = float(np.max(np.abs(point_only - pred_dev)))
    if parity > POINT_PARITY_TOL:
        raise RuntimeError(f"SPS head changed point prediction: {parity}")

    static_dev = static_score(
        pred_dev, target_dev, float(static_best["abs"]), float(static_best["rel"])
    )
    candidate_dev = adaptive_score(
        pred_dev,
        target_dev,
        sigma_dev,
        float(selected["floor"]),
        float(selected["mult_u"]),
        float(selected["mult_v"]),
        float(selected["rel"]),
    )
    for key in ("sps", "coverage", "mean_width_uv"):
        assert_close(f"static_dev.{key}", static_dev[key], static_best[key])
        assert_close(f"candidate_dev.{key}", candidate_dev[key], selected[key])

    gain = float(candidate_dev["sps"] - static_dev["sps"])
    width_ratio = float(candidate_dev["mean_width_uv"] / max(static_dev["mean_width_uv"], 1e-12))
    checks = {
        "sps_gain_ge_min": gain >= MIN_DEV_SPS_GAIN,
        "width_ratio_le_max": width_ratio <= MAX_DEV_WIDTH_RATIO + 1e-12,
        "point_parity_le_tol": parity <= POINT_PARITY_TOL,
    }
    gate_status = "GO" if all(checks.values()) else "NO_GO"

    args.out_dir.mkdir(parents=True)
    dump(args.out_dir / "selection_audit.json", {
        "status": "REVIEW_REQUIRED",
        "selection_policy": "maximize adaptive Seen-Dev SPS subject to frozen width cap",
        "source_run": str(args.source_run),
        "static_best": static_best,
        "all_steps": records,
        "selected_step": selected_step,
        "selected_calibration": selected,
        "source_head_sha256": head_hash,
        "optimizer_steps": 0,
        "training_performed": False,
    })

    summary = {
        "status": "REVIEW_REQUIRED",
        "recovery_kind": "review-only constrained calibration selection",
        "selection_policy": "maximize adaptive Seen-Dev SPS subject to frozen width cap",
        "source_run": str(args.source_run),
        "source_head_sha256": head_hash,
        "selected_step": selected_step,
        "selected_calibration": selected,
        "static_dev": static_dev,
        "candidate_dev": candidate_dev,
        "gate": {
            "status": gate_status,
            "checks": checks,
            "dev_sps_gain_vs_static": gain,
            "min_dev_sps_gain": MIN_DEV_SPS_GAIN,
            "dev_width_ratio": width_ratio,
            "max_dev_width_ratio": MAX_DEV_WIDTH_RATIO,
            "point_prediction_parity_max_abs": parity,
            "point_prediction_parity_tol": POINT_PARITY_TOL,
        },
        "optimizer_steps": 0,
        "training_performed": False,
        "holdout_accessed": False,
        "holdout_used_for_selection": False,
        "holdout_recalibrated": False,
        "codabench_accessed": False,
        "locked_final_accessed": False,
    }

    if gate_status == "GO":
        _, pred_h, _, sigma_h, target_h = collect(
            model, head, holdout_paths, device, args.workers
        )
        static_h = static_score(
            pred_h, target_h, float(static_best["abs"]), float(static_best["rel"])
        )
        candidate_h = adaptive_score(
            pred_h,
            target_h,
            sigma_h,
            float(selected["floor"]),
            float(selected["mult_u"]),
            float(selected["mult_v"]),
            float(selected["rel"]),
        )
        summary["static_holdout_aoa10"] = static_h
        summary["candidate_holdout_aoa10"] = candidate_h
        summary["gate"]["holdout_sps_delta_vs_static"] = float(candidate_h["sps"] - static_h["sps"])
        summary["gate"]["holdout_coverage_delta_vs_static"] = float(candidate_h["coverage"] - static_h["coverage"])
        summary["holdout_accessed"] = True

    dump(args.out_dir / "summary.json", summary)
    (args.out_dir / "DONE").touch()


if __name__ == "__main__":
    main()
