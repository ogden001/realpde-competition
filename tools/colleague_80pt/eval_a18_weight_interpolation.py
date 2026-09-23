#!/usr/bin/env python3
"""Evaluate frozen weight interpolation between current80 and Arm-A@18k.

Scientific variable only:
    theta_beta = theta_current80 + beta * (theta_A18 - theta_current80)

Only ResidualCorrector3D parameters/buffers are interpolated. The frozen
backbone must be bitwise identical in both checkpoints. No training, alpha
scan, parameter sweep, package build, Codabench access, or locked-final access
is implemented here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from residual_multi import (  # noqa: E402
    H5WindowDataset,
    evaluate_alphas,
    load_full_residual_model,
    paths_from_split_manifest,
)
from run_incremental_screen import (  # noqa: E402
    EXPECTED_BASE_SHA256,
    EXPECTED_START_SHA256,
    FIXED_TIME_SECONDS,
    require_clean_main_checkout,
    require_gpu,
    sha256,
    verify_allowed_data,
)

EXPECTED_A18_SHA256 = "852b41cda7d15ce9b205aecb32e206712e7a036b18981a62a9afc4e9a0b9a238"
BETAS = (0.50, 0.65, 0.80, 1.00)
BASELINE_CURRENT80 = {
    "rel_l2_raw": 0.08042041957378387,
    "tke_raw": 0.449158251285553,
    "mvpe_raw": 0.07111691683530807,
}
EXPECTED_A18 = {
    "rel_l2_raw": 0.0810459096,
    "tke_raw": 0.4235429764,
    "mvpe_raw": 0.0707221480,
}
A18_PARITY_ATOL = 5e-6


def _checkpoint_state(payload: object) -> dict[str, Tensor]:
    if not isinstance(payload, dict):
        raise TypeError("checkpoint must be a mapping")
    raw = payload.get("model_state_dict", payload)
    if not isinstance(raw, Mapping):
        raise TypeError("checkpoint model_state_dict must be a mapping")
    fixed: dict[str, Tensor] = {}
    for key, value in raw.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"state value for {key!r} is not a tensor")
        new_key = str(key)
        for prefix in ("module.", "model.", "rcm."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        fixed[new_key] = value.detach().cpu()
    return fixed


def _equal_tensor(a: Tensor, b: Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


def interpolate_residual_state(
    current: Mapping[str, Tensor],
    target: Mapping[str, Tensor],
    beta: float,
) -> dict[str, Tensor]:
    """Interpolate corrector state only and require frozen state parity."""
    beta = float(beta)
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    if set(current) != set(target):
        missing_current = sorted(set(target) - set(current))
        missing_target = sorted(set(current) - set(target))
        raise ValueError(
            f"checkpoint state keys differ: missing_current={missing_current[:5]} "
            f"missing_target={missing_target[:5]}"
        )

    result: dict[str, Tensor] = {}
    corrector_keys = 0
    for key in sorted(current):
        a = current[key]
        b = target[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"tensor metadata mismatch for {key}")

        if key.startswith("corrector."):
            corrector_keys += 1
            if torch.is_floating_point(a) or torch.is_complex(a):
                result[key] = torch.lerp(a, b, beta)
            else:
                if not _equal_tensor(a, b):
                    raise ValueError(f"non-floating corrector buffer differs for {key}")
                result[key] = a.clone()
        else:
            if not _equal_tensor(a, b):
                raise ValueError(
                    f"non-corrector state differs for {key}; "
                    "weight interpolation requires an identical frozen backbone"
                )
            result[key] = a.clone()

    if corrector_keys == 0:
        raise ValueError("no corrector.* tensors found in checkpoint state")
    return result


def pct_delta(value: float, baseline: float) -> float:
    return 100.0 * (float(value) - float(baseline)) / max(abs(float(baseline)), 1e-12)


def mechanical_gate(metrics: Mapping[str, float]) -> dict[str, object]:
    rel = pct_delta(metrics["rel_l2_raw"], BASELINE_CURRENT80["rel_l2_raw"])
    tke = pct_delta(metrics["tke_raw"], BASELINE_CURRENT80["tke_raw"])
    mvpe = pct_delta(metrics["mvpe_raw"], BASELINE_CURRENT80["mvpe_raw"])
    checks = {
        "tke_improvement_ge_3pct": tke <= -3.0,
        "rel_degradation_le_0p5pct": rel <= 0.5,
        "mvpe_degradation_le_0p3pct": mvpe <= 0.3,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "delta_rel_l2_pct_vs_current80": rel,
        "delta_tke_pct_vs_current80": tke,
        "delta_mvpe_pct_vs_current80": mvpe,
    }


def select_candidate(rows: list[dict[str, object]]) -> dict[str, object]:
    passing = [row for row in rows if bool(row["mechanical_gate"]["pass"])]
    pool = passing if passing else rows
    selected = max(pool, key=lambda row: float(row["final_est"]))
    return {
        "selection_policy": (
            "highest_final_est_among_gate_passing"
            if passing
            else "highest_final_est_no_gate_passing_candidate"
        ),
        "gate_passing_count": len(passing),
        "selected_beta": float(selected["beta"]),
        "selected_final_est": float(selected["final_est"]),
    }


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--current80-checkpoint", type=Path, required=True)
    parser.add_argument("--a18-checkpoint", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_root}")
    for path in (
        args.real_root,
        args.split_manifest,
        args.data_manifest,
        args.current80_checkpoint,
        args.a18_checkpoint,
        args.model_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    if sha256(args.current80_checkpoint) != EXPECTED_START_SHA256:
        raise RuntimeError("current80 residual checkpoint SHA-256 mismatch")
    if sha256(args.a18_checkpoint) != EXPECTED_A18_SHA256:
        raise RuntimeError("A18 residual checkpoint SHA-256 mismatch")

    execution_commit = require_clean_main_checkout(REPO_ROOT)
    gpu = require_gpu()
    verified_data = verify_allowed_data(
        args.real_root, args.data_manifest, args.split_manifest
    )

    current_payload = torch.load(args.current80_checkpoint, map_location="cpu")
    target_payload = torch.load(args.a18_checkpoint, map_location="cpu")
    current_state = _checkpoint_state(current_payload)
    target_state = _checkpoint_state(target_payload)

    current_config = current_payload.get("corrector_config")
    target_config = target_payload.get("corrector_config")
    if current_config != target_config:
        raise ValueError("current80 and A18 corrector_config differ")

    # Validate all non-corrector state before any GPU evaluation.
    interpolate_residual_state(current_state, target_state, 0.0)

    args.out_root.mkdir(parents=True)
    dump(
        args.out_root / "run_manifest.json",
        {
            "experiment": "current80_to_A18_corrector_weight_interpolation",
            "execution_commit": execution_commit,
            "gpu": gpu,
            "verified_data": verified_data,
            "current80_checkpoint": str(args.current80_checkpoint),
            "current80_sha256": sha256(args.current80_checkpoint),
            "a18_checkpoint": str(args.a18_checkpoint),
            "a18_sha256": sha256(args.a18_checkpoint),
            "stage1_base_expected_sha256": EXPECTED_BASE_SHA256,
            "betas": list(BETAS),
            "runtime_alpha": 1.0,
            "fixed_time_seconds": FIXED_TIME_SECONDS,
            "interpolated_scope": "corrector.* only",
            "frozen_backbone_requires_bitwise_parity": True,
            "training_started": False,
            "locked_final_accessed": False,
            "codabench_accessed": False,
        },
    )

    device = torch.device("cuda:0")
    model, _ = load_full_residual_model(
        args.current80_checkpoint, args.model_root, device
    )
    # load_full_residual_model freezes everything; corrector does not need grads here.
    model.eval()

    _, val_paths = paths_from_split_manifest(
        args.real_root,
        args.split_manifest,
        allow_train_dev_overlap=True,
    )
    val_dataset = H5WindowDataset(
        val_paths,
        in_steps=20,
        out_steps=20,
        stride=20,
        sub_sample=2,
        include_pressure=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    if len(val_dataset) != 640:
        raise RuntimeError(f"expected 640 Dev windows, got {len(val_dataset)}")

    rows: list[dict[str, object]] = []
    for beta in BETAS:
        state = interpolate_residual_state(current_state, target_state, beta)
        model.load_state_dict(state, strict=True)
        summaries = evaluate_alphas(
            model,
            val_loader,
            device,
            alphas=[1.0],
            abs_widths=[0.0075],
            rel_widths=[0.0075],
            fixed_time_seconds=FIXED_TIME_SECONDS,
        )
        if len(summaries) != 1:
            raise RuntimeError("expected exactly one alpha=1.0 evaluation row")
        summary = summaries[0]
        metrics = {
            "rel_l2_raw": float(summary["rel_l2"]),
            "tke_raw": float(summary["tke"]),
            "mvpe_raw": float(summary["mvpe"]),
        }
        gate = mechanical_gate(metrics)
        row = {
            "beta": beta,
            **metrics,
            "final_est": float(summary["best_final_est"]),
            "sps_score": float(summary["best_bounds"][0]["sps_score_used"]),
            "time_score": float(summary["time_score"]),
            "mechanical_gate": gate,
        }
        rows.append(row)
        dump(args.out_root / f"beta_{beta:.2f}.json", row)

    beta1 = next(row for row in rows if abs(float(row["beta"]) - 1.0) < 1e-12)
    parity = {
        key: abs(float(beta1[key]) - expected)
        for key, expected in EXPECTED_A18.items()
    }
    if any(delta > A18_PARITY_ATOL for delta in parity.values()):
        raise RuntimeError(f"beta=1.0 failed A18 parity: {parity}")

    selection = select_candidate(rows)
    report = {
        "status": "REVIEW_REQUIRED",
        "baseline_current80": BASELINE_CURRENT80,
        "expected_A18": EXPECTED_A18,
        "beta1_A18_parity_abs_error": parity,
        "rows": rows,
        **selection,
        "training_started": False,
        "automatic_merge_started": False,
        "codabench_accessed": False,
        "locked_final_accessed": False,
    }
    dump(args.out_root / "summary.json", report)

    with (args.out_root / "per_beta.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "beta",
            "rel_l2_raw",
            "tke_raw",
            "mvpe_raw",
            "delta_rel_l2_pct_vs_current80",
            "delta_tke_pct_vs_current80",
            "delta_mvpe_pct_vs_current80",
            "gate_pass",
            "final_est",
            "sps_score",
            "time_score",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            gate = row["mechanical_gate"]
            writer.writerow(
                {
                    "beta": row["beta"],
                    "rel_l2_raw": row["rel_l2_raw"],
                    "tke_raw": row["tke_raw"],
                    "mvpe_raw": row["mvpe_raw"],
                    "delta_rel_l2_pct_vs_current80": gate[
                        "delta_rel_l2_pct_vs_current80"
                    ],
                    "delta_tke_pct_vs_current80": gate[
                        "delta_tke_pct_vs_current80"
                    ],
                    "delta_mvpe_pct_vs_current80": gate[
                        "delta_mvpe_pct_vs_current80"
                    ],
                    "gate_pass": gate["pass"],
                    "final_est": row["final_est"],
                    "sps_score": row["sps_score"],
                    "time_score": row["time_score"],
                }
            )
    (args.out_root / "DONE").touch()
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
