#!/usr/bin/env python3
"""Low-memory late-checkpoint soup / prediction-ensemble evaluation for Track 1.

This is a bounded 50/16 analysis-only training-strategy probe. It combines the
three already-identified late P0-A + N2 validation checkpoints at absolute
updates 26240, 27880 and 30340. Only one CNO is resident on GPU at a time.
Prediction ensembles are analysis-only upper bounds; checkpoint soups remain
single-model candidates.

No training, locked-final, Codabench, SPS or full-data access is performed.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

EXPECTED_MANIFEST_SHA256 = "42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347"
EXPECTED_SCORER_SHA256 = "a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39"
EXPECTED_RAW = {
    26240: {"rel_l2": 0.112925, "tke": 0.494840, "mvpe": 0.084671},
    27880: {"rel_l2": 0.112398, "tke": 0.496896, "mvpe": 0.088587},
    30340: {"rel_l2": 0.112939, "tke": 0.492848, "mvpe": 0.088154},
}
METRIC_TOLERANCE = 7e-4
CHECKPOINT_ORDER = (26240, 27880, 30340)
SOUPS = {
    "soup_26240_27880": (0.5, 0.5, 0.0),
    "soup_26240_30340": (0.5, 0.0, 0.5),
    "soup_27880_30340": (0.0, 0.5, 0.5),
    "soup_equal3": (1 / 3, 1 / 3, 1 / 3),
    "soup_balanced3": (0.5, 0.25, 0.25),
}
ENSEMBLES = {
    "ens_equal3": (1 / 3, 1 / 3, 1 / 3),
    "ens_balanced3": (0.5, 0.25, 0.25),
}


def runtime_modules():
    import realpde_loss_official_v9 as core
    import realpde_mf01 as direct
    return core, direct


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def normalize_weights(weights) -> list[float]:
    values = [float(v) for v in weights]
    total = sum(values)
    if total <= 0 or any(v < 0 for v in values):
        raise ValueError(f"invalid nonnegative weights: {values}")
    return [v / total for v in values]


def average_state_dicts(states: list[dict[str, Tensor]], weights) -> dict[str, Tensor]:
    if not states:
        raise ValueError("no state dicts")
    weights = normalize_weights(weights)
    keys = set(states[0])
    if any(set(state) != keys for state in states[1:]):
        raise ValueError("state dict keys differ")
    result: dict[str, Tensor] = {}
    for key in states[0]:
        tensors = [state[key] for state in states]
        if any(t.shape != tensors[0].shape or t.dtype != tensors[0].dtype for t in tensors[1:]):
            raise ValueError(f"state tensor mismatch for {key}")
        first = tensors[0]
        if first.is_floating_point() or first.is_complex():
            acc = torch.zeros_like(first, device="cpu")
            for weight, tensor in zip(weights, tensors):
                acc.add_(tensor.detach().cpu(), alpha=weight)
            result[key] = acc
        else:
            reference = first.detach().cpu()
            if any(not torch.equal(reference, tensor.detach().cpu()) for tensor in tensors[1:]):
                raise ValueError(f"non-floating state differs for {key}")
            result[key] = reference.clone()
    return result


def blend_predictions(predictions: list[np.ndarray], weights) -> np.ndarray:
    if not predictions:
        raise ValueError("no predictions")
    weights = normalize_weights(weights)
    shape = predictions[0].shape
    if any(pred.shape != shape for pred in predictions):
        raise ValueError("prediction shapes differ")
    out = np.zeros(shape, dtype=np.float32)
    for weight, pred in zip(weights, predictions):
        out += np.asarray(pred, dtype=np.float32) * np.float32(weight)
    return out


def checkpoint_map(args: argparse.Namespace) -> dict[int, Path]:
    return {
        26240: args.checkpoint_26240,
        27880: args.checkpoint_27880,
        30340: args.checkpoint_30340,
    }


def validate_contract(args: argparse.Namespace, core, direct) -> tuple[list[Path], list[Path]]:
    if sha256(args.manifest) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("manifest checksum mismatch")
    if sha256(args.kit_root / "scoring.py") != EXPECTED_SCORER_SHA256:
        raise RuntimeError("official scorer checksum mismatch")
    _, train_paths = core.read_manifest(args.manifest, "train")
    _, dev_paths = core.read_manifest(args.manifest, "dev")
    if len(train_paths) != 50 or len(dev_paths) != 16:
        raise RuntimeError(f"requires frozen 50/16 split, got {len(train_paths)}/{len(dev_paths)}")
    if args.batch_size > 2:
        raise RuntimeError("low-memory concurrent probe freezes batch_size <= 2")
    return train_paths, dev_paths


def load_model_state(path: Path) -> dict[str, Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint has no model state: {path}")
    return {str(k): v.detach().cpu() if torch.is_tensor(v) else v for k, v in state.items()}


def build_model(kit_root: Path, builder, state: dict[str, Tensor], device: torch.device, direct) -> nn.Module:
    model = direct.cno_direct(kit_root, len(builder.feature_names), device)
    model.load_state_dict(state, strict=True)
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise FloatingPointError("non-finite checkpoint parameters")
    return model


def forward(model: nn.Module, builder, x: Tensor) -> Tensor:
    features = builder(x)
    return model(features.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)


@torch.no_grad()
def infer(model: nn.Module, builder, paths: list[Path], args: argparse.Namespace, device: torch.device, core):
    ds, loader = core.loader(paths, args, shuffle=False)
    predictions, targets, elapsed = [], [], 0.0
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        prediction = forward(model, builder, x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        predictions.append(prediction.cpu().numpy().astype(np.float32))
        targets.append(y.numpy().astype(np.float32))
    peak_reserved = float(torch.cuda.max_memory_reserved(device) / (1024 ** 3)) if device.type == "cuda" else 0.0
    return (
        np.concatenate(predictions),
        np.concatenate(targets),
        elapsed / len(ds),
        peak_reserved,
        ds,
    )


def score_arrays(prediction: np.ndarray, target: np.ndarray, mean_t: float, ds, args: argparse.Namespace, core, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    scored = core.score_bundle(args.kit_root, prediction, target, mean_t, out)
    rows, _ = core.trajectory_rows(ds, prediction, target, args.kit_root)
    with (out / "trajectory_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return scored | {"trajectory_rows": rows}


def assert_expected_raw(update: int, raw: dict[str, float]) -> None:
    mismatch = {}
    for metric, expected in EXPECTED_RAW[update].items():
        actual = float(raw[metric])
        if abs(actual - expected) > METRIC_TOLERANCE:
            mismatch[metric] = {"expected": expected, "actual": actual, "tol": METRIC_TOLERANCE}
    if mismatch:
        raise RuntimeError(f"checkpoint {update} dev parity mismatch: {json.dumps(mismatch, sort_keys=True)}")


def relative_to_balanced(raw: dict[str, float]) -> dict[str, float]:
    base = EXPECTED_RAW[26240]
    return {metric: (base[metric] - float(raw[metric])) / base[metric] * 100.0 for metric in base}


def run(args: argparse.Namespace) -> None:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    core, direct = runtime_modules()
    train_paths, dev_paths = validate_contract(args, core, direct)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    builder, feature_config = direct.build_features(train_paths, device)
    ckpts = checkpoint_map(args)
    states = {update: load_model_state(path) for update, path in ckpts.items()}

    metadata = {
        "experiment_id": "T1-ID-CHECKPOINT-SOUP-S20260906",
        "purpose": "low-memory checkpoint soup and prediction ensemble of late P0-A+N2 validation checkpoints",
        "checkpoint_paths": {str(k): str(v) for k, v in ckpts.items()},
        "checkpoint_sha256": {str(k): sha256(v) for k, v in ckpts.items()},
        "manifest_sha256": sha256(args.manifest),
        "scorer_sha256": sha256(args.kit_root / "scoring.py"),
        "feature_names": list(builder.feature_names),
        "feature_config": vars(feature_config),
        "train_trajectories": len(train_paths),
        "dev_trajectories": len(dev_paths),
        "batch_size": args.batch_size,
        "max_gpu_memory_gib": args.max_gpu_memory_gib,
        "locked_final_accessed": False,
        "codabench": False,
        "training": False,
        "prediction_ensembles_analysis_only": True,
    }
    save_json(args.out_dir / "run_metadata.json", metadata)

    base_predictions: dict[int, np.ndarray] = {}
    base_times: dict[int, float] = {}
    target_reference = None
    ds_reference = None
    rows = []

    for update in CHECKPOINT_ORDER:
        model = build_model(args.kit_root, builder, states[update], device, direct)
        pred, target, mean_t, peak_gib, ds = infer(model, builder, dev_paths, args, device, core)
        if peak_gib > args.max_gpu_memory_gib:
            raise RuntimeError(f"GPU memory cap exceeded: {peak_gib:.3f} GiB > {args.max_gpu_memory_gib:.3f} GiB")
        result = score_arrays(pred, target, mean_t, ds, args, core, args.out_dir / f"base_{update}")
        assert_expected_raw(update, result["raw_errors"])
        base_predictions[update] = pred
        base_times[update] = mean_t
        if target_reference is None:
            target_reference, ds_reference = target, ds
        elif not np.array_equal(target_reference, target):
            raise RuntimeError("dev target order differs across checkpoint inference")
        rows.append({
            "candidate": f"base_{update}", "kind": "single", **result["raw_errors"],
            "mean_t_neural_s": mean_t, "peak_gpu_reserved_gib": peak_gib,
            **{f"vs26240_{k}_pct": v for k, v in relative_to_balanced(result["raw_errors"]).items()},
        })
        del model
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()

    ordered_states = [states[update] for update in CHECKPOINT_ORDER]
    for name, weights in SOUPS.items():
        state = average_state_dicts(ordered_states, weights)
        model = build_model(args.kit_root, builder, state, device, direct)
        pred, target, mean_t, peak_gib, ds = infer(model, builder, dev_paths, args, device, core)
        if peak_gib > args.max_gpu_memory_gib:
            raise RuntimeError(f"GPU memory cap exceeded: {peak_gib:.3f} GiB > {args.max_gpu_memory_gib:.3f} GiB")
        result = score_arrays(pred, target, mean_t, ds, args, core, args.out_dir / name)
        rows.append({
            "candidate": name, "kind": "weight_soup", **result["raw_errors"],
            "mean_t_neural_s": mean_t, "peak_gpu_reserved_gib": peak_gib,
            **{f"vs26240_{k}_pct": v for k, v in relative_to_balanced(result["raw_errors"]).items()},
        })
        del model, state
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()

    ordered_predictions = [base_predictions[update] for update in CHECKPOINT_ORDER]
    assert target_reference is not None and ds_reference is not None
    for name, weights in ENSEMBLES.items():
        pred = blend_predictions(ordered_predictions, weights)
        mean_t = sum(weight * base_times[update] for weight, update in zip(normalize_weights(weights), CHECKPOINT_ORDER))
        result = score_arrays(pred, target_reference, mean_t, ds_reference, args, core, args.out_dir / name)
        rows.append({
            "candidate": name, "kind": "prediction_ensemble_analysis_only", **result["raw_errors"],
            "mean_t_neural_s": mean_t, "peak_gpu_reserved_gib": 0.0,
            **{f"vs26240_{k}_pct": v for k, v in relative_to_balanced(result["raw_errors"]).items()},
        })

    with (args.out_dir / "candidate_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    best = {
        metric: min(rows, key=lambda row: float(row[metric]))["candidate"]
        for metric in ("rel_l2", "tke", "mvpe")
    }
    soup_rows = [row for row in rows if row["kind"] == "weight_soup"]
    best_soup = min(
        soup_rows,
        key=lambda row: np.mean([float(row["rel_l2"]) / EXPECTED_RAW[26240]["rel_l2"],
                                 float(row["tke"]) / EXPECTED_RAW[26240]["tke"],
                                 float(row["mvpe"]) / EXPECTED_RAW[26240]["mvpe"]]),
    )["candidate"]
    save_json(args.out_dir / "summary.json", {
        "best_candidate_by_raw_metric": best,
        "best_weight_soup_balanced_diagnostic": best_soup,
        "candidates": rows,
        "note": "balanced diagnostic is not an official final-score proxy",
        "locked_final_accessed": False,
        "codabench": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--checkpoint-26240", type=Path, required=True)
    parser.add_argument("--checkpoint-27880", type=Path, required=True)
    parser.add_argument("--checkpoint-30340", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--max-gpu-memory-gib", type=float, default=9.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
