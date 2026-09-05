#!/usr/bin/env python3
"""Track 1 training-strategy screen: reconstruction-horizon curriculum.

The architecture, P0-A feature set, N2 weights, data split, initialization,
optimizer, and total 3000-update budget match the registered Direct control.
Only the reconstruction objective schedule changes. TKE is always computed on
all Future20 frames to avoid changing the physical-statistics definition.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SEED = 20260901
UPDATES = 3000
EXPECTED_MANIFEST_SHA256 = "42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347"
EXPECTED_START_CHECKPOINT_SHA256 = "af85374bfd06c0e386ec803d777396c21484978392213025697c5a7470106b6b"
EXPECTED_REFERENCE_CHECKPOINT_SHA256 = "9fa52c905c2603179da39e90da7689a1d65005d5c00d0df2d9fe2e310cf32aeb"
EXPECTED_SCORER_SHA256 = "a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39"
N2_WEIGHTS = {"mse": 1.0, "tke": 0.05, "rel": 0.027514, "mvpe": 0.009757}
RECON_TERMS = ("mse", "rel", "mvpe")
MILESTONES = [500, 1000, 1500, 2000, 2500, 3000]


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def curriculum_stage(step: int) -> dict[str, float | int]:
    if step < 1 or step > UPDATES:
        raise ValueError(f"curriculum step must be in [1,{UPDATES}], got {step}")
    if step <= 1000:
        return {"short_horizon": 5, "short_weight": 0.75, "full_weight": 0.25}
    if step <= 2000:
        return {"short_horizon": 10, "short_weight": 0.5, "full_weight": 0.5}
    return {"short_horizon": 20, "short_weight": 0.0, "full_weight": 1.0}


def curriculum_loss(pred: Tensor, target: Tensor, step: int, core) -> tuple[Tensor, dict]:
    if pred.shape[1] != 20 or target.shape[1] != 20:
        raise ValueError(f"expected Future20 tensors, got {pred.shape[1]}/{target.shape[1]}")
    stage = curriculum_stage(step)
    full = core.loss_parts(pred, target)
    if stage["short_weight"] > 0.0:
        horizon = int(stage["short_horizon"])
        short = core.loss_parts(pred[:, :horizon], target[:, :horizon])
    else:
        short = full
    blended = {
        name: float(stage["short_weight"]) * short[name] + float(stage["full_weight"]) * full[name]
        for name in RECON_TERMS
    }
    blended["tke"] = full["tke"]
    loss = sum(N2_WEIGHTS[name] * blended[name] for name in N2_WEIGHTS)
    evidence = dict(stage) | {"tke_source": "full20"}
    return loss, evidence


def aggregate_improvements(baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    return {
        metric: (float(baseline[metric]) - float(candidate[metric])) / max(float(baseline[metric]), 1e-12) * 100.0
        for metric in ("rel_l2", "tke", "mvpe")
    }


def classify_gate(baseline: dict[str, float], candidate: dict[str, float]) -> dict:
    improvements = aggregate_improvements(baseline, candidate)
    if improvements["rel_l2"] >= 3.0 and improvements["mvpe"] >= 3.0 and improvements["tke"] >= -2.0:
        status = "PROMISING"
    elif improvements["tke"] < -2.0:
        status = "NO_GO"
    elif max(improvements.values()) >= 1.0:
        status = "WEAK_SIGNAL_PARKED"
    else:
        status = "NO_GO"
    return {
        "status": status,
        "improvement_pct": improvements,
        "gate": {"rel_l2_min_pct": 3.0, "mvpe_min_pct": 3.0, "tke_min_pct": -2.0},
    }


def compare_trajectory_metrics(reference_rows: list[dict], candidate_rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
    reference = {row["trajectory_id"]: row for row in reference_rows}
    candidate = {row["trajectory_id"]: row for row in candidate_rows}
    if set(reference) != set(candidate):
        raise ValueError("reference/candidate trajectory sets differ")
    rows, wins = [], {"rel_l2": 0, "tke": 0, "mvpe": 0}
    for name in sorted(reference):
        row = {"trajectory_id": name}
        for metric in wins:
            base = float(reference[name][metric])
            value = float(candidate[name][metric])
            improvement = (base - value) / max(base, 1e-12) * 100.0
            row[f"reference_{metric}"] = base
            row[f"curriculum_{metric}"] = value
            row[f"improvement_{metric}_pct"] = improvement
            row[f"{metric}_win"] = improvement > 0.0
            wins[metric] += int(improvement > 0.0)
        rows.append(row)
    return rows, wins


def validate_contract(args: argparse.Namespace, core, direct) -> tuple[list[Path], list[Path]]:
    checks = {
        "manifest": (args.manifest, EXPECTED_MANIFEST_SHA256),
        "start_checkpoint": (args.checkpoint, EXPECTED_START_CHECKPOINT_SHA256),
        "reference_checkpoint": (args.reference_checkpoint, EXPECTED_REFERENCE_CHECKPOINT_SHA256),
        "scorer": (args.kit_root / "scoring.py", EXPECTED_SCORER_SHA256),
    }
    mismatches = {}
    for name, (path, expected) in checks.items():
        actual = sha256(path)
        if actual != expected:
            mismatches[name] = {"path": str(path), "expected": expected, "actual": actual}
    if mismatches:
        raise RuntimeError(f"curriculum contract checksum mismatch: {json.dumps(mismatches, sort_keys=True)}")
    if dict(direct.N2_WEIGHTS) != N2_WEIGHTS:
        raise RuntimeError(f"N2 weights drifted: expected={N2_WEIGHTS}, actual={direct.N2_WEIGHTS}")
    _, train_paths = core.read_manifest(args.manifest, "train")
    _, dev_paths = core.read_manifest(args.manifest, "dev")
    if len(train_paths) != 50 or len(dev_paths) != 16:
        raise RuntimeError(f"curriculum requires frozen 50/16 split, got {len(train_paths)}/{len(dev_paths)}")
    return train_paths, dev_paths


def build_p0a_model(kit_root: Path, checkpoint: Path, train_paths: list[Path], device: torch.device, direct):
    builder, feature_config = direct.build_features(train_paths, device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = direct.cno_direct(kit_root, len(builder.feature_names), device)
    state = payload.get("model_state_dict", payload)
    input_key = "lift.inter_CNOBlock.convolution.weight"
    if input_key in state and state[input_key].shape[1] == len(builder.feature_names):
        model.load_state_dict(state, strict=True)
    else:
        direct.adapt_input_weight(model, payload, len(builder.feature_names))
    return builder, feature_config, model


def load_reference(kit_root: Path, checkpoint: Path, builder, device: torch.device, direct):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = direct.cno_direct(kit_root, len(builder.feature_names), device)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state, strict=True)
    return model


def forward(model: nn.Module, builder, x: Tensor) -> Tensor:
    features = builder(x)
    return model(features.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


@torch.no_grad()
def evaluate(model: nn.Module, builder, paths: list[Path], args: argparse.Namespace, device: torch.device,
             core, out: Path, save_predictions: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ds, loader = core.loader(paths, args, shuffle=False)
    model.eval()
    predictions, targets, elapsed = [], [], 0.0
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
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    scored = core.score_bundle(args.kit_root, prediction, target, elapsed / len(ds), out)
    rows, _ = core.trajectory_rows(ds, prediction, target, args.kit_root)
    with (out / "trajectory_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    if save_predictions:
        np.savez_compressed(out / "predictions.npz", prediction=prediction, target=target)
    return scored | {"windows": len(ds), "trajectories": len(rows), "trajectory_rows": rows,
                     "prediction": prediction if save_predictions else None,
                     "target": target if save_predictions else None}


def horizon_rel_l2_rows(reference_prediction: np.ndarray, candidate_prediction: np.ndarray, target: np.ndarray) -> list[dict]:
    rows = []
    for t in range(20):
        y = target[:, t, ..., :2].reshape(target.shape[0], -1)
        r = reference_prediction[:, t, ..., :2].reshape(target.shape[0], -1)
        c = candidate_prediction[:, t, ..., :2].reshape(target.shape[0], -1)
        denom = np.linalg.norm(y, axis=1).clip(min=1e-12)
        ref = float(np.mean(np.linalg.norm(r - y, axis=1) / denom))
        cand = float(np.mean(np.linalg.norm(c - y, axis=1) / denom))
        rows.append({"horizon": t + 1, "reference_rel_l2": ref, "curriculum_rel_l2": cand,
                     "improvement_pct": (ref - cand) / max(ref, 1e-12) * 100.0})
    return rows


def run_preflight(args: argparse.Namespace) -> None:
    if args.preflight_out is None:
        raise ValueError("--preflight-out is required with --preflight")
    if args.preflight_out.exists() and any(args.preflight_out.iterdir()):
        raise FileExistsError(args.preflight_out)
    args.preflight_out.mkdir(parents=True, exist_ok=True)
    core, direct = runtime_modules()
    set_seed(args.seed)
    train_paths, _ = validate_contract(args, core, direct)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    builder, _, model = build_p0a_model(args.kit_root, args.checkpoint, train_paths, device, direct)
    raw_model = core.load_cno(args.kit_root, args.checkpoint, device)
    loader_args = argparse.Namespace(batch_size=args.batch_size, workers=0, max_windows=None, seed=args.seed)
    _, loader = core.loader(train_paths, loader_args, shuffle=False)
    x, y, _, _ = next(iter(loader))
    x, y = x.to(device), y.to(device)
    model.eval(); raw_model.eval()
    with torch.no_grad():
        p0a_init = forward(model, builder, x)
        raw_init = core.forward(raw_model, x)
        init_max_abs = float((p0a_init - raw_init).abs().max())
    if init_max_abs > 1e-6:
        raise RuntimeError(f"P0-A adapted initialization drifted from raw CNO: {init_max_abs}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    pred = forward(model, builder, x)
    loss, stage = curriculum_loss(pred, y, 1, core)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite curriculum preflight loss")
    optimizer.zero_grad(set_to_none=True); loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    if grad_norm <= 0:
        raise RuntimeError("zero curriculum gradient")
    optimizer.step()
    after = [p.detach() for p in model.parameters() if p.requires_grad]
    parameter_delta = float(sum((a - b).square().sum() for a, b in zip(after, before)).sqrt())
    if parameter_delta <= 0:
        raise RuntimeError("curriculum optimizer step did not change parameters")

    save_json(args.preflight_out / "preflight.json", {
        "passed": True,
        "initial_p0a_vs_raw_max_abs": init_max_abs,
        "step1_loss": float(loss.detach()),
        "step1_stage": stage,
        "step1_grad_norm": grad_norm,
        "step1_parameter_delta": parameter_delta,
        "train_trajectories": 50,
        "dev_trajectories": 16,
        "locked_final_accessed": False,
        "codabench": False,
        "device": str(device),
    })


def run(args: argparse.Namespace) -> None:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(args.out_dir)
    if args.preflight_evidence is None:
        raise ValueError("--preflight-evidence is required for formal curriculum training")
    preflight = json.loads(args.preflight_evidence.read_text(encoding="utf-8"))
    if not preflight.get("passed"):
        raise RuntimeError("refusing curriculum training: preflight did not pass")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    core, direct = runtime_modules()
    set_seed(args.seed)
    train_paths, dev_paths = validate_contract(args, core, direct)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    builder, feature_config, model = build_p0a_model(args.kit_root, args.checkpoint, train_paths, device, direct)
    reference_model = load_reference(args.kit_root, args.reference_checkpoint, builder, device, direct)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    metadata = {
        "experiment_id": args.experiment_id,
        "strategy": "horizon_curriculum_reconstruction_only",
        "architecture": "P0-A Direct CNO",
        "unique_variable": "reconstruction horizon curriculum; TKE always full Future20",
        "schedule": {
            "1-1000": {"short_horizon": 5, "short_weight": 0.75, "full_weight": 0.25},
            "1001-2000": {"short_horizon": 10, "short_weight": 0.5, "full_weight": 0.5},
            "2001-3000": {"short_horizon": 20, "short_weight": 0.0, "full_weight": 1.0},
        },
        "loss_weights": N2_WEIGHTS,
        "seed": args.seed, "lr": args.lr, "batch_size": args.batch_size, "workers": args.workers,
        "updates": args.updates, "milestones": list(args.milestones),
        "manifest_sha256": sha256(args.manifest), "checkpoint_sha256": sha256(args.checkpoint),
        "reference_checkpoint_sha256": sha256(args.reference_checkpoint),
        "scorer_sha256": sha256(args.kit_root / "scoring.py"),
        "feature_names": list(builder.feature_names), "feature_config": vars(feature_config),
        "train_trajectories": len(train_paths), "dev_trajectories": len(dev_paths),
        "locked_final_accessed": False, "codabench": False, "device": str(device),
        "runner_sha256": sha256(Path(__file__)), "preflight_evidence": str(args.preflight_evidence),
    }
    save_json(args.out_dir / "run_metadata.json", metadata)

    reference = evaluate(reference_model, builder, dev_paths, args, device, core,
                         args.out_dir / "eval_reference_direct3000", save_predictions=True)
    initial = evaluate(model, builder, dev_paths, args, device, core,
                       args.out_dir / "eval_initial", save_predictions=False)
    history = [{"update": 0, "model": "Curriculum_init", **initial["raw_errors"]},
               {"update": 3000, "model": "Direct_reference", **reference["raw_errors"]}]

    train_ds, loader = core.loader(train_paths, args, shuffle=True)
    iterator = iter(loader)
    started = time.monotonic()
    final_result = None
    for step in range(1, args.updates + 1):
        try:
            x, y, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader); x, y, _, _ = next(iterator)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        model.train()
        pred = forward(model, builder, x)
        loss, stage = curriculum_loss(pred, y, step, core)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step in args.milestones:
            result = evaluate(model, builder, dev_paths, args, device, core,
                              args.out_dir / f"eval_{step:05d}", save_predictions=step == args.updates)
            row = {"update": step, "model": "HorizonCurriculum", **result["raw_errors"],
                   "short_horizon": stage["short_horizon"], "short_weight": stage["short_weight"],
                   "full_weight": stage["full_weight"], "mean_t_neural_s": result["mean_t_neural_s"],
                   "elapsed_seconds": time.monotonic() - started}
            history.append(row)
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "iteration": step, "metadata": metadata}, args.out_dir / f"model_update_{step:05d}.pth")
            print(json.dumps(row, sort_keys=True), flush=True)
            if step == args.updates:
                final_result = result

    assert final_result is not None
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "iteration": args.updates, "metadata": metadata}, args.out_dir / "model_last.pth")

    trajectory_rows, wins = compare_trajectory_metrics(reference["trajectory_rows"], final_result["trajectory_rows"])
    with (args.out_dir / "trajectory_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectory_rows[0])); writer.writeheader(); writer.writerows(trajectory_rows)
    horizon_rows = horizon_rel_l2_rows(reference["prediction"], final_result["prediction"], final_result["target"])
    with (args.out_dir / "horizon_rel_l2.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(horizon_rows[0])); writer.writeheader(); writer.writerows(horizon_rows)

    gate = classify_gate(reference["raw_errors"], final_result["raw_errors"])
    gate.update({"trajectory_wins": wins, "reference_raw_errors": reference["raw_errors"],
                 "candidate_raw_errors": final_result["raw_errors"], "candidate_update": args.updates})
    save_json(args.out_dir / "gate_result.json", gate)
    with (args.out_dir / "update_curve.csv").open("w", newline="") as handle:
        fields = list(dict.fromkeys(key for row in history for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(history)
    save_json(args.out_dir / "summary.json", {
        "metadata": metadata | {"train_windows": len(train_ds), "elapsed_seconds": time.monotonic() - started},
        "history": history, "gate": gate,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Exact official sim_pretrain CNO")
    parser.add_argument("--reference-checkpoint", type=Path, required=True, help="Exact matched Direct@3000")
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--milestones", type=int, nargs="+", default=MILESTONES)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-out", type=Path)
    parser.add_argument("--preflight-evidence", type=Path)
    args = parser.parse_args()
    if args.seed != SEED or args.updates != UPDATES or args.batch_size != 8 or args.lr != 1e-5:
        raise ValueError("horizon curriculum protocol is frozen: seed=20260901, updates=3000, batch=8, lr=1e-5")
    if list(args.milestones) != MILESTONES:
        raise ValueError(f"horizon curriculum milestones are frozen at {MILESTONES}")
    if args.preflight:
        run_preflight(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
