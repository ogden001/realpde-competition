#!/usr/bin/env python3
"""Matched Direct-vs-MF long-convergence campaign for RealPDE Track 1.

Both arms start from their already-validated absolute-update-3000 checkpoints and
continue for another 12,000 optimizer updates under the same frozen 50/16,
P0-A, N2, seed, batch, optimizer and data-order semantics.  The purpose is to
measure whether the large MF@3000 advantage persists as both models converge.

No locked-final, Codabench, SPS, full-data training or model/loss/feature change
is performed by this runner.
"""
from __future__ import annotations

import argparse
import csv
import gc
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
START_ABSOLUTE_UPDATE = 3000
ADDITIONAL_UPDATES = 12000
MILESTONE_INTERVAL = 3000
EXPECTED_MANIFEST_SHA256 = "42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347"
EXPECTED_DIRECT3000_SHA256 = "9fa52c905c2603179da39e90da7689a1d65005d5c00d0df2d9fe2e310cf32aeb"
EXPECTED_MF1500_SHA256 = "488a8118f489789d385ec90e02856ef6a8482d6fa75c252e2e5d2d1f50e72226"
EXPECTED_SCORER_SHA256 = "a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39"
EXPECTED_DIRECT_EXPERIMENT_ID = "T1-ID-MF-DIRECT3000-CLOSEOUT-S20260901"
EXPECTED_MF_EXPERIMENT_ID = "T1-ID-MF-C02-CONT-S20260901"
N2_WEIGHTS = {"mse": 1.0, "tke": 0.05, "rel": 0.027514, "mvpe": 0.009757}
EXPECTED_START_RAW = {
    "direct": {"rel_l2": 0.175829, "tke": 0.594649, "mvpe": 0.151631},
    "mf": {"rel_l2": 0.164327, "tke": 0.582928, "mvpe": 0.130374},
}
START_METRIC_TOLERANCE = 5e-4


def runtime_modules():
    import realpde_loss_official_v9 as core
    import realpde_mf01 as mf01
    return core, mf01


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def absolute_milestones(start: int, additional: int, interval: int) -> list[int]:
    if start != START_ABSOLUTE_UPDATE or additional != ADDITIONAL_UPDATES:
        raise ValueError(
            f"long-convergence campaign is frozen at {START_ABSOLUTE_UPDATE}+{ADDITIONAL_UPDATES} updates"
        )
    if interval != MILESTONE_INTERVAL or additional % interval:
        raise ValueError(f"milestone interval is frozen at {MILESTONE_INTERVAL}")
    return [start + relative for relative in range(interval, additional + 1, interval)]


def validate_mf3000_metadata(payload: dict) -> None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("MF@3000 metadata mismatch: missing metadata")
    expected = {
        "experiment_id": EXPECTED_MF_EXPERIMENT_ID,
        "mode": "c0",
        "seed": SEED,
        "updates": 1500,
        "lr": 1e-5,
        "batch_size": 8,
        "workers": 2,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "checkpoint_sha256": EXPECTED_MF1500_SHA256,
        "scorer_sha256": EXPECTED_SCORER_SHA256,
    }
    mismatches = {}
    for key, expected_value in expected.items():
        actual = metadata.get(key)
        if actual != expected_value:
            mismatches[key] = {"expected": expected_value, "actual": actual}
    if int(payload.get("iteration", -1)) != 1500:
        mismatches["iteration"] = {"expected": 1500, "actual": payload.get("iteration")}
    if mismatches:
        raise RuntimeError(f"MF@3000 metadata mismatch: {json.dumps(mismatches, sort_keys=True)}")


def validate_direct3000_metadata(payload: dict) -> None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("Direct@3000 metadata mismatch: missing metadata")
    mismatches = {}
    if metadata.get("experiment_id") != EXPECTED_DIRECT_EXPERIMENT_ID:
        mismatches["experiment_id"] = {
            "expected": EXPECTED_DIRECT_EXPERIMENT_ID,
            "actual": metadata.get("experiment_id"),
        }
    if int(payload.get("iteration", -1)) != START_ABSOLUTE_UPDATE:
        mismatches["iteration"] = {
            "expected": START_ABSOLUTE_UPDATE,
            "actual": payload.get("iteration"),
        }
    if metadata.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        mismatches["manifest_sha256"] = {
            "expected": EXPECTED_MANIFEST_SHA256,
            "actual": metadata.get("manifest_sha256"),
        }
    if metadata.get("scorer_sha256") != EXPECTED_SCORER_SHA256:
        mismatches["scorer_sha256"] = {
            "expected": EXPECTED_SCORER_SHA256,
            "actual": metadata.get("scorer_sha256"),
        }
    if mismatches:
        raise RuntimeError(f"Direct@3000 metadata mismatch: {json.dumps(mismatches, sort_keys=True)}")


def mf_improvement_pct(direct: dict[str, float], mf: dict[str, float]) -> dict[str, float]:
    return {
        metric: (float(direct[metric]) - float(mf[metric])) / max(float(direct[metric]), 1e-12) * 100.0
        for metric in ("rel_l2", "tke", "mvpe")
    }


def restore_optimizer_state(optimizer: torch.optim.Optimizer, payload: dict) -> None:
    state = payload.get("optimizer_state_dict")
    if state is None:
        raise KeyError("continuation checkpoint lacks optimizer_state_dict")
    optimizer.load_state_dict(state)


def validate_contract(args: argparse.Namespace, core, mf01) -> tuple[list[Path], list[Path], dict, dict]:
    checks = {
        "manifest": (args.manifest, EXPECTED_MANIFEST_SHA256),
        "direct3000": (args.direct_checkpoint, EXPECTED_DIRECT3000_SHA256),
        "scorer": (args.kit_root / "scoring.py", EXPECTED_SCORER_SHA256),
    }
    mismatches = {}
    for name, (path, expected) in checks.items():
        actual = sha256(path)
        if actual != expected:
            mismatches[name] = {"path": str(path), "expected": expected, "actual": actual}
    if mismatches:
        raise RuntimeError(f"long-convergence contract checksum mismatch: {json.dumps(mismatches, sort_keys=True)}")
    if dict(mf01.N2_WEIGHTS) != N2_WEIGHTS:
        raise RuntimeError(f"N2 weights drifted: expected={N2_WEIGHTS}, actual={mf01.N2_WEIGHTS}")

    direct_payload = torch.load(args.direct_checkpoint, map_location="cpu", weights_only=False)
    mf_payload = torch.load(args.mf_checkpoint, map_location="cpu", weights_only=False)
    validate_direct3000_metadata(direct_payload)
    validate_mf3000_metadata(mf_payload)

    _, train_paths = core.read_manifest(args.manifest, "train")
    _, dev_paths = core.read_manifest(args.manifest, "dev")
    if len(train_paths) != 50 or len(dev_paths) != 16:
        raise RuntimeError(f"long-convergence requires frozen 50/16 split, got {len(train_paths)}/{len(dev_paths)}")
    return train_paths, dev_paths, direct_payload, mf_payload


def build_model(arm: str, kit_root: Path, builder, payload: dict, device: torch.device, mf01):
    if arm == "direct":
        model = mf01.cno_direct(kit_root, len(builder.feature_names), device)
    elif arm == "mf":
        model = mf01.MF01CNO(kit_root, len(builder.feature_names), device)
    else:
        raise ValueError(arm)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state, strict=True)
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise FloatingPointError(f"non-finite {arm} checkpoint")
    return model


def forward(model: nn.Module, builder, x: Tensor, mf01) -> Tensor:
    return mf01.forward(model, builder, x)


@torch.no_grad()
def evaluate(model: nn.Module, builder, paths: list[Path], args: argparse.Namespace, device: torch.device,
             core, mf01, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ds, loader = core.loader(paths, args, shuffle=False)
    predictions, targets, elapsed = [], [], 0.0
    model.eval()
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        prediction = forward(model, builder, x, mf01)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        predictions.append(prediction.cpu().numpy().astype(np.float32))
        targets.append(y.numpy().astype(np.float32))
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    scored = core.score_bundle(args.kit_root, prediction, target, elapsed / len(ds), out)
    rows, _ = core.trajectory_rows(ds, prediction, target, args.kit_root)
    with (out / "trajectory_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return scored | {"windows": len(ds), "trajectories": len(rows), "trajectory_rows": rows}


def assert_expected_start_metrics(arm: str, raw: dict[str, float]) -> None:
    failures = {}
    for metric, expected in EXPECTED_START_RAW[arm].items():
        actual = float(raw[metric])
        if abs(actual - expected) > START_METRIC_TOLERANCE:
            failures[metric] = {"expected_approx": expected, "actual": actual, "tolerance": START_METRIC_TOLERANCE}
    if failures:
        raise RuntimeError(f"{arm}@3000 dev parity mismatch: {json.dumps(failures, sort_keys=True)}")


def one_step_trainability_check(arm: str, model: nn.Module, builder, payload: dict, train_paths: list[Path],
                                args: argparse.Namespace, device: torch.device, core, mf01) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    restore_optimizer_state(optimizer, payload)
    loader_args = argparse.Namespace(batch_size=args.batch_size, workers=0, max_windows=None, seed=args.seed)
    _, loader = core.loader(train_paths, loader_args, shuffle=False)
    x, y, _, _ = next(iter(loader))
    x, y = x.to(device), y.to(device)
    first_parameter = next(model.parameters())
    before = first_parameter.detach().clone()
    model.train()
    prediction = forward(model, builder, x, mf01)
    if not torch.isfinite(prediction).all():
        raise FloatingPointError(f"non-finite {arm} prediction in preflight")
    parts = core.loss_parts(prediction, y)
    loss = sum(N2_WEIGHTS[name] * parts[name] for name in N2_WEIGHTS)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()
    parameter_delta = float((first_parameter.detach() - before).norm())
    if not np.isfinite(float(loss.detach())) or grad_norm <= 0 or parameter_delta <= 0:
        raise RuntimeError(
            f"{arm} preflight trainability failed: loss={float(loss.detach())}, grad={grad_norm}, delta={parameter_delta}"
        )
    return {
        "loss": float(loss.detach()),
        "grad_norm_before_clip": grad_norm,
        "first_parameter_delta": parameter_delta,
        "optimizer_state_entries": len(optimizer.state),
    }


def run_preflight(args: argparse.Namespace) -> None:
    if args.preflight_out is None:
        raise ValueError("--preflight-out is required with --preflight")
    if args.preflight_out.exists() and any(args.preflight_out.iterdir()):
        raise FileExistsError(args.preflight_out)
    args.preflight_out.mkdir(parents=True, exist_ok=True)
    core, mf01 = runtime_modules()
    set_seed(args.seed)
    train_paths, dev_paths, direct_payload, mf_payload = validate_contract(args, core, mf01)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    builder, feature_config = mf01.build_features(train_paths, device)

    evidence = {
        "passed": False,
        "manifest_sha256": sha256(args.manifest),
        "direct3000_sha256": sha256(args.direct_checkpoint),
        "mf3000_sha256": sha256(args.mf_checkpoint),
        "scorer_sha256": sha256(args.kit_root / "scoring.py"),
        "feature_names": list(builder.feature_names),
        "feature_config": vars(feature_config),
        "train_trajectories": 50,
        "dev_trajectories": 16,
        "device": str(device),
        "locked_final_accessed": False,
        "codabench": False,
    }

    for arm, payload in (("direct", direct_payload), ("mf", mf_payload)):
        set_seed(args.seed)
        model = build_model(arm, args.kit_root, builder, payload, device, mf01)
        start_eval = evaluate(model, builder, dev_paths, args, device, core, mf01,
                              args.preflight_out / f"eval_{arm}_3000")
        assert_expected_start_metrics(arm, start_eval["raw_errors"])
        set_seed(args.seed)
        model_for_step = build_model(arm, args.kit_root, builder, payload, device, mf01)
        trainability = one_step_trainability_check(
            arm, model_for_step, builder, payload, train_paths, args, device, core, mf01
        )
        evidence[arm] = {
            "start_raw_errors": start_eval["raw_errors"],
            "official_v9_subscores": start_eval["official_v9_subscores"],
            "trainability": trainability,
        }
        del model, model_for_step
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    evidence["passed"] = True
    save_json(args.preflight_out / "preflight.json", evidence)


def train_arm(arm: str, payload: dict, builder, train_paths: list[Path], dev_paths: list[Path],
              args: argparse.Namespace, device: torch.device, core, mf01, out_root: Path) -> dict:
    arm_out = out_root / arm
    arm_out.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    model = build_model(arm, args.kit_root, builder, payload, device, mf01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    restore_optimizer_state(optimizer, payload)

    start_eval = evaluate(model, builder, dev_paths, args, device, core, mf01, arm_out / "eval_03000")
    assert_expected_start_metrics(arm, start_eval["raw_errors"])
    history = [{"absolute_update": START_ABSOLUTE_UPDATE, "arm": arm, **start_eval["raw_errors"]}]
    evals = {START_ABSOLUTE_UPDATE: start_eval}

    train_ds, loader = core.loader(train_paths, args, shuffle=True)
    iterator = iter(loader)
    milestones = set(absolute_milestones(START_ABSOLUTE_UPDATE, ADDITIONAL_UPDATES, MILESTONE_INTERVAL))
    started = time.monotonic()
    for relative in range(1, ADDITIONAL_UPDATES + 1):
        try:
            x, y, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y, _, _ = next(iterator)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        model.train()
        prediction = forward(model, builder, x, mf01)
        parts = core.loss_parts(prediction, y)
        loss = sum(N2_WEIGHTS[name] * parts[name] for name in N2_WEIGHTS)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        absolute = START_ABSOLUTE_UPDATE + relative
        if absolute in milestones:
            result = evaluate(model, builder, dev_paths, args, device, core, mf01,
                              arm_out / f"eval_{absolute:05d}")
            row = {
                "absolute_update": absolute,
                "arm": arm,
                **result["raw_errors"],
                "mean_t_neural_s": result["mean_t_neural_s"],
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(row)
            evals[absolute] = result
            checkpoint_metadata = {
                "experiment_id": args.experiment_id,
                "arm": arm,
                "start_absolute_update": START_ABSOLUTE_UPDATE,
                "absolute_update": absolute,
                "seed": args.seed,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "workers": args.workers,
                "manifest_sha256": sha256(args.manifest),
                "source_checkpoint_sha256": sha256(args.direct_checkpoint if arm == "direct" else args.mf_checkpoint),
                "scorer_sha256": sha256(args.kit_root / "scoring.py"),
                "loss_weights": N2_WEIGHTS,
                "locked_final_accessed": False,
                "codabench": False,
            }
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "iteration": absolute,
                "metadata": checkpoint_metadata,
            }, arm_out / f"model_update_{absolute:05d}.pth")
            print(json.dumps(row, sort_keys=True), flush=True)

    with (arm_out / "update_curve.csv").open("w", newline="") as handle:
        fields = list(dict.fromkeys(key for row in history for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)
    save_json(arm_out / "summary.json", {
        "arm": arm,
        "train_windows": len(train_ds),
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
    })
    final_rows = evals[START_ABSOLUTE_UPDATE + ADDITIONAL_UPDATES]["trajectory_rows"]
    result = {"history": history, "evals": evals, "final_rows": final_rows}
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def trajectory_wins(direct_rows: list[dict], mf_rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
    direct = {row["trajectory_id"]: row for row in direct_rows}
    mf = {row["trajectory_id"]: row for row in mf_rows}
    if set(direct) != set(mf):
        raise RuntimeError("Direct/MF final trajectory sets differ")
    rows = []
    wins = {"rel_l2": 0, "tke": 0, "mvpe": 0}
    for name in sorted(direct):
        row = {"trajectory_id": name}
        for metric in wins:
            base = float(direct[name][metric])
            candidate = float(mf[name][metric])
            improvement = (base - candidate) / max(base, 1e-12) * 100.0
            row[f"direct_{metric}"] = base
            row[f"mf_{metric}"] = candidate
            row[f"mf_improvement_{metric}_pct"] = improvement
            row[f"mf_{metric}_win"] = improvement > 0.0
            wins[metric] += int(improvement > 0.0)
        rows.append(row)
    return rows, wins


def campaign_status(paired_rows: list[dict]) -> dict:
    final = paired_rows[-1]
    final_improvement = {metric: float(final[f"mf_improvement_{metric}_pct"]) for metric in ("rel_l2", "tke", "mvpe")}
    strong_at = []
    for row in paired_rows[1:]:
        rel = float(row["mf_improvement_rel_l2_pct"])
        tke = float(row["mf_improvement_tke_pct"])
        mvpe = float(row["mf_improvement_mvpe_pct"])
        if rel >= 3.0 and mvpe >= 3.0 and tke >= -2.0:
            strong_at.append(int(row["absolute_update"]))
    if final_improvement["rel_l2"] >= 3.0 and final_improvement["mvpe"] >= 3.0 and final_improvement["tke"] >= -2.0:
        status = "PERSISTENT_STRONG"
    elif sum(final_improvement[m] >= 3.0 for m in final_improvement) >= 2 and min(final_improvement.values()) >= -2.0:
        status = "PERSISTENT_SUPPORTIVE"
    else:
        status = "WASHED_OUT_OR_MIXED"
    return {
        "status": status,
        "final_improvement_pct": final_improvement,
        "strong_gate_absolute_updates": strong_at,
        "strong_gate_definition": {"rel_l2_min_pct": 3.0, "mvpe_min_pct": 3.0, "tke_min_pct": -2.0},
        "decision_owner": "ChatGPT/Sol review required",
    }


def run(args: argparse.Namespace) -> None:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(args.out_dir)
    if args.preflight_evidence is None:
        raise ValueError("--preflight-evidence is required for formal long-convergence training")
    preflight = json.loads(args.preflight_evidence.read_text(encoding="utf-8"))
    if not preflight.get("passed"):
        raise RuntimeError("refusing long-convergence training: preflight did not pass")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    core, mf01 = runtime_modules()
    set_seed(args.seed)
    train_paths, dev_paths, direct_payload, mf_payload = validate_contract(args, core, mf01)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    builder, feature_config = mf01.build_features(train_paths, device)
    metadata = {
        "experiment_id": args.experiment_id,
        "purpose": "matched Direct-vs-MF convergence from absolute 3000 to 15000",
        "start_absolute_update": START_ABSOLUTE_UPDATE,
        "additional_updates_per_arm": ADDITIONAL_UPDATES,
        "final_absolute_update": START_ABSOLUTE_UPDATE + ADDITIONAL_UPDATES,
        "absolute_milestones": absolute_milestones(START_ABSOLUTE_UPDATE, ADDITIONAL_UPDATES, MILESTONE_INTERVAL),
        "arm_order": ["direct", "mf"],
        "data_order": "each arm creates an independently seeded shuffle loader with the same seed",
        "seed": args.seed,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "loss_weights": N2_WEIGHTS,
        "manifest_sha256": sha256(args.manifest),
        "direct3000_sha256": sha256(args.direct_checkpoint),
        "mf3000_sha256": sha256(args.mf_checkpoint),
        "scorer_sha256": sha256(args.kit_root / "scoring.py"),
        "feature_names": list(builder.feature_names),
        "feature_config": vars(feature_config),
        "train_trajectories": len(train_paths),
        "dev_trajectories": len(dev_paths),
        "preflight_evidence": str(args.preflight_evidence),
        "locked_final_accessed": False,
        "codabench": False,
        "device": str(device),
        "runner_sha256": sha256(Path(__file__)),
    }
    save_json(args.out_dir / "run_metadata.json", metadata)

    direct_result = train_arm(
        "direct", direct_payload, builder, train_paths, dev_paths, args, device, core, mf01, args.out_dir
    )
    mf_result = train_arm(
        "mf", mf_payload, builder, train_paths, dev_paths, args, device, core, mf01, args.out_dir
    )

    direct_by_update = {int(row["absolute_update"]): row for row in direct_result["history"]}
    mf_by_update = {int(row["absolute_update"]): row for row in mf_result["history"]}
    if set(direct_by_update) != set(mf_by_update):
        raise RuntimeError("Direct/MF checkpoint curves do not have the same updates")
    paired_rows = []
    for update in sorted(direct_by_update):
        direct_raw = {metric: direct_by_update[update][metric] for metric in ("rel_l2", "tke", "mvpe")}
        mf_raw = {metric: mf_by_update[update][metric] for metric in ("rel_l2", "tke", "mvpe")}
        improvement = mf_improvement_pct(direct_raw, mf_raw)
        paired_rows.append({
            "absolute_update": update,
            **{f"direct_{metric}": float(direct_raw[metric]) for metric in direct_raw},
            **{f"mf_{metric}": float(mf_raw[metric]) for metric in mf_raw},
            **{f"mf_improvement_{metric}_pct": float(improvement[metric]) for metric in improvement},
        })
    with (args.out_dir / "paired_convergence.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

    trajectory_rows, wins = trajectory_wins(direct_result["final_rows"], mf_result["final_rows"])
    with (args.out_dir / "final_trajectory_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectory_rows[0]))
        writer.writeheader()
        writer.writerows(trajectory_rows)

    status = campaign_status(paired_rows)
    status["final_trajectory_wins"] = wins
    save_json(args.out_dir / "gate_result.json", status)
    save_json(args.out_dir / "summary.json", {
        "metadata": metadata,
        "paired_convergence": paired_rows,
        "gate": status,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", default="T1-ID-MF-LONG-CONVERGENCE-S20260906")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--direct-checkpoint", type=Path, required=True)
    parser.add_argument("--mf-checkpoint", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--start-update", type=int, default=START_ABSOLUTE_UPDATE)
    parser.add_argument("--additional-updates", type=int, default=ADDITIONAL_UPDATES)
    parser.add_argument("--milestone-interval", type=int, default=MILESTONE_INTERVAL)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-out", type=Path)
    parser.add_argument("--preflight-evidence", type=Path)
    args = parser.parse_args()

    if args.seed != SEED or args.lr != 1e-5 or args.batch_size != 8 or args.workers != 2:
        raise ValueError("campaign seed/lr/batch/workers are frozen at 20260901/1e-5/8/2")
    absolute_milestones(args.start_update, args.additional_updates, args.milestone_interval)
    if args.preflight:
        run_preflight(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
