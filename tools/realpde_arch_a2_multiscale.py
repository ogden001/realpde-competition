#!/usr/bin/env python3
"""Track 1 Architecture A2: Direct CNO + coarse-scale residual branch.

The global path is the registered P0-A Direct CNO. The only experimental
variable is a lightweight residual branch that downsamples runtime-safe Past20
u/v by 2x in space, models it on the coarse grid, then upsamples a Future20 u/v
residual back to 32x64. The final projection is zero-initialized so update 0 is
exactly the supplied Direct checkpoint.
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
import torch.nn.functional as F
from torch import Tensor, nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SEED = 20260901
START_UPDATE = 1500
RELATIVE_UPDATES = 1500
EXPECTED_MANIFEST_SHA256 = "42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347"
EXPECTED_START_CHECKPOINT_SHA256 = "5499e60a3b8146bf095070dc76d03c85eae57b5f1ec794444276bab362458ec4"
EXPECTED_REFERENCE_CHECKPOINT_SHA256 = "9fa52c905c2603179da39e90da7689a1d65005d5c00d0df2d9fe2e310cf32aeb"
EXPECTED_SCORER_SHA256 = "a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39"
N2_WEIGHTS = {"mse": 1.0, "tke": 0.05, "rel": 0.027514, "mvpe": 0.009757}


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


class CoarseResidualBranch(nn.Module):
    """Lightweight 2x-spatial-coarse residual operator over Past20 u/v only."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.input = nn.Conv3d(2, hidden, kernel_size=(3, 5, 5), padding=(1, 2, 2))
        self.middle = nn.Conv3d(hidden, hidden, kernel_size=3, padding=1)
        self.activation = nn.GELU()
        self.output = nn.Conv3d(hidden, 2, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.last_coarse_shape: tuple[int, int, int] | None = None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5 or x.shape[1] != 20 or x.shape[-1] < 2:
            raise ValueError(f"expected [B,20,H,W,C>=2], got {tuple(x.shape)}")
        if x.shape[2] < 2 or x.shape[3] < 2:
            raise ValueError(f"spatial grid too small for 2x coarse branch: {tuple(x.shape)}")
        uv = x[..., :2].permute(0, 4, 1, 2, 3)
        coarse = F.avg_pool3d(uv, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.last_coarse_shape = tuple(int(v) for v in coarse.shape[2:])
        hidden = self.activation(self.input(coarse))
        hidden = self.activation(self.middle(hidden))
        residual = self.output(hidden)
        b, c, t, hc, wc = residual.shape
        h, w = x.shape[2], x.shape[3]
        frames = residual.permute(0, 2, 1, 3, 4).reshape(b * t, c, hc, wc)
        frames = F.interpolate(frames, size=(h, w), mode="bilinear", align_corners=False)
        return frames.reshape(b, t, c, h, w).permute(0, 1, 3, 4, 2)


def fuse_prediction(global_prediction: Tensor, x: Tensor, branch: CoarseResidualBranch) -> Tensor:
    if global_prediction.ndim != 5 or global_prediction.shape[-1] < 3:
        raise ValueError(f"expected global prediction [B,20,H,W,C>=3], got {tuple(global_prediction.shape)}")
    residual = branch(x)
    if residual.shape != global_prediction[..., :2].shape:
        raise ValueError(f"residual/global shape mismatch: {tuple(residual.shape)} vs {tuple(global_prediction.shape)}")
    output = global_prediction.clone()
    output[..., :2] = global_prediction[..., :2] + residual
    return output


def optimizer_audit(global_model: nn.Module, branch: CoarseResidualBranch, optimizer: torch.optim.Optimizer) -> dict:
    global_params = [p for p in global_model.parameters() if p.requires_grad]
    branch_params = [p for p in branch.parameters() if p.requires_grad]
    expected = global_params + branch_params
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    ids = [id(p) for p in actual]
    expected_ids = {id(p) for p in expected}
    global_ids = {id(p) for p in global_params}
    branch_ids = {id(p) for p in branch_params}
    return {
        "global_parameter_tensors": len(global_params),
        "coarse_parameter_tensors": len(branch_params),
        "optimizer_global_parameter_tensors": sum(id(p) in global_ids for p in actual),
        "optimizer_coarse_parameter_tensors": sum(id(p) in branch_ids for p in actual),
        "missing_params": len(expected_ids - set(ids)),
        "duplicate_params": len(ids) - len(set(ids)),
        "passed": set(ids) == expected_ids and len(ids) == len(set(ids)),
    }


def make_optimizer(global_model: nn.Module, branch: CoarseResidualBranch, lr: float) -> torch.optim.Optimizer:
    params = [p for p in global_model.parameters() if p.requires_grad] + [p for p in branch.parameters() if p.requires_grad]
    if len(params) != len({id(p) for p in params}):
        raise RuntimeError("duplicate trainable parameters")
    return torch.optim.AdamW(params, lr=lr)


def restore_global_optimizer_state(optimizer: torch.optim.Optimizer, checkpoint_payload: dict, global_model: nn.Module, lr: float) -> None:
    state = checkpoint_payload.get("optimizer_state_dict")
    if state is None:
        raise KeyError("matched A2 continuation requires optimizer_state_dict in Direct@1500 checkpoint")
    direct_optimizer = torch.optim.AdamW(global_model.parameters(), lr=lr)
    direct_optimizer.load_state_dict(state)
    for param in global_model.parameters():
        if param in direct_optimizer.state:
            optimizer.state[param] = {
                key: value.detach().clone() if torch.is_tensor(value) else value
                for key, value in direct_optimizer.state[param].items()
            }


def aggregate_improvements(baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    result = {}
    for metric in ("rel_l2", "tke", "mvpe"):
        base = float(baseline[metric])
        result[metric] = (base - float(candidate[metric])) / max(base, 1e-12) * 100.0
    return result


def classify_gate(baseline: dict[str, float], candidate: dict[str, float]) -> dict:
    improvements = aggregate_improvements(baseline, candidate)
    promising = (
        improvements["rel_l2"] >= 3.0
        and improvements["mvpe"] >= 3.0
        and improvements["tke"] >= -2.0
    )
    if promising:
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
    rows = []
    wins = {"rel_l2": 0, "tke": 0, "mvpe": 0}
    for name in sorted(reference):
        out = {"trajectory_id": name}
        for metric in wins:
            base = float(reference[name][metric])
            value = float(candidate[name][metric])
            improvement = (base - value) / max(base, 1e-12) * 100.0
            out[f"reference_{metric}"] = base
            out[f"a2_{metric}"] = value
            out[f"improvement_{metric}_pct"] = improvement
            out[f"{metric}_win"] = improvement > 0.0
            wins[metric] += int(improvement > 0.0)
        rows.append(out)
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
        raise RuntimeError(f"A2 contract checksum mismatch: {json.dumps(mismatches, sort_keys=True)}")
    if dict(direct.N2_WEIGHTS) != N2_WEIGHTS:
        raise RuntimeError(f"N2 weights drifted: expected={N2_WEIGHTS}, actual={direct.N2_WEIGHTS}")
    _, train_paths = core.read_manifest(args.manifest, "train")
    _, dev_paths = core.read_manifest(args.manifest, "dev")
    if len(train_paths) != 50 or len(dev_paths) != 16:
        raise RuntimeError(f"A2 requires frozen 50/16 split, got {len(train_paths)}/{len(dev_paths)}")
    return train_paths, dev_paths


def build_features_and_global(kit_root: Path, checkpoint: Path, train_paths: list[Path], device: torch.device, direct):
    builder, feature_config = direct.build_features(train_paths, device)
    model = direct.cno_direct(kit_root, len(builder.feature_names), device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    input_key = "lift.inter_CNOBlock.convolution.weight"
    if input_key in state and state[input_key].shape[1] == len(builder.feature_names):
        model.load_state_dict(state, strict=True)
    else:
        direct.adapt_input_weight(model, payload, len(builder.feature_names))
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("non-finite Direct checkpoint")
    return builder, feature_config, model, payload


def load_global_with_builder(kit_root: Path, checkpoint: Path, builder, device: torch.device, direct):
    model = direct.cno_direct(kit_root, len(builder.feature_names), device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    input_key = "lift.inter_CNOBlock.convolution.weight"
    if input_key in state and state[input_key].shape[1] == len(builder.feature_names):
        model.load_state_dict(state, strict=True)
    else:
        direct.adapt_input_weight(model, payload, len(builder.feature_names))
    return model


def forward_global(model: nn.Module, builder, x: Tensor) -> Tensor:
    features = builder(x)
    return model(features.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)


def forward(model: nn.Module, builder, x: Tensor, branch: CoarseResidualBranch | None = None) -> Tensor:
    global_prediction = forward_global(model, builder, x)
    return global_prediction if branch is None else fuse_prediction(global_prediction, x, branch)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


@torch.no_grad()
def evaluate(model: nn.Module, builder, branch: CoarseResidualBranch | None, paths: list[Path], args: argparse.Namespace,
             device: torch.device, core, out: Path, save_predictions: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ds, loader = core.loader(paths, args, shuffle=False)
    model.eval()
    if branch is not None:
        branch.eval()
    predictions, targets, residuals = [], [], []
    elapsed = 0.0
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        prediction = forward(model, builder, x, branch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        predictions.append(prediction.cpu().numpy().astype(np.float32))
        targets.append(y.numpy().astype(np.float32))
        if branch is not None and save_predictions:
            residuals.append(branch(x).cpu().numpy().astype(np.float32))
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    scored = core.score_bundle(args.kit_root, prediction, target, elapsed / len(ds), out)
    rows, _ = core.trajectory_rows(ds, prediction, target, args.kit_root)
    with (out / "trajectory_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if save_predictions:
        payload = {"prediction": prediction, "target": target}
        if residuals:
            payload["coarse_residual_uv"] = np.concatenate(residuals)
        np.savez_compressed(out / "predictions.npz", **payload)
    return scored | {"windows": len(ds), "trajectories": len(rows), "trajectory_rows": rows}


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
    builder, _, model, payload = build_features_and_global(args.kit_root, args.checkpoint, train_paths, device, direct)
    branch = CoarseResidualBranch(args.coarse_hidden).to(device)
    optimizer = make_optimizer(model, branch, args.lr)
    restore_global_optimizer_state(optimizer, payload, model, args.lr)
    audit = optimizer_audit(model, branch, optimizer)
    if not audit["passed"]:
        raise RuntimeError(f"optimizer audit failed: {audit}")

    loader_args = argparse.Namespace(batch_size=args.batch_size, workers=0, max_windows=None, seed=args.seed)
    _, loader = core.loader(train_paths, loader_args, shuffle=False)
    x, y, _, _ = next(iter(loader))
    x, y = x.to(device), y.to(device)
    model.eval(); branch.eval()
    with torch.no_grad():
        global0 = forward_global(model, builder, x)
        residual0 = branch(x)
        fused0 = fuse_prediction(global0, x, branch)
    step0 = {
        "full_equals_global": bool(torch.equal(fused0, global0)),
        "pressure_exact": bool(torch.equal(fused0[..., 2], global0[..., 2])),
        "residual_zero": bool(residual0.abs().max() == 0),
        "coarse_shape": branch.last_coarse_shape,
    }
    if not all(step0[key] for key in ("full_equals_global", "pressure_exact", "residual_zero")):
        raise RuntimeError(f"zero-init parity failed: {step0}")

    initial_input = branch.input.weight.detach().clone()
    initial_middle = branch.middle.weight.detach().clone()
    initial_output = branch.output.weight.detach().clone()
    step_metrics = []
    model.train(); branch.train()
    for step in range(1, 4):
        prediction = forward(model, builder, x, branch)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("non-finite A2 prediction during preflight")
        parts = core.loss_parts(prediction, y)
        loss = sum(N2_WEIGHTS[name] * parts[name] for name in N2_WEIGHTS)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == 1:
            final_grad = float(branch.output.weight.grad.norm())
            if final_grad <= 0:
                raise RuntimeError("zero gradient on A2 final projection")
        optimizer.step()
        step_metrics.append({
            "step": step,
            "loss": float(loss.detach()),
            "output_weight_delta": float((branch.output.weight.detach() - initial_output).norm()),
            "input_weight_delta": float((branch.input.weight.detach() - initial_input).norm()),
            "middle_weight_delta": float((branch.middle.weight.detach() - initial_middle).norm()),
        })
    if step_metrics[-1]["input_weight_delta"] <= 0 or step_metrics[-1]["middle_weight_delta"] <= 0:
        raise RuntimeError("upstream A2 coarse convolutions did not update after three steps")

    branch.eval()
    with torch.no_grad():
        changed = x.clone()
        if changed.shape[-1] > 2:
            changed[..., 2] = changed[..., 2] + 1234.0
        uv_only_max_abs = float((branch(x) - branch(changed)).abs().max())
    if uv_only_max_abs != 0.0:
        raise RuntimeError(f"A2 branch depends on non-u/v input: max_abs={uv_only_max_abs}")

    checkpoint = args.preflight_out / "roundtrip.pth"
    torch.save(branch.state_dict(), checkpoint)
    reloaded = CoarseResidualBranch(args.coarse_hidden).to(device)
    reloaded.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
    with torch.no_grad():
        roundtrip_error = float((reloaded(x) - branch(x)).abs().max())
    if roundtrip_error != 0.0:
        raise RuntimeError(f"A2 checkpoint roundtrip failed: {roundtrip_error}")

    save_json(args.preflight_out / "preflight.json", {
        "passed": True,
        "optimizer_audit": audit,
        "zero_init": step0,
        "first_step_final_projection_grad_norm": final_grad,
        "three_step_training": step_metrics,
        "uv_only_max_abs_when_pressure_changes": uv_only_max_abs,
        "checkpoint_roundtrip_max_abs": roundtrip_error,
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
        raise ValueError("--preflight-evidence is required for formal A2 training")
    preflight = json.loads(args.preflight_evidence.read_text(encoding="utf-8"))
    if not preflight.get("passed"):
        raise RuntimeError("refusing A2 training: preflight did not pass")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    core, direct = runtime_modules()
    set_seed(args.seed)
    train_paths, dev_paths = validate_contract(args, core, direct)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    builder, feature_config, model, start_payload = build_features_and_global(
        args.kit_root, args.checkpoint, train_paths, device, direct
    )
    reference_model = load_global_with_builder(args.kit_root, args.reference_checkpoint, builder, device, direct)
    branch = CoarseResidualBranch(args.coarse_hidden).to(device)
    optimizer = make_optimizer(model, branch, args.lr)
    restore_global_optimizer_state(optimizer, start_payload, model, args.lr)
    audit = optimizer_audit(model, branch, optimizer)
    if not audit["passed"]:
        raise RuntimeError(f"optimizer audit failed before formal A2 training: {audit}")

    metadata = {
        "experiment_id": args.experiment_id,
        "architecture": "P0-A Direct CNO + 2x coarse Past20-u/v residual branch",
        "coarse_operator": f"avgpool2x + Conv3d(2,{args.coarse_hidden},3x5x5) + GELU + Conv3d({args.coarse_hidden},{args.coarse_hidden},3x3x3) + GELU + zero-init 1x1 -> bilinear upsample",
        "fusion": "prediction_uv = global_uv + coarse_residual_uv; pressure = global_pressure exactly",
        "global_input": list(builder.feature_names),
        "coarse_input": "Past20 runtime-safe raw u/v only",
        "feature_config": vars(feature_config),
        "loss_weights": N2_WEIGHTS,
        "seed": args.seed,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "start_absolute_update": args.start_update,
        "relative_updates": args.updates,
        "final_absolute_update": args.start_update + args.updates,
        "milestones": list(args.milestones),
        "manifest_sha256": sha256(args.manifest),
        "start_checkpoint_sha256": sha256(args.checkpoint),
        "reference_checkpoint_sha256": sha256(args.reference_checkpoint),
        "scorer_sha256": sha256(args.kit_root / "scoring.py"),
        "train_trajectories": len(train_paths),
        "dev_trajectories": len(dev_paths),
        "global_parameters": sum(p.numel() for p in model.parameters()),
        "coarse_parameters": sum(p.numel() for p in branch.parameters()),
        "optimizer_audit": audit,
        "preflight_evidence": str(args.preflight_evidence),
        "locked_final_accessed": False,
        "codabench": False,
        "device": str(device),
        "runner_sha256": sha256(Path(__file__)),
    }
    save_json(args.out_dir / "run_metadata.json", metadata)

    reference = evaluate(reference_model, builder, None, dev_paths, args, device, core,
                         args.out_dir / "eval_reference_direct3000", save_predictions=False)
    start_eval = evaluate(model, builder, None, dev_paths, args, device, core,
                          args.out_dir / "eval_start_direct1500", save_predictions=False)
    history = [
        {"absolute_update": args.start_update, "model": "Direct_start", **start_eval["raw_errors"]},
        {"absolute_update": args.start_update + args.updates, "model": "Direct_reference", **reference["raw_errors"]},
    ]

    train_ds, loader = core.loader(train_paths, args, shuffle=True)
    iterator = iter(loader)
    started = time.monotonic()
    final_result = None
    for relative in range(1, args.updates + 1):
        try:
            x, y, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y, _, _ = next(iterator)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        model.train(); branch.train()
        prediction = forward(model, builder, x, branch)
        parts = core.loss_parts(prediction, y)
        loss = sum(N2_WEIGHTS[name] * parts[name] for name in N2_WEIGHTS)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(branch.parameters()), 1.0)
        optimizer.step()

        if relative in set(args.milestones) or relative == args.updates:
            absolute = args.start_update + relative
            result = evaluate(model, builder, branch, dev_paths, args, device, core,
                              args.out_dir / f"eval_a2_{absolute:05d}", save_predictions=relative == args.updates)
            row = {
                "absolute_update": absolute,
                "model": "A2",
                **result["raw_errors"],
                "mean_t_neural_s": result["mean_t_neural_s"],
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(row)
            torch.save({
                "model_state_dict": model.state_dict(),
                "coarse_state_dict": branch.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "iteration": absolute,
                "metadata": metadata,
            }, args.out_dir / f"model_update_{absolute:05d}.pth")
            print(json.dumps(row, sort_keys=True), flush=True)
            if relative == args.updates:
                final_result = result

    assert final_result is not None
    torch.save({
        "model_state_dict": model.state_dict(),
        "coarse_state_dict": branch.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": args.start_update + args.updates,
        "metadata": metadata,
    }, args.out_dir / "model_last.pth")

    trajectory_comparison, wins = compare_trajectory_metrics(reference["trajectory_rows"], final_result["trajectory_rows"])
    with (args.out_dir / "trajectory_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectory_comparison[0]))
        writer.writeheader()
        writer.writerows(trajectory_comparison)
    gate = classify_gate(reference["raw_errors"], final_result["raw_errors"])
    gate["trajectory_wins"] = wins
    gate["reference_raw_errors"] = reference["raw_errors"]
    gate["candidate_raw_errors"] = final_result["raw_errors"]
    gate["reference_checkpoint_sha256"] = EXPECTED_REFERENCE_CHECKPOINT_SHA256
    gate["candidate_absolute_update"] = args.start_update + args.updates
    save_json(args.out_dir / "gate_result.json", gate)

    with (args.out_dir / "update_curve.csv").open("w", newline="") as handle:
        fields = list(dict.fromkeys(key for row in history for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)
    save_json(args.out_dir / "summary.json", {
        "metadata": metadata | {"train_windows": len(train_ds), "elapsed_seconds": time.monotonic() - started},
        "history": history,
        "gate": gate,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Exact matched Direct@1500 checkpoint")
    parser.add_argument("--reference-checkpoint", type=Path, required=True, help="Exact matched Direct@3000 checkpoint")
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--start-update", type=int, default=START_UPDATE)
    parser.add_argument("--updates", type=int, default=RELATIVE_UPDATES)
    parser.add_argument("--milestones", type=int, nargs="+", default=[500, 1000, 1500])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--coarse-hidden", type=int, default=32)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-out", type=Path)
    parser.add_argument("--preflight-evidence", type=Path)
    args = parser.parse_args()
    if args.start_update != START_UPDATE or args.updates != RELATIVE_UPDATES:
        raise ValueError(f"A2 screening is frozen at Direct@{START_UPDATE} + {RELATIVE_UPDATES} updates")
    if list(args.milestones) != [500, 1000, 1500]:
        raise ValueError("A2 screening milestones are frozen at 500/1000/1500 relative updates")
    if args.preflight:
        run_preflight(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
