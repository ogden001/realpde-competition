#!/usr/bin/env python3
"""Matched low-memory C0/V1 paired final validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

FIXED_LAMBDA_VORT = 15.5385751724
START_UPDATE = 3000
FINAL_UPDATE = 12000
MILESTONES = (3000, 4500, 6000, 9000, 12000)
EXPECTED_START = {
    "C0": {"rel_l2": 0.1613427249, "tke": 0.5579713427, "mvpe": 0.1314960284},
    "V1": {"rel_l2": 0.1560233235, "tke": 0.5555717349, "mvpe": 0.1215854809},
}


def validate_lowmem_protocol(*, micro_batch: int, accumulation_steps: int,
                             effective_batch: int, memory_cap_gib: float, seed: int,
                             lr: float, start_update: int, final_update: int,
                             lambda_vort: float, arm: str) -> dict:
    if arm not in {"C0", "V1"}:
        raise ValueError("low-memory arms must be C0 or V1")
    if micro_batch < 1 or accumulation_steps < 1 or micro_batch * accumulation_steps != effective_batch:
        raise ValueError("micro_batch * accumulation_steps must equal effective_batch")
    if effective_batch != 8:
        raise ValueError("effective batch must remain 8")
    if not 0.0 < memory_cap_gib <= 12.0:
        raise ValueError("low-memory cap must be in (0, 12] GiB")
    if seed != 20260901 or lr != 1e-5:
        raise ValueError("seed 20260901 and AdamW lr=1e-5 are fixed")
    if start_update != START_UPDATE or final_update != FINAL_UPDATE:
        raise ValueError("low-memory continuation must be absolute 3000 -> 12000")
    expected_lambda = FIXED_LAMBDA_VORT if arm == "V1" else 0.0
    if lambda_vort != expected_lambda:
        raise ValueError(f"{arm} lambda_vort is not fixed: expected {expected_lambda}")
    return {"micro_batch": micro_batch, "accumulation_steps": accumulation_steps,
            "effective_batch": effective_batch, "memory_cap_gib": memory_cap_gib}


def is_owned_old_vorticity_command(command: str) -> bool:
    """Match only this project's old Vorticity runner, not shared GPU work."""
    markers = (
        "realpde_vorticity_long_final.py",
        "vorticity_long_final_20260914",
        "V1-LONG",
        "C0-LONG",
    )
    return any(marker in command for marker in markers)


def window_order_digest(loader, seed: int) -> str:
    order = list(iter(loader.sampler))
    return hashlib.sha256(np.asarray(order, dtype=np.int64).tobytes() + str(seed).encode("ascii")).hexdigest()


def configure_cuda_memory_cap(device: torch.device, memory_cap_gib: float) -> int:
    if device.type != "cuda":
        return 0
    total = torch.cuda.get_device_properties(device).total_memory
    cap = int(memory_cap_gib * 1024 ** 3)
    if cap > total:
        raise ValueError("memory cap exceeds GPU total memory")
    torch.cuda.set_per_process_memory_fraction(cap / total, torch.cuda.current_device())
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return cap


def _load_model(kit_root: Path, checkpoint: Path, train_paths: list[Path], device: torch.device):
    import realpde_mf01 as direct

    builder, config = direct.build_features(train_paths, device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("iteration", -1)) != START_UPDATE:
        raise ValueError(f"checkpoint iteration is not {START_UPDATE}: {checkpoint}")
    model = direct.cno_direct(kit_root, len(builder.feature_names), device)
    state = payload.get("model_state_dict", payload)
    key = "lift.inter_CNOBlock.convolution.weight"
    if key in state and state[key].shape[1] == len(builder.feature_names):
        model.load_state_dict(state, strict=True)
    else:
        direct.adapt_input_weight(model, payload, len(builder.feature_names))
    return model, builder, config, payload


def _forward_zero_pressure(model, builder, x: torch.Tensor) -> torch.Tensor:
    import realpde_mf01 as direct

    pred = direct.forward(model, builder, x).clone()
    pred[..., 2] = 0.0
    return pred


def _loss(pred: torch.Tensor, target: torch.Tensor, arm: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    import realpde_loss_official_v9 as core
    import realpde_mf01 as direct
    from realpde_structured_temporal_dynamics import vorticity

    parts = core.loss_parts(pred, target)
    total = sum(direct.N2_WEIGHTS[name] * parts[name] for name in direct.N2_WEIGHTS)
    if arm == "V1":
        parts["vorticity"] = (vorticity(pred, dx=1.0, dy=1.0) - vorticity(target, dx=1.0, dy=1.0)).square().mean()
        total = total + FIXED_LAMBDA_VORT * parts["vorticity"]
    return total, parts


@torch.no_grad()
def evaluate(model, builder, paths: list[Path], args: argparse.Namespace, device: torch.device,
             out: Path, arm: str, update: int, save_predictions: bool) -> dict:
    import realpde_loss_official_v9 as core
    import realpde_structured_temporal_dynamics as diagnostics

    out.mkdir(parents=True, exist_ok=True)
    ds, loader = core.loader(paths, args, shuffle=False)
    model.eval()
    predictions, targets, elapsed = [], [], 0.0
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        pred = _forward_zero_pressure(model, builder, x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        predictions.append(pred.cpu().numpy().astype(np.float32))
        targets.append(y.numpy().astype(np.float32))
    pred, target = np.concatenate(predictions), np.concatenate(targets)
    if not np.isfinite(pred).all() or not np.isfinite(target).all() or float(np.abs(pred[..., 2]).max()) != 0.0:
        raise FloatingPointError("nonfinite prediction/target or nonzero pressure")
    scored = core.score_bundle(args.kit_root, pred, target, elapsed / len(ds), out)
    diagnostics.diagnostics(ds, pred, target, out, arm, update, args.kit_root)
    if save_predictions:
        np.savez_compressed(out / "predictions.npz", prediction=pred[..., :2], target=target[..., :2],
                            trajectory=np.asarray([ref.path.name for ref in ds.refs]),
                            window_start=np.asarray([ref.start for ref in ds.refs]))
    result = scored | {"windows": len(ds), "update": update, "arm": arm}
    (out / "scores.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _write_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                      update: int, preflight: dict) -> None:
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "iteration": update, "preflight": preflight}, path)


def _read_old_task_status(path: Path) -> dict:
    status = json.loads(path.read_text(encoding="utf-8"))
    if not status.get("old_task_stopped") or status.get("old_v1_auto_launch_remaining"):
        raise RuntimeError("old Vorticity task has not been fully stopped")
    return status


def run(args: argparse.Namespace) -> None:
    import realpde_loss_official_v9 as core
    from realpde_structured_temporal_dynamics import write_rows

    config_batch = validate_lowmem_protocol(
        micro_batch=args.micro_batch, accumulation_steps=args.accumulation_steps,
        effective_batch=args.effective_batch, memory_cap_gib=args.memory_cap_gib,
        seed=args.seed, lr=args.lr, start_update=START_UPDATE,
        final_update=args.final_update, lambda_vort=args.lambda_vort, arm=args.arm)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    old_status = _read_old_task_status(args.old_task_status)
    core.set_seed(args.seed)
    manifest, train_paths = core.read_manifest(args.manifest, "train")
    _, dev_paths = core.read_manifest(args.manifest, "dev")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("low-memory final requires the target CUDA GPU")
    memory_cap_bytes = configure_cuda_memory_cap(device, args.memory_cap_gib)
    model, builder, feature_config, payload = _load_model(args.kit_root, args.checkpoint, train_paths, device)
    args.batch_size = args.micro_batch

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    state_before = len(optimizer.state)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    state_after = len(optimizer.state)
    if state_after <= 0:
        raise RuntimeError("R2 optimizer state did not restore")
    if any(abs(group["lr"] - args.lr) > 1e-15 for group in optimizer.param_groups):
        raise RuntimeError("restored optimizer lr differs from frozen lr")

    # Every preflight loader is separate. The formal training loader is created last,
    # so preflight iteration cannot consume its shuffle generator.
    _, probe_loader = core.loader(train_paths, args, shuffle=True)
    probe_order_digest = window_order_digest(probe_loader, args.seed)
    probe_x, probe_y, _, _ = next(iter(probe_loader))
    probe_x, probe_y = probe_x.to(device), probe_y.to(device)
    model.eval()
    probe_pred = _forward_zero_pressure(model, builder, probe_x)
    if not torch.isfinite(probe_pred).all() or float(probe_pred[..., 2].abs().max()) != 0.0:
        raise FloatingPointError("initial prediction preflight failed")
    replay = evaluate(model, builder, dev_paths, args, device, args.out_dir / "eval_03000", args.arm, 3000, False)
    expected = EXPECTED_START[args.arm]
    replay_delta = {name: float(replay["raw_errors"][name] - expected[name]) for name in expected}
    if any(abs(value) > args.start_tolerance for value in replay_delta.values()):
        raise RuntimeError(f"R2@3000 replay mismatch for {args.arm}: {replay_delta}")
    preflight_peak_allocated = int(torch.cuda.max_memory_allocated())
    preflight_peak_reserved = int(torch.cuda.max_memory_reserved())
    if preflight_peak_reserved > memory_cap_bytes:
        raise RuntimeError(f"preflight reserved memory exceeded {args.memory_cap_gib} GiB")

    _, order_loader = core.loader(train_paths, args, shuffle=True)
    train_order_digest = window_order_digest(order_loader, args.seed)
    if probe_order_digest != train_order_digest:
        raise RuntimeError("seeded train DataLoader window ordering is not reproducible")
    preflight = {
        "status": "REVIEW_REQUIRED", "arm": args.arm, "start_update": START_UPDATE,
        "checkpoint_iteration": int(payload["iteration"]), "checkpoint_sha256": core.sha256(args.checkpoint),
        "manifest_sha256": core.sha256(args.manifest), "scorer_sha256": core.sha256(args.kit_root / "scoring.py"),
        "r2_replay_raw_errors": replay["raw_errors"], "r2_replay_delta": replay_delta,
        "optimizer_state_before_restore": state_before, "optimizer_state_after_restore": state_after,
        "batch": config_batch, "seed": args.seed, "window_protocol": "T_in=20,T_out=20,stride=20,seeded_train_loader_generator",
        "train_window_order_digest": train_order_digest, "probe_window_order_digest": probe_order_digest,
        "dataloader_order_match": probe_order_digest == train_order_digest,
        "lambda_vort": FIXED_LAMBDA_VORT if args.arm == "V1" else 0.0,
        "pressure_max_abs": float(probe_pred[..., 2].abs().max()), "finite": bool(torch.isfinite(probe_pred).all()),
        "memory_cap_bytes": memory_cap_bytes, "gpu_model": torch.cuda.get_device_name(device),
        "preflight_peak_gpu_memory_allocated": preflight_peak_allocated,
        "preflight_peak_gpu_memory_reserved": preflight_peak_reserved,
        "old_task_stopped": old_status["old_task_stopped"],
        "old_v1_auto_launch_remaining": old_status["old_v1_auto_launch_remaining"],
        "shared_gpu": True, "locked_final_accessed": False, "full_data_accessed": False,
        "sps_accessed": False, "codabench_accessed": False,
    }
    (args.out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True), encoding="utf-8")
    checkpoints = args.out_dir / "checkpoints"
    checkpoints.mkdir()
    shutil.copy2(args.checkpoint, checkpoints / "model_update_03000.pth")
    torch.load(checkpoints / "model_update_03000.pth", map_location="cpu", weights_only=False)
    if args.preflight_only:
        return

    curve: list[dict] = []
    aggregate: list[dict] = [{"arm": args.arm, "update": 3000, **replay["raw_errors"],
                              "inference_time": replay["mean_t_neural_s"]}]
    relative_targets = {update - START_UPDATE for update in args.milestones if update > START_UPDATE}
    _, train_loader = core.loader(train_paths, args, shuffle=True)
    iterator = iter(train_loader)
    started = time.monotonic()
    peak_allocated = peak_reserved = 0
    for relative in range(1, args.final_update - START_UPDATE + 1):
        optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        for micro_step in range(args.accumulation_steps):
            try:
                x, y, _, _ = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                x, y, _, _ = next(iterator)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            model.train()
            pred = _forward_zero_pressure(model, builder, x)
            loss, parts = _loss(pred, y, args.arm)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at update {START_UPDATE + relative}")
            (loss / args.accumulation_steps).backward()
            totals["total"] = totals.get("total", 0.0) + float(loss.detach().cpu()) / args.accumulation_steps
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) / args.accumulation_steps
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device.type == "cuda":
            peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated()))
            peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved()))
            if peak_reserved > memory_cap_bytes:
                raise RuntimeError(f"CUDA reserved memory exceeded {args.memory_cap_gib} GiB")
        curve.append({"update": START_UPDATE + relative, **totals})
        if relative in relative_targets:
            update = START_UPDATE + relative
            ev = evaluate(model, builder, dev_paths, args, device, args.out_dir / f"eval_{update:05d}",
                          args.arm, update, update == FINAL_UPDATE)
            aggregate.append({"arm": args.arm, "update": update, **ev["raw_errors"],
                              "inference_time": ev["mean_t_neural_s"]})
            _write_checkpoint(checkpoints / f"model_update_{update:05d}.pth", model, optimizer, update, preflight)
    write_rows(args.out_dir / "training_curve.csv", curve)
    write_rows(args.out_dir / "aggregate_metrics.csv", aggregate)
    runtime = {
        "arm": args.arm, "micro_batch": args.micro_batch, "accumulation_steps": args.accumulation_steps,
        "effective_batch": args.effective_batch, "memory_cap_gib": args.memory_cap_gib,
        "parameter_count": sum(p.numel() for p in model.parameters()), "added_parameter_count": 0,
        "training_wall_seconds": time.monotonic() - started, "peak_gpu_memory_allocated": peak_allocated,
        "peak_gpu_memory_reserved": peak_reserved, "inference_latency_final": aggregate[-1]["inference_time"],
        "gpu_model": torch.cuda.get_device_name(device), "shared_gpu": True,
    }
    (args.out_dir / "runtime.json").write_text(json.dumps(runtime, indent=2, sort_keys=True), encoding="utf-8")
    metadata = {
        "status": "REVIEW_REQUIRED", "arm": args.arm, "manifest_sha256": core.sha256(args.manifest),
        "scorer_sha256": core.sha256(args.kit_root / "scoring.py"),
        "initialization_checkpoint_sha256": core.sha256(args.checkpoint),
        "initialization_checkpoint_iteration": START_UPDATE, "seed": args.seed, "optimizer": "AdamW",
        "lr": args.lr, "batch": config_batch, "lambda_vort": FIXED_LAMBDA_VORT if args.arm == "V1" else 0.0,
        "feature_config": vars(feature_config), "train_trajectories": len(manifest["train"]),
        "dev_trajectories": len(json.loads(args.manifest.read_text())["dev"]), "milestones": list(args.milestones),
        "old_task_stopped": old_status["old_task_stopped"], "old_v1_auto_launch_remaining": old_status["old_v1_auto_launch_remaining"],
        "shared_gpu": True, "locked_final_accessed": False, "full_data_accessed": False,
        "sps_accessed": False, "codabench_accessed": False,
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("C0", "V1"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--old-task-status", type=Path, required=True)
    parser.add_argument("--final-update", type=int, default=FINAL_UPDATE)
    parser.add_argument("--milestones", type=int, nargs="+", default=list(MILESTONES))
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--effective-batch", type=int, default=8)
    parser.add_argument("--memory-cap-gib", type=float, default=12.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--lambda-vort", type=float, default=0.0)
    parser.add_argument("--start-tolerance", type=float, default=2e-5)
    parser.add_argument("--preflight-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
