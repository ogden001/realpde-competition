#!/usr/bin/env python3
"""Frozen Direct@1500 structured-temporal-dynamics screening runner.

Each invocation runs exactly one pre-registered arm.  It deliberately accepts
all external locations on the CLI and never reads the locked-final split.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

import realpde_loss_official_v9 as core
import realpde_mf01 as direct

N2 = direct.N2_WEIGHTS


def validate_batch_configuration(*, micro_batch_size: int, accumulation_steps: int,
                                 effective_batch_size: int, max_gpu_memory_gib: float) -> dict:
    if micro_batch_size < 1 or accumulation_steps < 1:
        raise ValueError("micro batch and accumulation steps must be positive")
    if micro_batch_size * accumulation_steps != effective_batch_size:
        raise ValueError("micro_batch_size * accumulation_steps must equal effective_batch_size")
    if not 0.0 < max_gpu_memory_gib <= 24.0:
        raise ValueError("max GPU memory must be in (0, 24] GiB")
    return {"micro_batch_size": micro_batch_size, "accumulation_steps": accumulation_steps,
            "effective_batch_size": effective_batch_size, "max_gpu_memory_gib": max_gpu_memory_gib}


def configure_cuda_memory_cap(device: torch.device, max_gpu_memory_gib: float) -> int:
    """Cap this process's CUDA allocator; return its byte budget."""
    if device.type != "cuda": return 0
    total = torch.cuda.get_device_properties(device).total_memory
    cap = int(max_gpu_memory_gib * 1024 ** 3)
    if cap > total: raise ValueError("requested CUDA cap exceeds device memory")
    device_index = torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(cap / total, device_index)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device_index)
    return cap


def future_deltas(past: Tensor, future: Tensor) -> Tensor:
    """Velocity differences with Future[0] anchored at Past[-1]."""
    uv = future[..., :2]
    return torch.cat((uv[:, :1] - past[:, -1:, ..., :2], uv[:, 1:] - uv[:, :-1]), dim=1)


def vorticity(field: Tensor, *, dx: float, dy: float) -> Tensor:
    """Interior dv/dx - du/dy, with ordinary centered finite differences."""
    uv = field[..., :2]
    dv_dx = (uv[:, :, 1:-1, 2:, 1] - uv[:, :, 1:-1, :-2, 1]) / (2.0 * dx)
    du_dy = (uv[:, :, 2:, 1:-1, 0] - uv[:, :, :-2, 1:-1, 0]) / (2.0 * dy)
    return dv_dx - du_dy


class TemporalMixer(nn.Module):
    """Residual Future20-only mixer: Conv3d(2,16,(3,1,1)) -> GELU -> Conv3d."""

    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Conv3d(2, 16, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.activation = nn.GELU()
        self.output = nn.Conv3d(16, 2, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, uv: Tensor) -> Tensor:
        return self.output(self.activation(self.input(uv.permute(0, 4, 1, 2, 3)))).permute(0, 2, 3, 4, 1)


def apply_temporal_mixer(prediction: Tensor, mixer: TemporalMixer) -> Tensor:
    result = prediction.clone()
    result[..., :2] = prediction[..., :2] + mixer(prediction[..., :2])
    return result


def forward_direct_zero_pressure(model, builder, x: Tensor) -> Tensor:
    """Canonical Direct prediction with the unobserved pressure fixed to zero."""
    result = direct.forward(model, builder, x)
    result[..., 2] = 0.0
    return result


def loss_with_arm(pred: Tensor, target: Tensor, past: Tensor, arm: str, weight: float) -> tuple[Tensor, dict[str, Tensor]]:
    parts = core.loss_parts(pred, target)
    total = sum(N2[name] * parts[name] for name in N2)
    if arm == "T1":
        parts["delta"] = (future_deltas(past, pred) - future_deltas(past, target)).square().mean()
        total = total + weight * parts["delta"]
    elif arm == "T2":
        parts["vorticity"] = (vorticity(pred, dx=1.0, dy=1.0) - vorticity(target, dx=1.0, dy=1.0)).square().mean()
        total = total + weight * parts["vorticity"]
    return total, parts


def calibration_weight(pred: Tensor, target: Tensor, past: Tensor, arm: str) -> float:
    base = sum(N2[name] * value for name, value in core.loss_parts(pred, target).items() if name in N2)
    if arm == "T1": raw = (future_deltas(past, pred) - future_deltas(past, target)).square().mean()
    elif arm == "T2": raw = (vorticity(pred, dx=1.0, dy=1.0) - vorticity(target, dx=1.0, dy=1.0)).square().mean()
    else: return 0.0
    return float((0.10 * base / raw.clamp_min(1e-12)).detach().cpu())


def metric_triplet(pred: np.ndarray, target: np.ndarray, scoring) -> dict[str, float]:
    channels = scoring.measured_channels(target)
    return {"rel_l2": float(np.mean(scoring.rel_l2_per_sample(pred, target, channels))),
            "tke": float(np.mean(scoring.tke_rel_l2_per_sample(pred, target, channels))),
            "mvpe": float(scoring.mvpe_rel_l2(pred, target))}


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows: return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def diagnostics(ds, prediction: np.ndarray, target: np.ndarray, out: Path, arm: str, update: int, kit_root: Path) -> None:
    import sys
    sys.path.insert(0, str(kit_root)); import scoring
    window_rows, horizon_rows, th_rows, temporal_rows = [], [], [], []
    groups: dict[str, list[int]] = defaultdict(list)
    for index, ref in enumerate(ds.refs):
        groups[ref.path.name].append(index)
        m = metric_triplet(prediction[index:index + 1], target[index:index + 1], scoring)
        window_rows.append({"arm": arm, "update": update, "trajectory": ref.path.name, "window_start": ref.start, **m})
    for h in range(prediction.shape[1]):
        m = metric_triplet(prediction[:, h:h + 1], target[:, h:h + 1], scoring)
        horizon_rows.append({"arm": arm, "update": update, "horizon": h + 1, "metric_label": "diagnostic horizon raw metric", **m})
        prev_p = prediction[:, h - 1] if h else np.asarray([ds[i][0][-1].numpy() for i in range(len(ds))])
        prev_y = target[:, h - 1] if h else np.asarray([ds[i][0][-1].numpy() for i in range(len(ds))])
        temporal_rows.append({"arm": arm, "update": update, "horizon": h + 1,
                              "increment_uv_mse": float(np.mean((prediction[:, h, ..., :2] - prev_p[..., :2] - (target[:, h, ..., :2] - prev_y[..., :2])) ** 2)),
                              "prediction_change_l2": float(np.mean(np.linalg.norm((prediction[:, h, ..., :2] - prev_p[..., :2]).reshape(len(ds), -1), axis=1))),
                              "target_change_l2": float(np.mean(np.linalg.norm((target[:, h, ..., :2] - prev_y[..., :2]).reshape(len(ds), -1), axis=1))),
                              "error_l2": float(np.mean(np.linalg.norm((prediction[:, h, ..., :2] - target[:, h, ..., :2]).reshape(len(ds), -1), axis=1)))})
        for name, indices in groups.items():
            m = metric_triplet(prediction[indices, h:h + 1], target[indices, h:h + 1], scoring)
            th_rows.append({"arm": arm, "update": update, "trajectory": name, "horizon": h + 1, **m})
    trajectory_rows = []
    for name, indices in sorted(groups.items()):
        m = metric_triplet(np.concatenate([prediction[i] for i in indices])[None], np.concatenate([target[i] for i in indices])[None], scoring)
        trajectory_rows.append({"arm": arm, "update": update, "trajectory": name, **m})
    write_rows(out / "window_metrics.csv", window_rows); write_rows(out / "horizon_metrics.csv", horizon_rows)
    write_rows(out / "trajectory_metrics.csv", trajectory_rows); write_rows(out / "trajectory_horizon_metrics.csv", th_rows)
    write_rows(out / "temporal_metrics.csv", temporal_rows)


@torch.no_grad()
def evaluate(model, builder, mixer, paths, args, device, out: Path, arm: str, update: int, save_predictions: bool) -> dict:
    out.mkdir(parents=True, exist_ok=True); ds, loader = core.loader(paths, args, shuffle=False)
    pred_all, target_all, elapsed = [], [], 0.0; model.eval()
    if mixer is not None: mixer.eval()
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda": torch.cuda.synchronize()
        started = time.perf_counter(); pred = forward_direct_zero_pressure(model, builder, x)
        if mixer is not None: pred = apply_temporal_mixer(pred, mixer)
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed += time.perf_counter() - started; pred_all.append(pred.cpu().numpy().astype(np.float32)); target_all.append(y.numpy().astype(np.float32))
    pred, target = np.concatenate(pred_all), np.concatenate(target_all)
    if not np.isfinite(pred).all() or not np.isfinite(target).all() or np.abs(pred[..., 2]).max() != 0: raise FloatingPointError("nonfinite prediction or nonzero pressure")
    scored = core.score_bundle(args.kit_root, pred, target, elapsed / len(ds), out)
    diagnostics(ds, pred, target, out, arm, update, args.kit_root)
    if save_predictions: np.savez_compressed(out / "predictions.npz", prediction=pred[..., :2], target=target[..., :2], trajectory=np.asarray([r.path.name for r in ds.refs]), window_start=np.asarray([r.start for r in ds.refs]))
    return scored | {"windows": len(ds)}


def build(args, train_paths, device):
    builder, config = direct.build_features(train_paths, device); payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = direct.cno_direct(args.kit_root, len(builder.feature_names), device); state = payload.get("model_state_dict", payload)
    key = "lift.inter_CNOBlock.convolution.weight"
    if key in state and state[key].shape[1] == len(builder.feature_names): model.load_state_dict(state, strict=True)
    else: direct.adapt_input_weight(model, payload, len(builder.feature_names))
    return model, builder, config, payload


def run(args) -> None:
    if args.out_dir.exists() and any(args.out_dir.iterdir()): raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True); core.set_seed(args.seed)
    batch_config = validate_batch_configuration(micro_batch_size=args.micro_batch_size,
                                                accumulation_steps=args.accumulation_steps,
                                                effective_batch_size=args.batch_size,
                                                max_gpu_memory_gib=args.max_gpu_memory_gib)
    args.batch_size = args.micro_batch_size
    _, train_paths = core.read_manifest(args.manifest, "train"); _, dev_paths = core.read_manifest(args.manifest, "dev")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    memory_cap_bytes = configure_cuda_memory_cap(device, args.max_gpu_memory_gib)
    model, builder, config, payload = build(args, train_paths, device)
    mixer = TemporalMixer().to(device) if args.arm == "T3" else None
    ds, loader = core.loader(train_paths, args, shuffle=True); x, y, _, _ = next(iter(loader)); x, y = x.to(device), y.to(device)
    initial = forward_direct_zero_pressure(model, builder, x); pred = apply_temporal_mixer(initial, mixer) if mixer else initial
    if not torch.isfinite(pred).all() or float(pred[..., 2].abs().max()) != 0: raise FloatingPointError("initial output invalid")
    parity = float((pred - initial).abs().max()); weight = calibration_weight(pred, y, x, args.arm)
    params = list(model.parameters()) + ([] if mixer is None else list(mixer.parameters())); optimizer = torch.optim.AdamW(params, lr=args.lr)
    if "optimizer_state_dict" in payload and mixer is None: optimizer.load_state_dict(payload["optimizer_state_dict"])
    if mixer is not None and not any(id(p) in {id(q) for g in optimizer.param_groups for q in g["params"]} for p in mixer.parameters()): raise RuntimeError("mixer missing from optimizer")
    preflight_gradient = None
    if mixer is not None:
        probe_loss, _ = loss_with_arm(pred, y, x, args.arm, weight)
        probe_loss.backward()
        preflight_gradient = float(mixer.output.weight.grad.abs().max().detach().cpu()) if mixer.output.weight.grad is not None else 0.0
        optimizer.zero_grad(set_to_none=True)
        if preflight_gradient <= 0.0: raise RuntimeError("T3 mixer output has zero preflight gradient")
    preflight = {"passed": True, "arm": args.arm, "shape": list(pred.shape), "initial_max_abs_diff_from_direct": parity, "pressure_max_abs": float(pred[..., 2].abs().max()), "calibration_weight": weight, "t3_output_gradient_max_abs": preflight_gradient}
    (args.out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    if args.preflight_only:
        if args.preflight_evaluate:
            evaluation = evaluate(model, builder, mixer, dev_paths, args, device, args.out_dir / "eval_01500", args.arm, 1500, False)
            preflight["canonical_dev_raw_errors"] = evaluation["raw_errors"]
            (args.out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
        return
    baseline = evaluate(model, builder, mixer, dev_paths, args, device, args.out_dir / "eval_01500", args.arm, 1500, False)
    history = [{"update": 1500, **baseline["raw_errors"]}]; curve = []
    iterator = iter(loader); started = time.monotonic(); peak_allocated = peak_reserved = 0
    for relative in range(1, args.updates + 1):
        optimizer.zero_grad(set_to_none=True); totals: dict[str, float] = defaultdict(float)
        for micro_step in range(args.accumulation_steps):
            try: x, y, _, _ = next(iterator)
            except StopIteration: iterator = iter(loader); x, y, _, _ = next(iterator)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True); model.train()
            if mixer is not None: mixer.train()
            pred = forward_direct_zero_pressure(model, builder, x); pred = apply_temporal_mixer(pred, mixer) if mixer else pred
            loss, parts = loss_with_arm(pred, y, x, args.arm, weight)
            if not torch.isfinite(loss): raise FloatingPointError("nonfinite loss")
            (loss / args.accumulation_steps).backward()
            totals["total"] += float(loss.detach().cpu()) / args.accumulation_steps
            for name, value in parts.items(): totals[name] += float(value.detach().cpu()) / args.accumulation_steps
        if mixer is not None and relative == 1 and (mixer.output.weight.grad is None or torch.count_nonzero(mixer.output.weight.grad) == 0): raise RuntimeError("mixer output has zero gradient")
        torch.nn.utils.clip_grad_norm_(params, 1.0); optimizer.step()
        if device.type == "cuda":
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device)); peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
            if peak_reserved > memory_cap_bytes: raise RuntimeError("CUDA allocator exceeded configured memory cap")
        curve.append({"update": 1500 + relative, **totals})
        if relative in args.milestones:
            absolute = 1500 + relative; ev = evaluate(model, builder, mixer, dev_paths, args, device, args.out_dir / f"eval_{absolute:05d}", args.arm, absolute, absolute == 3000)
            history.append({"update": absolute, **ev["raw_errors"], "inference_time": ev["mean_t_neural_s"]})
            torch.save({"model_state_dict": model.state_dict(), "mixer_state_dict": mixer.state_dict() if mixer else None, "optimizer_state_dict": optimizer.state_dict(), "iteration": absolute}, args.out_dir / f"model_update_{absolute:05d}.pth")
    write_rows(args.out_dir / "training_curve.csv", curve); write_rows(args.out_dir / "aggregate_metrics.csv", [{"arm": args.arm, **r} for r in history])
    runtime = {"parameter_count": sum(p.numel() for p in model.parameters()), "added_parameter_count": 0 if mixer is None else sum(p.numel() for p in mixer.parameters()), "peak_gpu_memory_allocated": peak_allocated, "peak_gpu_memory_reserved": peak_reserved, "gpu_memory_cap_bytes": memory_cap_bytes, "training_wall_seconds": time.monotonic() - started, "inference_latency": history[-1].get("inference_time")}
    (args.out_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    meta = {"status": "REVIEW_REQUIRED", "arm": args.arm, "manifest_sha256": core.sha256(args.manifest), "scorer_sha256": core.sha256(args.kit_root / "scoring.py"), "initialization_checkpoint_sha256": core.sha256(args.checkpoint), "seed": args.seed, "optimizer": "AdamW", "lr": args.lr, "batch": batch_config, "lambda_delta": weight if args.arm == "T1" else None, "lambda_vort": weight if args.arm == "T2" else None, "feature_config": vars(config), "locked_final_accessed": False, "codabench_accessed": False}
    (args.out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--arm", choices=("C0", "T1", "T2", "T3"), required=True); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--kit-root", type=Path, required=True); p.add_argument("--out-dir", type=Path, required=True); p.add_argument("--updates", type=int, default=1500); p.add_argument("--milestones", type=int, nargs="+", default=[500, 1000, 1500]); p.add_argument("--batch-size", type=int, default=8); p.add_argument("--micro-batch-size", type=int, default=2); p.add_argument("--accumulation-steps", type=int, default=4); p.add_argument("--max-gpu-memory-gib", type=float, default=12.0); p.add_argument("--workers", type=int, default=2); p.add_argument("--max-windows", type=int); p.add_argument("--lr", type=float, default=1e-5); p.add_argument("--seed", type=int, default=20260901); p.add_argument("--preflight-only", action="store_true"); p.add_argument("--preflight-evaluate", action="store_true")
    run(p.parse_args())


if __name__ == "__main__": main()
