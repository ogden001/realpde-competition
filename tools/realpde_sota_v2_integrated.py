#!/usr/bin/env python3
"""Frozen 50/16 SOTA-V2 integrated trial.

Recipe: Dense-All + P0-A + MF-CNO + N2 + fixed Vorticity.
Stage A: 0..30000 @ 1e-5. Stage B: 30001..35000 @ 3e-6 plus
one extra Rel-L2 term (0.027514). Only manifest train/dev are read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

import realpde_loss_official_v9 as core
from realpde_mf01 import MF01CNO
from realpde_p0_data import DenseAllWindowSampler, H5WindowDataset
from realpde_p0_features import P0FeatureBuilder, P0FeatureConfig
from realpde_structured_temporal_dynamics import diagnostics, vorticity

SEED = 20260901
N2 = {"mse": 1.0, "tke": 0.05, "rel": 0.027514, "mvpe": 0.009757}
LAMBDA_VORT = 15.5385751724
EXTRA_REL = 0.027514
STAGE_A_END, FINAL_UPDATE = 30_000, 35_000
STAGE_A_LR, STAGE_B_LR = 1e-5, 3e-6
MILESTONES = (7_500, 15_000, 20_000, 25_000, 30_000, 31_000, 32_500, 35_000)
MANIFEST_SHA = "42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347"
EXPECTED = {"train": 50, "dev": 16, "canonical_train": 2052, "dense_train": 40488, "dev_windows": 659, "features": 20}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def rows(path: Path, values: list[dict]) -> None:
    if not values:
        return
    fields = list(dict.fromkeys(k for row in values for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(values)


def stage_config(update: int) -> dict[str, float | str]:
    if not 1 <= update <= FINAL_UPDATE:
        raise ValueError(f"update must be in [1,{FINAL_UPDATE}]")
    return ({"stage": "A", "lr": STAGE_A_LR, "extra_rel": 0.0}
            if update <= STAGE_A_END else
            {"stage": "B", "lr": STAGE_B_LR, "extra_rel": EXTRA_REL})


def validate_protocol(args: argparse.Namespace) -> None:
    if args.seed != SEED or args.final_update != FINAL_UPDATE or tuple(args.milestones) != MILESTONES:
        raise ValueError("seed/final-update/milestones differ from frozen SOTA-V2 protocol")
    if args.micro_batch * args.accumulation_steps != 8:
        raise ValueError("effective batch must equal 8")
    if min(args.micro_batch, args.accumulation_steps, args.workers + 1) < 1:
        raise ValueError("invalid batch/worker configuration")


def split_paths(manifest_path: Path, split: str, data_root: Path | None) -> list[Path]:
    if split not in {"train", "dev"}:
        raise ValueError("only train/dev are permitted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = data_root or Path(manifest["real_root"])
    paths = [root / row["file"] for row in manifest[split]]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {split} files: {missing[:3]}")
    return paths


def sota_spacing(paths: list[Path]) -> tuple[float, float]:
    """Match the historical online-SOTA P0-A derivative spacing semantics."""
    ref = None
    for path in paths:
        with h5py.File(path, "r") as f:
            spacing = (float(f["x"][0, 2] - f["x"][0, 0]) / 2.0,
                       float(f["y"][2, 0] - f["y"][0, 0]) / 2.0)
        if ref is None:
            ref = spacing
        elif not np.allclose(ref, spacing, rtol=1e-6, atol=1e-9):
            raise ValueError(f"grid spacing mismatch: {spacing} vs {ref}")
    if ref is None or 0.0 in ref:
        raise ValueError("invalid P0-A grid spacing")
    return ref


def build_features(paths: list[Path], device: torch.device):
    dx, dy = sota_spacing(paths)
    config = P0FeatureConfig(include_p0_a=True, include_p0_b=False, dx=dx, dy=dy)
    builder = P0FeatureBuilder(config).to(device)
    if len(builder.feature_names) != EXPECTED["features"]:
        raise ValueError(f"P0-A feature count {len(builder.feature_names)} != {EXPECTED['features']}")
    return builder, config


def direct_state(payload: dict) -> dict[str, Tensor]:
    state = payload.get("model_state_dict", payload)
    return {name.removeprefix("cno."): value for name, value in state.items()}


def expand_lift_weight(source: Tensor, target: Tensor) -> Tensor:
    if source.shape[0] != target.shape[0] or source.shape[2:] != target.shape[2:]:
        raise ValueError("lift geometry mismatch")
    out = target.clone()
    if source.shape[1] == target.shape[1]:
        out.copy_(source)
    elif source.shape[1] == 3 and target.shape[1] > 3:
        out.zero_(); out[:, :3].copy_(source)
    else:
        raise ValueError(f"unsupported direct input width {source.shape[1]}")
    return out


def expand_project_parameter(source: Tensor, target: Tensor) -> Tensor:
    """Direct [u,v,p] -> MF [mean_u,mean_v,fluct_u,fluct_v,p]."""
    if source.shape[0] != 3 or target.shape[0] != 5 or source.shape[1:] != target.shape[1:]:
        raise ValueError("project geometry mismatch")
    out = target.clone()
    out[:2].copy_(source[:2]); out[2:4].copy_(source[:2]); out[4].copy_(source[2])
    return out


def init_mf_from_direct(model: MF01CNO, payload: dict, in_channels: int) -> None:
    source, target = direct_state(payload), model.cno.state_dict()
    lift = "lift.inter_CNOBlock.convolution.weight"
    project = {"project.convolution.weight", "project.convolution.bias"}
    for name in target:
        if name == lift or name in project:
            continue
        if name not in source or source[name].shape != target[name].shape:
            raise ValueError(f"Direct->MF mismatch at {name}")
        target[name] = source[name]
    if target[lift].shape[1] != in_channels:
        raise ValueError("target input width != P0-A feature width")
    target[lift] = expand_lift_weight(source[lift], target[lift])
    for name in project:
        target[name] = expand_project_parameter(source[name], target[name])
    model.cno.load_state_dict(target, strict=True)


def build_direct(kit_root: Path, builder: P0FeatureBuilder, payload: dict, device: torch.device):
    sys.path.insert(0, str(kit_root))
    from rpde_baselines.model.cno import CNO3d
    model = CNO3d(in_dim=len(builder.feature_names), out_dim=3, out_dim_mult=1, in_size=64, N_layers=3).to(device)
    source, target = direct_state(payload), model.state_dict()
    lift = "lift.inter_CNOBlock.convolution.weight"
    for name in target:
        if name == lift:
            target[name] = expand_lift_weight(source[name], target[name])
        elif name in source and source[name].shape == target[name].shape:
            target[name] = source[name]
        else:
            raise ValueError(f"direct parity mismatch at {name}")
    model.load_state_dict(target, strict=True)
    return model


def forward_mf(model: MF01CNO, builder: P0FeatureBuilder, x: Tensor) -> Tensor:
    pred = model(builder(x)).clone(); pred[..., 2] = 0.0
    return pred


def forward_direct(model, builder: P0FeatureBuilder, x: Tensor) -> Tensor:
    z = builder(x).permute(0, 4, 1, 2, 3)
    pred = model(z).permute(0, 2, 3, 4, 1).clone(); pred[..., 2] = 0.0
    return pred


def integrated_loss(pred: Tensor, target: Tensor, update: int):
    parts = core.loss_parts(pred, target)
    parts["vorticity"] = (vorticity(pred, dx=1.0, dy=1.0) - vorticity(target, dx=1.0, dy=1.0)).square().mean()
    cfg = stage_config(update)
    total = sum(N2[k] * parts[k] for k in N2) + LAMBDA_VORT * parts["vorticity"]
    total = total + float(cfg["extra_rel"]) * parts["rel"]
    return total, parts


def horizon_error_summary(pred: np.ndarray, target: np.ndarray) -> dict:
    if pred.shape != target.shape or pred.ndim != 5 or pred.shape[1] != 20:
        raise ValueError("expected matched [N,20,H,W,C]")
    sq = np.square(pred[..., :2] - target[..., :2]).sum(axis=(0, 2, 3, 4), dtype=np.float64)
    total = float(sq.sum())
    def group(a, b):
        value = float(sq[a:b].sum()); return {"squared_error": value, "fraction": value / max(total, 1e-30)}
    return {"total_squared_error": total, "per_horizon_squared_error": [float(x) for x in sq],
            "h1_h15": group(0, 15), "h16_h18": group(15, 18), "h19": group(18, 19),
            "h20": group(19, 20), "h19_h20": group(18, 20)}


def dense_loader(paths, args):
    preload_to_ram = bool(getattr(args, "preload_to_ram", False))
    prefetch_factor = int(getattr(args, "prefetch_factor", 4))
    ds = H5WindowDataset(
        paths,
        in_steps=20,
        out_steps=20,
        stride=20,
        sub_sample=2,
        include_pressure=False,
        window_mode="dense_all",
        preload_to_ram=preload_to_ram,
    )
    sampler = DenseAllWindowSampler(ds, seed=args.seed)
    loader = DataLoader(
        ds,
        batch_size=args.micro_batch,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=bool(args.workers > 0),
        prefetch_factor=prefetch_factor if args.workers > 0 else None,
        drop_last=True,
    )
    return ds, sampler, loader


def dev_loader(paths, args):
    ds = H5WindowDataset(paths, in_steps=20, out_steps=20, stride=20, sub_sample=2,
                         include_pressure=False, window_mode="fixed")
    return ds, DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers,
                          pin_memory=True, persistent_workers=False)


@torch.no_grad()
def evaluate(model, builder, dev_paths, args, device, out: Path, update: int):
    out.mkdir(parents=True, exist_ok=True); ds, loader = dev_loader(dev_paths, args)
    model.eval(); preds, targets, elapsed = [], [], 0.0
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter(); p = forward_mf(model, builder, x)
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed += time.perf_counter() - t0; preds.append(p.cpu().numpy()); targets.append(y.numpy())
    pred, target = np.concatenate(preds).astype(np.float32), np.concatenate(targets).astype(np.float32)
    if not np.isfinite(pred).all() or float(np.abs(pred[..., 2]).max()) != 0.0:
        raise FloatingPointError("invalid dev prediction")
    scored = core.score_bundle(args.kit_root, pred, target, elapsed / len(ds), out)
    trajectory, anatomy = core.trajectory_rows(ds, pred, target, args.kit_root)
    rows(out / "trajectory_metrics.csv", trajectory)
    result = scored | {"update": update, "windows": len(ds), "trajectories": len(trajectory), "trajectory_anatomy": anatomy}
    dump(out / "scores.json", result)
    if update in {STAGE_A_END, FINAL_UPDATE}:
        diagnostics(ds, pred, target, out, "SOTA-V2", update, args.kit_root)
        np.savez_compressed(out / "predictions.npz", prediction=pred[..., :2], target=target[..., :2],
                            trajectory=np.asarray([r.path.name for r in ds.refs]), window_start=np.asarray([r.start for r in ds.refs]))
        dump(out / "horizon_error_summary.json", horizon_error_summary(pred, target))
    return result


def save_checkpoint(path, model, optimizer, update, config, metadata):
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "iteration": update, "feature_set": "P0-A", "feature_config": vars(config),
                "loss_weights": N2, "lambda_vort": LAMBDA_VORT, "extra_rel_stage_b": EXTRA_REL,
                "metadata": metadata}, path)


def run(args):
    validate_protocol(args)
    if args.out_dir.exists() and any(args.out_dir.iterdir()): raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if sha256(args.manifest) != MANIFEST_SHA: raise ValueError("frozen manifest SHA mismatch")
    if not (args.kit_root / "scoring.py").is_file(): raise FileNotFoundError(args.kit_root / "scoring.py")
    core.set_seed(args.seed)
    train, dev = split_paths(args.manifest, "train", args.data_root), split_paths(args.manifest, "dev", args.data_root)
    if (len(train), len(dev)) != (EXPECTED["train"], EXPECTED["dev"]): raise ValueError("split is not frozen 50/16")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda": raise RuntimeError("CUDA required")
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)

    builder, config = build_features(train, device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = MF01CNO(args.kit_root, len(builder.feature_names), device); init_mf_from_direct(model, payload, len(builder.feature_names))
    optimizer = torch.optim.AdamW(model.parameters(), lr=STAGE_A_LR)
    train_ds, sampler, loader = dense_loader(train, args)
    canonical = H5WindowDataset(train, window_mode="fixed"); dev_ds = H5WindowDataset(dev, window_mode="fixed")
    if (len(canonical), len(train_ds), len(dev_ds)) != (EXPECTED["canonical_train"], EXPECTED["dense_train"], EXPECTED["dev_windows"]):
        raise ValueError(f"window audit mismatch: {len(canonical)}/{len(train_ds)}/{len(dev_ds)}")
    metadata = {"recipe": "Dense-All+P0-A+MF-CNO+N2+Vorticity+StageB-extra-Rel", "seed": args.seed,
                "manifest_sha256": MANIFEST_SHA, "checkpoint_sha256": sha256(args.checkpoint),
                "scorer_sha256": sha256(args.kit_root / "scoring.py"), "feature_config": vars(config),
                "p0a_spacing_mode": "historical_online_sota", "n2": N2, "lambda_vort": LAMBDA_VORT,
                "stage_a": {"end": STAGE_A_END, "lr": STAGE_A_LR},
                "stage_b": {"end": FINAL_UPDATE, "lr": STAGE_B_LR, "extra_rel": EXTRA_REL},
                "micro_batch": args.micro_batch, "accumulation_steps": args.accumulation_steps,
                "effective_batch": 8, "canonical_train_windows": len(canonical), "dense_train_windows": len(train_ds),
                "dense_expansion": len(train_ds)/len(canonical), "dev_windows": len(dev_ds), "milestones": list(args.milestones),
                "locked_final_accessed": False, "full_data_accessed": False, "sps_accessed": False, "codabench_accessed": False}
    dump(args.out_dir / "run_metadata.json", metadata)

    # Preflight: exact UV parity after Direct->P0-A/MF surgery, finite loss/gradient.
    probe_x, probe_y, _, _ = next(iter(dev_loader(dev, argparse.Namespace(eval_batch_size=2, workers=0))[1]))
    probe_x, probe_y = probe_x.to(device), probe_y.to(device)
    mf_pred = forward_mf(model, builder, probe_x)
    direct = build_direct(args.kit_root, builder, payload, device); direct_pred = forward_direct(direct, builder, probe_x)
    parity = float((mf_pred[..., :2] - direct_pred[..., :2]).abs().max())
    if parity > args.parity_tolerance: raise RuntimeError(f"Direct->MF parity failed: {parity}")
    loss, parts = integrated_loss(mf_pred, probe_y, 1); optimizer.zero_grad(set_to_none=True); loss.backward()
    grad = max(float(p.grad.abs().max()) for p in model.parameters() if p.grad is not None); optimizer.zero_grad(set_to_none=True)
    preflight = {"passed": bool(torch.isfinite(mf_pred).all() and torch.isfinite(loss) and grad > 0),
                 "direct_to_mf_uv_max_abs_diff": parity, "pressure_max_abs": float(mf_pred[...,2].abs().max()),
                 "loss": float(loss.detach()), "loss_parts": {k: float(v.detach()) for k,v in parts.items()},
                 "gradient_max_abs": grad, "dense_train_windows": len(train_ds), "feature_count": len(builder.feature_names)}
    dump(args.out_dir / "preflight.json", preflight)
    if not preflight["passed"]: raise RuntimeError("preflight failed")
    if args.preflight_only: return
    del direct, direct_pred

    checkpoints = args.out_dir / "checkpoints"; checkpoints.mkdir()
    history, iterator, epoch = [], iter(loader), 0
    started = time.monotonic(); current_lr = STAGE_A_LR
    for update in range(1, args.final_update + 1):
        cfg = stage_config(update)
        if float(cfg["lr"]) != current_lr:
            current_lr = float(cfg["lr"])
            for group in optimizer.param_groups: group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True); total_parts = {k: 0.0 for k in (*N2, "vorticity")}
        for _ in range(args.accumulation_steps):
            try: x, y, _, _ = next(iterator)
            except StopIteration:
                epoch += 1; sampler.set_epoch(epoch); iterator = iter(loader); x, y, _, _ = next(iterator)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            model.train(); pred = forward_mf(model, builder, x); loss, parts = integrated_loss(pred, y, update)
            if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite loss @ {update}")
            (loss / args.accumulation_steps).backward()
            for k in total_parts: total_parts[k] += float(parts[k].detach()) / args.accumulation_steps
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if update % args.log_every == 0:
            print(json.dumps({"update": update, "stage": cfg["stage"], "lr": current_lr, **total_parts}), flush=True)
        if update in args.milestones:
            ev = evaluate(model, builder, dev, args, device, args.out_dir / f"eval_{update:05d}", update)
            row = {"update": update, "stage": cfg["stage"], "lr": current_lr, **ev["raw_errors"],
                   "mean_t_neural_s": ev["mean_t_neural_s"], "elapsed_seconds": time.monotonic()-started}
            history.append(row); rows(args.out_dir / "aggregate_metrics.csv", history)
            save_checkpoint(checkpoints / f"model_update_{update:05d}.pth", model, optimizer, update, config, metadata)
            save_checkpoint(checkpoints / "model_latest.pth", model, optimizer, update, config, metadata)
            print(json.dumps(row), flush=True)
    dump(args.out_dir / "runtime.json", {"training_wall_seconds": time.monotonic()-started,
         "peak_gpu_memory_allocated": int(torch.cuda.max_memory_allocated()) if device.type=="cuda" else 0,
         "peak_gpu_memory_reserved": int(torch.cuda.max_memory_reserved()) if device.type=="cuda" else 0,
         "dense_epochs_completed": epoch + 1})
    dump(args.out_dir / "status.json", {"state": "DONE", "update": FINAL_UPDATE})


def main():
    p = argparse.ArgumentParser(); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--data-root", type=Path)
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--kit-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True); p.add_argument("--final-update", type=int, default=FINAL_UPDATE)
    p.add_argument("--milestones", type=int, nargs="+", default=list(MILESTONES)); p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--accumulation-steps", type=int, default=1); p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2); p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--log-every", type=int, default=100); p.add_argument("--parity-tolerance", type=float, default=1e-6)
    p.add_argument("--require-cuda", action="store_true"); p.add_argument("--preflight-only", action="store_true"); run(p.parse_args())


if __name__ == "__main__": main()
