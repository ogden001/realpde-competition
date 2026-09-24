#!/usr/bin/env python3
"""Clean residual-aware SPS validation for REALPDE Track 1.

This reproduces the successful colleague-80 uncertainty semantics on the clean
baseline: the uncertainty head observes Past20 + the *pre-residual* base CNO
forecast, while its target is the error of the *post-residual* final forecast.
The point predictor is frozen.  Seen-Dev12 selects the head/calibration; AoA10
holdout is evaluated once with the frozen Seen-Dev calibration and is never
used for selection.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from colleague_80pt.realpde_h5_feature_adapter_train import H5WindowDataset, paths_from_split_manifest
from colleague_80pt.residual_multi import load_full_residual_model
from colleague_80pt.train_head_fast import Head3D, HeadConfig, initialize_coupled
from colleague_80pt import realpde_sps_scoring as SPS

SEED = 41
UPDATES = 5000
EVAL_EVERY = 500
BATCH = 16
LR = 1e-3
WEIGHT_DECAY = 1e-5
WARMUP = 200
EPS = 1e-6
FLOORS = (0.0, 0.0025, 0.005)
MULTS = (0.5, 0.75, 1.0, 1.25, 1.5)
RELS = (0.0, 0.0025, 0.005, 0.0075)
STATIC_ABS = (0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.04)
STATIC_REL = (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.08, 0.1)


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def dataset(paths: list[Path], stride: int) -> H5WindowDataset:
    return H5WindowDataset(paths, in_steps=20, out_steps=20, stride=stride,
                           sub_sample=2, include_pressure=False, window_mode="fixed")


@torch.no_grad()
def collect(model, head, paths: list[Path], device: torch.device, workers: int):
    ds = dataset(paths, 20)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")
    preds, bases, sigmas, targets = [], [], [], []
    model.eval(); head.eval()
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
    return ds, np.concatenate(preds), np.concatenate(bases), np.concatenate(sigmas), np.concatenate(targets)


def adaptive_score(pred, target, sigma, floor: float, mu: float, mv: float, rel: float) -> dict[str, float]:
    half = np.zeros_like(pred, dtype=np.float32)
    half[..., 0] = floor + mu * sigma[..., 0] + rel * np.abs(pred[..., 0])
    half[..., 1] = floor + mv * sigma[..., 1] + rel * np.abs(pred[..., 1])
    c = SPS.measured_channels(target)
    raw, cov = SPS.aggregate_sps(pred, target, c, pred - half, pred + half)
    return {"sps": 100.0 * raw, "coverage": cov, "mean_width_uv": float(np.mean(2.0 * half[..., :2]))}


def calibrate_adaptive(pred, target, sigma) -> tuple[list[dict[str, float]], dict[str, float]]:
    rows = []
    for floor in FLOORS:
        for mu in MULTS:
            for mv in MULTS:
                for rel in RELS:
                    rows.append({"floor": floor, "mult_u": mu, "mult_v": mv, "rel": rel,
                                 **adaptive_score(pred, target, sigma, floor, mu, mv, rel)})
    rows.sort(key=lambda r: -r["sps"])
    return rows, rows[0]


def static_score(pred, target, abs_w: float, rel_w: float) -> dict[str, float]:
    half = (abs_w + rel_w * np.abs(pred)).astype(np.float32)
    half[..., 2] = 0.0
    c = SPS.measured_channels(target)
    raw, cov = SPS.aggregate_sps(pred, target, c, pred - half, pred + half)
    return {"sps": 100.0 * raw, "coverage": cov, "mean_width_uv": float(np.mean(2.0 * half[..., :2]))}


def calibrate_static(pred, target) -> tuple[list[dict[str, float]], dict[str, float]]:
    rows = []
    for a in STATIC_ABS:
        for r in STATIC_REL:
            rows.append({"abs": a, "rel": r, **static_score(pred, target, a, r)})
    rows.sort(key=lambda row: -row["sps"])
    return rows, rows[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--real-root", type=Path, required=True)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--holdout-manifest", type=Path, required=True)
    p.add_argument("--residual-checkpoint", type=Path, required=True)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--require-cuda", action="store_true")
    args = p.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    train_paths, dev_paths = paths_from_split_manifest(args.real_root, args.train_manifest)
    _, holdout_paths = paths_from_split_manifest(args.real_root, args.holdout_manifest)
    if (len(train_paths), len(dev_paths), len(holdout_paths)) != (51, 12, 18):
        raise ValueError("expected clean 51/12/18 split")
    if set(p.name for p in holdout_paths) & (set(p.name for p in train_paths) | set(p.name for p in dev_paths)):
        raise ValueError("AoA10 holdout overlaps train/dev")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")
    torch.manual_seed(SEED); np.random.seed(SEED)
    model, _ = load_full_residual_model(args.residual_checkpoint, args.model_root, device)
    model.eval()

    cfg = HeadConfig(hidden=64, blocks=2, dropout=0.0, include_pressure=True,
                     history_context=False, sigma0=0.02, min_sigma=1e-4,
                     max_sigma=1.0, include_delta=False)
    head = Head3D(cfg).to(device)
    initialize_coupled(head, seed=SEED)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARMUP) * (0.5 * (1 + np.cos(np.pi * min(1.0, s / UPDATES))))
    )

    tr_ds = dataset(train_paths, 5)
    gen = torch.Generator(); gen.manual_seed(SEED)
    loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, generator=gen,
                        drop_last=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    iterator = iter(loader)

    _, pred_before, _, _, target_before = collect(model, head, dev_paths, device, args.workers)
    static_grid, static_best = calibrate_static(pred_before, target_before)
    dump(args.out_dir / "static_calibration_grid.json", static_grid)

    evals = []
    losses = []
    best_state = None
    best = None
    started = time.monotonic()
    for step in range(1, UPDATES + 1):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader); x, y = next(iterator)
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with torch.no_grad():
            base = model.base_predict(x)
            delta = model.predict_delta(x, base)
            final = model.combine(base, delta, 1.0)
        log_std = head(x, base)
        err = (final[..., :2] - y[..., :2]).abs()
        mask = (y[..., :2] != 0.0).to(log_std.dtype)
        loss = ((log_std - torch.log(err + EPS)).abs() * mask).sum() / mask.sum().clamp_min(1.0)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite logmae @ {step}")
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sched.step()
        if step == 1 or step % 100 == 0:
            losses.append({"step": step, "loss": float(loss.detach().cpu()), "lr": float(opt.param_groups[0]["lr"])})
        if step % EVAL_EVERY == 0:
            _, pred_dev, _, sigma_dev, target_dev = collect(model, head, dev_paths, device, args.workers)
            grid, chosen = calibrate_adaptive(pred_dev, target_dev, sigma_dev)
            record = {"step": step, "best": chosen}
            evals.append(record)
            dump(args.out_dir / f"calibration_grid_{step:05d}.json", grid)
            if best is None or chosen["sps"] > best["best"]["sps"]:
                best = record
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    if best_state is None or best is None:
        raise RuntimeError("no SPS checkpoint selected")
    head.load_state_dict(best_state, strict=True); head.eval()
    _, pred_after, _, sigma_after, target_after = collect(model, head, dev_paths, device, args.workers)
    parity = float(np.max(np.abs(pred_after - pred_before)))
    if parity > 1e-7:
        raise RuntimeError(f"SPS head changed point prediction: {parity}")
    selected = best["best"]
    dev_candidate = adaptive_score(pred_after, target_after, sigma_after, selected["floor"], selected["mult_u"], selected["mult_v"], selected["rel"])
    dev_gain = dev_candidate["sps"] - static_best["sps"]

    _, pred_h, _, sigma_h, target_h = collect(model, head, holdout_paths, device, args.workers)
    hold_candidate = adaptive_score(pred_h, target_h, sigma_h, selected["floor"], selected["mult_u"], selected["mult_v"], selected["rel"])
    hold_static = static_score(pred_h, target_h, static_best["abs"], static_best["rel"])
    gate = {
        "status": "GO" if dev_gain >= 1.0 and dev_candidate["mean_width_uv"] <= 1.20 * static_best["mean_width_uv"] else "NO_GO",
        "dev_sps_gain_vs_static": dev_gain,
        "min_dev_sps_gain": 1.0,
        "dev_width_ratio": dev_candidate["mean_width_uv"] / max(static_best["mean_width_uv"], 1e-12),
        "max_dev_width_ratio": 1.20,
        "point_prediction_parity_max_abs": parity,
        "holdout_sps_delta_vs_static": hold_candidate["sps"] - hold_static["sps"],
        "holdout_coverage_delta_vs_static": hold_candidate["coverage"] - hold_static["coverage"],
    }

    torch.save({"head_state_dict": best_state, "head_config": cfg.__dict__, "selected_step": best["step"], "calibration": selected}, args.out_dir / "head_best.pth")
    dump(args.out_dir / "training_progress.json", {"losses": losses, "evals": evals})
    dump(args.out_dir / "summary.json", {
        "status": "REVIEW_REQUIRED",
        "recipe": "colleague80 residual-aware SPS: h64/b2/logmae/base-pre-residual features/final-error target",
        "train_stride": 5, "dev_stride": 20, "updates": UPDATES,
        "selected_step": best["step"], "selected_calibration": selected,
        "static_dev": static_best, "candidate_dev": dev_candidate,
        "static_holdout_aoa10": hold_static, "candidate_holdout_aoa10": hold_candidate,
        "gate": gate, "point_prediction_parity_max_abs": parity,
        "training_wall_seconds": time.monotonic() - started,
        "holdout_used_for_selection": False, "codabench_accessed": False, "locked_final_accessed": False,
    })
    (args.out_dir / "DONE").touch()


if __name__ == "__main__":
    main()
