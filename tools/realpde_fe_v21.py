#!/usr/bin/env python3
"""Reproducible Feature Engineering V2.1 runner for RealPDE Track 1.

The runner intentionally keeps the official three-channel CNO intact.  Feature
groups feed an identically-shaped, zero-initialised residual head, which makes
the raw-control / feature comparisons interpretable.  It uses only the scored
input window at inference time and writes official-v9 raw metrics and
subscores without inventing a leaderboard composite.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn

import realpde_loss_official_v9 as core


N2_WEIGHTS = {"mse": 1.0, "tke": 0.05, "rel": 0.027514, "mvpe": 0.009757,
              "mean": 0.0, "fluct": 0.0}
FEATURE_GROUPS = ("baseline", "raw_control", "temporal", "spatial", "pixel")
FEATURE_LABELS = {
    "baseline": "FE-00-CNO-Baseline", "raw_control": "FE-00R-ResidualRaw-Control",
    "temporal": "FE-01-Temporal", "spatial": "FE-02-SpatialPhysics",
    "pixel": "FE-03-PixelPosition",
}
C_MAX = 8


def sha256(path: Path) -> str:
    return core.sha256(path)


def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("FE_SOURCE_GIT_COMMIT")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Checkpoints are loaded with map_location=device for the model and
    # optimizer; RNG snapshots must nevertheless remain CPU ByteTensors.
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


@dataclass(frozen=True)
class FeatureConfig:
    group: str
    c_max: int = C_MAX
    smoothing_kernel: int = 3
    clipping_abs: float = 10.0
    derivative_space: str = "pixel-space; spacing=1 normalized pixel"
    mask_policy: str = "invalid iff u==0 and v==0 at the same pixel for all 20 input frames"


class FeatureNormalizer(nn.Module):
    def __init__(self, mean: Tensor | None = None, std: Tensor | None = None):
        super().__init__()
        self.register_buffer("mean", torch.zeros(C_MAX) if mean is None else mean.float())
        self.register_buffer("std", torch.ones(C_MAX) if std is None else std.float().clamp_min(1e-6))

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean.view(1, 1, 1, 1, -1)) / self.std.view(1, 1, 1, 1, -1)


def _pad(channels: list[Tensor]) -> Tensor:
    x = torch.stack(channels, dim=-1)
    if x.shape[-1] > C_MAX:
        raise ValueError(f"feature count {x.shape[-1]} exceeds C_MAX={C_MAX}")
    if x.shape[-1] < C_MAX:
        x = torch.nn.functional.pad(x, (0, C_MAX - x.shape[-1]))
    return x


def _masked_difference(x: Tensor, valid: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    """Pixel-space derivative, one-sided only at outer image boundaries.

    A pixel is invalid only when *both* velocity components are exactly zero
    through the complete 20-frame input history.  Across a valid/invalid
    interface this uses a valid-neighbour one-sided difference, never a
    central difference against zero-filled data.
    """
    forward_x, backward_x = torch.roll(x, -1, dims=dim), torch.roll(x, 1, dims=dim)
    forward_ok, backward_ok = torch.roll(valid, -1, dims=dim), torch.roll(valid, 1, dims=dim)
    edge0 = [slice(None)] * x.ndim; edge1 = [slice(None)] * x.ndim
    edge0[dim], edge1[dim] = 0, -1
    forward_ok[tuple(edge1)] = False
    backward_ok[tuple(edge0)] = False
    central_ok = valid & forward_ok & backward_ok
    forward_only = valid & forward_ok & ~backward_ok
    backward_only = valid & backward_ok & ~forward_ok
    derivative_ok = central_ok | forward_only | backward_only
    value = torch.zeros_like(x)
    value = torch.where(central_ok, 0.5 * (forward_x - backward_x), value)
    value = torch.where(forward_only, forward_x - x, value)
    value = torch.where(backward_only, x - backward_x, value)
    return value, derivative_ok


def build_features(x: Tensor, config: FeatureConfig) -> Tensor:
    """Construct a [B,T,H,W,C_MAX] tensor from a scored input only."""
    if x.ndim != 5 or x.shape[-1] < 3:
        raise ValueError("expected official [B,T,H,W,C>=3] input")
    if not torch.isfinite(x[..., :3]).all():
        raise ValueError("non-finite scored input is unsupported")
    u, v = x[..., 0], x[..., 1]
    if config.group == "raw_control":
        return _pad([u, v, x[..., 2]])
    if config.group == "temporal":
        du, dv = torch.zeros_like(u), torch.zeros_like(v)
        du[:, 1:], dv[:, 1:] = u[:, 1:] - u[:, :-1], v[:, 1:] - v[:, :-1]
        mean_u, mean_v = u.mean(1, keepdim=True), v.mean(1, keepdim=True)
        std_u, std_v = u.std(1, keepdim=True, unbiased=False), v.std(1, keepdim=True, unbiased=False)
        t = torch.linspace(-1.0, 1.0, x.shape[1], device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        denom = t.square().sum(dim=1, keepdim=True).clamp_min(1e-6)
        slope_u = (u * t).sum(dim=1, keepdim=True) / denom
        slope_v = (v * t).sum(dim=1, keepdim=True) / denom
        rep = lambda z: z.expand_as(u)
        return _pad([du, dv, rep(mean_u), rep(mean_v), rep(std_u), rep(std_v), rep(slope_u), rep(slope_v)])
    if config.group == "spatial":
        # Fixed 3x3 mean smoothing is applied only to derivative features.
        uv = torch.stack([u, v], dim=1)  # [B,2,T,H,W]
        smooth = torch.nn.functional.avg_pool3d(uv, kernel_size=(1, config.smoothing_kernel, config.smoothing_kernel),
                                                stride=1, padding=(0, config.smoothing_kernel // 2, config.smoothing_kernel // 2))
        su, sv = smooth[:, 0], smooth[:, 1]
        # Do not interpret one zero-velocity frame as a body.  The history rule
        # is reconstructible inside submission.py from the scored input alone.
        invalid = ((u == 0) & (v == 0)).all(dim=1, keepdim=True)
        valid = (~invalid).expand_as(u) & torch.isfinite(x[..., :2]).all(dim=-1)
        du_dx, x_ok = _masked_difference(su, valid, -1)
        du_dy, y_ok = _masked_difference(su, valid, -2)
        dv_dx, _ = _masked_difference(sv, valid, -1)
        dv_dy, _ = _masked_difference(sv, valid, -2)
        vort, div = dv_dx - du_dy, du_dx + dv_dy
        strain = torch.sqrt(du_dx.square() + dv_dy.square() + 0.5 * (du_dy + dv_dx).square())
        derivative_valid = (x_ok & y_ok).to(x.dtype)
        result = _pad([du_dx, du_dy, dv_dx, dv_dy, vort, div, strain, derivative_valid])
        return result.clamp(-config.clipping_abs, config.clipping_abs)
    if config.group == "pixel":
        h, w = x.shape[2], x.shape[3]
        yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand_as(u)
        xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand_as(u)
        return _pad([xx, yy])
    raise ValueError(f"features are not defined for group={config.group!r}")


@torch.no_grad()
def fit_normalizer(paths: list[Path], args: argparse.Namespace, config: FeatureConfig) -> FeatureNormalizer:
    """Fit per-channel feature moments on train windows only."""
    _, loader = core.loader(paths, args, shuffle=False)
    total = torch.zeros(C_MAX, dtype=torch.float64)
    total_sq = torch.zeros(C_MAX, dtype=torch.float64)
    count = 0
    for x, _, _, _ in loader:
        f = build_features(x, config).double()
        total += f.sum((0, 1, 2, 3))
        total_sq += f.square().sum((0, 1, 2, 3))
        count += int(np.prod(f.shape[:4]))
    mean = total / max(count, 1)
    var = (total_sq / max(count, 1) - mean.square()).clamp_min(1e-12)
    return FeatureNormalizer(mean.float(), var.sqrt().float())


class ResidualHead(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.first = nn.Conv3d(C_MAX, hidden, 1)
        self.activation = nn.SiLU()
        self.last = nn.Conv3d(hidden, 2, 1)
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    def forward(self, features: Tensor) -> Tensor:
        return self.last(self.activation(self.first(features)))


class FEModel(nn.Module):
    def __init__(self, cno: nn.Module, group: str, config: FeatureConfig | None = None,
                 normalizer: FeatureNormalizer | None = None):
        super().__init__()
        self.cno, self.group = cno, group
        self.config, self.normalizer = config, normalizer
        self.head = None if group == "baseline" else ResidualHead()

    def forward(self, x: Tensor) -> Tensor:
        base = self.cno(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        if self.head is None:
            return base
        assert self.config is not None and self.normalizer is not None
        features = self.normalizer(build_features(x, self.config))
        delta = self.head(features.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        result = base.clone()
        result[..., :2] = result[..., :2] + delta
        return result


def load_model(kit_root: Path, checkpoint: Path, group: str, config: FeatureConfig | None,
               normalizer: FeatureNormalizer | None, device: torch.device) -> FEModel:
    sys.path.insert(0, str(kit_root))
    from rpde_baselines.model.cno import CNO3d
    cno = CNO3d(in_dim=3, out_dim=3, out_dim_mult=1, in_size=64, N_layers=3).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cno.load_state_dict(state.get("model_state_dict", state), strict=True)
    return FEModel(cno, group, config, normalizer).to(device)


@torch.no_grad()
def evaluate(model: FEModel, paths: list[Path], args: argparse.Namespace, device: torch.device,
             kit_root: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ds, loader = core.loader(paths, args, shuffle=False)
    model.eval(); predictions: list[np.ndarray] = []; targets: list[np.ndarray] = []; elapsed = 0.0
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda": torch.cuda.synchronize()
        started = time.perf_counter(); prediction = model(x)
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        predictions.append(prediction.cpu().numpy().astype(np.float32)); targets.append(y.numpy().astype(np.float32))
    pred, target = np.concatenate(predictions), np.concatenate(targets)
    result = core.score_bundle(kit_root, pred, target, elapsed / len(ds), out)
    rows, anatomy = core.trajectory_rows(ds, pred, target, kit_root)
    result.update({"windows": len(ds), "trajectories": len(rows), "trajectory_anatomy": anatomy,
                   "local_model_plus_features_s_per_window": elapsed / len(ds)})
    with (out / "trajectory_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return result


def checkpoint_payload(model: FEModel, optimizer: torch.optim.Optimizer, elapsed: float, step: int,
                       batches_seen: int, config: dict, history: list[dict]) -> dict:
    return {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None, "grad_scaler_state_dict": None, "rng_state": rng_state(),
            "elapsed_train_seconds": elapsed, "global_step": step, "batches_seen": batches_seen,
            "config": config, "history": history}


def atomic_torch_save(payload: dict, path: Path) -> None:
    """Write an interruption-safe checkpoint without exposing a partial file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _advance(loader: Iterable, batches: int):
    iterator = iter(loader)
    for _ in range(batches):
        try: next(iterator)
        except StopIteration: iterator = iter(loader); next(iterator)
    return iterator


def train_one(group: str, args: argparse.Namespace, locked: dict, resume: Path | None = None) -> dict:
    out = args.out_dir / FEATURE_LABELS[group]
    if out.exists() and resume is None:
        # A process can be interrupted after its immutable configuration is
        # written but before the first checkpoint. Permit an explicit, safe
        # fresh restart only for that config-only state; never overwrite an
        # existing checkpoint, evaluation, or completed summary.
        existing = {path.name for path in out.iterdir()}
        if not (args.restart_incomplete and existing <= {"config.json"}):
            raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    _, train_paths = core.read_manifest(args.manifest, "train")
    _, dev_paths = core.read_manifest(args.manifest, "dev")
    config = None if group == "baseline" else FeatureConfig(group)
    normalizer = None if config is None else fit_normalizer(train_paths, args, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    model = load_model(args.kit_root, args.checkpoint, group, config, normalizer, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_ds, loader = core.loader(train_paths, args, shuffle=True)
    initial_elapsed = 0.0; step = 0; batches_seen = 0; history: list[dict] = []
    if resume is not None:
        saved = torch.load(resume, map_location=device, weights_only=False)
        if saved["config"]["group"] != group or saved["config"]["locked_manifest_sha256"] != locked["manifest_sha256"]:
            raise ValueError("resume checkpoint is incompatible with locked configuration")
        model.load_state_dict(saved["model_state_dict"]); optimizer.load_state_dict(saved["optimizer_state_dict"])
        initial_elapsed, step, batches_seen = saved["elapsed_train_seconds"], saved["global_step"], saved["batches_seen"]
        history = list(saved.get("history", []))
        iterator = _advance(loader, batches_seen); restore_rng(saved["rng_state"])
    else:
        iterator = iter(loader)
    run_config = {"group": group, "feature_config": asdict(config) if config else None,
                  "normalizer": None if normalizer is None else {"mean": normalizer.mean.cpu().tolist(), "std": normalizer.std.cpu().tolist()},
                  "c_max": C_MAX, "residual_head_params": 0 if model.head is None else sum(p.numel() for p in model.head.parameters()),
                  "locked_manifest_sha256": locked["manifest_sha256"], "checkpoint_sha256": locked["checkpoint_sha256"],
                  "n2_weights": N2_WEIGHTS, "seed": args.seed, "batch_size": args.batch_size,
                  "checkpoint_rule": "last at requested train minutes", "mask_policy": config.mask_policy if config else None}
    core.json_dump(out / "config.json", run_config)
    train_elapsed = initial_elapsed
    session_started = time.monotonic()
    interruption_requested = False
    previous_handlers: dict[int, signal.Handlers] = {}

    def request_interruption(signum: int, _frame: object) -> None:
        nonlocal interruption_requested
        interruption_requested = True

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, request_interruption)

    def save_progress(reason: str) -> None:
        atomic_torch_save(checkpoint_payload(model, optimizer, train_elapsed, step, batches_seen,
                                             run_config, history), out / "progress.pth")
        core.json_dump(out / "progress.json", {"reason": reason, "global_step": step,
                                                "elapsed_train_seconds": train_elapsed,
                                                "updated_at": time.time()})

    stop_reason = "max_train_seconds"; next_eval = ((step // args.eval_interval) + 1) * args.eval_interval
    # Deliberately exclude dev scoring/checkpoint I/O from the 30-minute
    # optimizer-training budget; those costs remain separately observable.
    while train_elapsed < args.max_train_seconds:
        try: x, y, _, _ = next(iterator)
        except StopIteration: iterator = iter(loader); x, y, _, _ = next(iterator)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        update_started = time.monotonic()
        model.train(); parts = core.loss_parts(model(x), y); loss = sum(N2_WEIGHTS[k] * parts[k] for k in N2_WEIGHTS)
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if device.type == "cuda": torch.cuda.synchronize()
        train_elapsed += time.monotonic() - update_started
        step += 1; batches_seen += 1
        if step % args.checkpoint_interval == 0:
            save_progress("periodic")
        if interruption_requested:
            save_progress("preempted_by_queue")
            return None
        if args.max_session_seconds and time.monotonic() - session_started >= args.max_session_seconds:
            save_progress("session_time_slice")
            return None
        if step >= next_eval:
            dev = evaluate(model, dev_paths, args, device, args.kit_root, out / f"eval_{step:05d}")
            history.append({"iteration": step, "train_seconds": train_elapsed, **dev["raw_errors"]})
            next_eval += args.eval_interval
    elapsed = train_elapsed
    dev = evaluate(model, dev_paths, args, device, args.kit_root, out / f"eval_last_{step:05d}")
    history.append({"iteration": step, "train_seconds": elapsed, **dev["raw_errors"]})
    atomic_torch_save(checkpoint_payload(model, optimizer, elapsed, step, batches_seen, run_config, history), out / "last.pth")
    result = {"metadata": run_config | {"actual_updates": step, "train_seconds": elapsed, "train_windows": len(train_ds),
              "samples_seen": batches_seen * args.batch_size, "stop_reason": stop_reason}, "evaluation": dev, "history": history}
    core.json_dump(out / "summary.json", result)
    for signum, handler in previous_handlers.items():
        signal.signal(signum, handler)
    return result


def _read_trajectory(path: Path) -> dict[str, dict[str, float]]:
    with path.open() as f:
        return {r["trajectory_id"]: {m: float(r[m]) for m in ("rel_l2", "tke", "mvpe")} for r in csv.DictReader(f)}


def paired_statistics(control: Path, candidate: Path, out: Path, seed: int, draws: int) -> list[dict]:
    a, b = _read_trajectory(control), _read_trajectory(candidate)
    ids = sorted(set(a) & set(b)); rng = np.random.default_rng(seed); result = []
    for metric in ("rel_l2", "tke", "mvpe"):
        base, cand = np.asarray([a[i][metric] for i in ids]), np.asarray([b[i][metric] for i in ids])
        delta = base - cand; boot = rng.choice(delta, size=(draws, len(delta)), replace=True).mean(axis=1)
        result.append({"metric": metric, "trajectories": len(ids), "control_macro_mean": float(base.mean()),
                       "candidate_macro_mean": float(cand.mean()), "mean_delta": float(delta.mean()),
                       "win_rate": float((delta > 0).mean()), "bootstrap_95_low": float(np.percentile(boot, 2.5)),
                       "bootstrap_95_high": float(np.percentile(boot, 97.5))})
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0])); writer.writeheader(); writer.writerows(result)
    return result


def _history_at_common_step(history: list[dict], step: int) -> dict:
    eligible = [row for row in history if row["iteration"] <= step]
    return max(eligible, key=lambda row: row["iteration"])


def local_wrapper_latency(group: str, args: argparse.Namespace, locked: dict) -> dict | None:
    """Measure a local 3090 proxy, not the organiser's A800 Time Score."""
    if not torch.cuda.is_available():
        return None
    summary_path = args.out_dir / FEATURE_LABELS[group] / "summary.json"
    checkpoint_path = args.out_dir / FEATURE_LABELS[group] / "last.pth"
    if not (summary_path.exists() and checkpoint_path.exists()):
        return None
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    feature_cfg = FeatureConfig(**cfg["feature_config"]) if cfg["feature_config"] else None
    normalizer = None
    if cfg["normalizer"] is not None:
        normalizer = FeatureNormalizer(torch.tensor(cfg["normalizer"]["mean"]), torch.tensor(cfg["normalizer"]["std"]))
    device = torch.device("cuda")
    model = load_model(args.kit_root, args.checkpoint, group, feature_cfg, normalizer, device)
    model.load_state_dict(saved["model_state_dict"]); model.eval()
    _, dev_paths = core.read_manifest(args.manifest, "dev")
    _, loader = core.loader(dev_paths, args, shuffle=False)
    x, _, _, _ = next(iter(loader)); array = np.asarray(x[:1].numpy(), dtype=np.float32)

    @torch.no_grad()
    def call() -> np.ndarray:
        tensor = torch.from_numpy(np.asarray(array, dtype=np.float32)).to(device)
        return model(tensor).cpu().numpy()

    for _ in range(args.latency_warmup): call()
    timings = []
    for _ in range(args.latency_repeats):
        torch.cuda.synchronize(); started = time.perf_counter(); call(); torch.cuda.synchronize()
        timings.append(1000.0 * (time.perf_counter() - started))
    values = np.asarray(timings)
    return {"local_submission_latency_3090_mean_ms": float(values.mean()),
            "local_submission_latency_3090_median_ms": float(np.median(values)),
            "local_submission_latency_3090_p95_ms": float(np.percentile(values, 95)),
            "local_submission_latency_3090_std_ms": float(values.std()),
            "latency_scope": "numpy input conversion + feature construction + residual head + CNO + numpy output; local RTX 3090 proxy only"}


def write_summary_and_report(args: argparse.Namespace, locked: dict) -> None:
    summaries: dict[str, dict] = {}
    for group in FEATURE_GROUPS:
        path = args.out_dir / FEATURE_LABELS[group] / "summary.json"
        if path.exists(): summaries[group] = json.loads(path.read_text())
    if not summaries:
        return
    shared_step = min(value["metadata"]["actual_updates"] for value in summaries.values())
    paired: dict[str, list[dict]] = {}
    for group in ("temporal", "spatial", "pixel"):
        path = args.out_dir / f"paired_{group}_vs_raw_control.csv"
        if path.exists():
            with path.open() as f: paired[group] = list(csv.DictReader(f))
    rows = []
    for group, summary in summaries.items():
        metadata, evaluation = summary["metadata"], summary["evaluation"]
        shared = _history_at_common_step(summary["history"], shared_step)
        subscores = evaluation["official_v9_subscores"]
        row = {"experiment": FEATURE_LABELS[group], "feature_group": group, "seed": metadata["seed"],
               "start_checkpoint_sha": metadata["checkpoint_sha256"], "train_minutes": metadata["train_seconds"] / 60.0,
               "micro_batch": metadata["batch_size"], "effective_batch": metadata["batch_size"],
               "optimizer_steps": metadata["actual_updates"], "samples_seen": metadata["samples_seen"],
               "residual_head_params": metadata["residual_head_params"], "C_max": metadata["c_max"],
               "shared_step": shared_step, "rel_l2_30m": evaluation["raw_errors"]["rel_l2"],
               "tke_30m": evaluation["raw_errors"]["tke"], "mvpe_30m": evaluation["raw_errors"]["mvpe"],
               "rel_l2_shared": shared["rel_l2"], "tke_shared": shared["tke"], "mvpe_shared": shared["mvpe"],
               **subscores, "status": "COMPLETED"}
        if group in paired:
            for item in paired[group]:
                metric = item["metric"]
                row[f"trajectory_{metric}_macro"] = item["candidate_macro_mean"]
                row[f"trajectory_{metric}_win_rate"] = item["win_rate"]
                row[f"trajectory_{metric}_bootstrap_ci_low"] = item["bootstrap_95_low"]
                row[f"trajectory_{metric}_bootstrap_ci_high"] = item["bootstrap_95_high"]
        rows.append(row)
    columns = sorted({key for row in rows for key in row})
    with (args.out_dir / "feature_engineering_v2_1_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    lines = ["# RealPDE Track 1 — Feature Engineering V2.1", "", "## Scope", "",
             "This report uses the supplied Track 1 v9 scorer. It reports its five subscores and raw errors only; no unpublished leaderboard composite is estimated.",
             "", "## Locked setup", "", f"- Start checkpoint SHA-256: `{locked['checkpoint_sha256']}`", f"- Manifest SHA-256: `{locked['manifest_sha256']}`",
             f"- N2 weights: `{json.dumps(locked['n2_weights'], sort_keys=True)}`", f"- Shared comparison step: `{shared_step}`", "",
             "## Last-checkpoint results", "", "| Experiment | Rel-L2 | TKE | MVPE |", "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['experiment']} | {row['rel_l2_30m']:.6f} | {row['tke_30m']:.6f} | {row['mvpe_30m']:.6f} |")
    lines += ["", "## Paired trajectory analysis", "", "The candidate-minus-control statistics are in `paired_*_vs_raw_control.csv`. Positive `mean_delta` means lower candidate error.",
              "", "## Local timing", "", "Any timing produced here is a local RTX 3090 wrapper proxy, not the official A800 Time Score."]
    (args.out_dir / "report_feature_engineering_v2_1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_locked_manifest(args: argparse.Namespace) -> dict:
    source_manifest = json.loads(args.manifest.read_text())
    result = {"version": "FE-V2.1", "created_at": time.time(), "git_commit": _git_commit(Path(__file__).resolve().parents[1]),
              "runner_sha256": sha256(Path(__file__).resolve()),
              "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256(args.manifest),
              "data_root": source_manifest["real_root"], "split_counts": {k: len(source_manifest[k]) for k in ("train", "dev", "final")},
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
              "checkpoint_source": "official sim_pretrain CNO", "n2_weights": N2_WEIGHTS,
              "optimizer": {"name": "AdamW", "lr": args.lr}, "scheduler": None, "amp": False,
              "scorer": {"path": str((args.kit_root / "scoring.py").resolve()), "sha256": sha256(args.kit_root / "scoring.py"),
                         "subscores": ["rel_l2_score", "tke_score", "mvpe_score", "time_score", "sps_score"]},
              "in_shape": [20, 32, 64, 3], "checkpoint_rule": "last at requested train duration",
              "seed": args.seed, "bootstrap": {"seed": args.bootstrap_seed, "draws": args.bootstrap_draws, "unit": "trajectory", "paired": True},
              "thresholds": {"rel_l2_min_effect": 0.002, "mvpe_min_effect": 0.002, "tke_degradation_tolerance": 0.002,
                             "source": "pre-registered conservative absolute raw-error thresholds; no repeated FE history exists"},
              "feature_policy": "only scored input plus shape-derived constants; no metadata, CFD, target, Re, or AoA"}
    core.json_dump(args.out_dir / "fe_v2_1_locked_manifest.json", result)
    return result


def _feature_self_test() -> None:
    x = torch.zeros(2, 20, 32, 64, 3)
    x[..., 0] = torch.linspace(0, 1, 64).view(1, 1, 1, 64)
    x[..., 1] = torch.linspace(0, 1, 32).view(1, 1, 32, 1)
    for group in ("raw_control", "temporal", "spatial", "pixel"):
        value = build_features(x, FeatureConfig(group))
        assert value.shape == (2, 20, 32, 64, C_MAX) and torch.isfinite(value).all()
    # Persistent paired zeros are invalid; a low, non-zero valid flow remains valid.
    assert build_features(torch.zeros_like(x), FeatureConfig("spatial"))[..., 7].eq(0).all()
    assert build_features(torch.full_like(x, 1e-8), FeatureConfig("spatial"))[..., 7].eq(1).all()
    print("feature self-test: PASS")


def preflight(args: argparse.Namespace) -> None:
    """One warmup plus one full update per group; never reuses these objects."""
    if not torch.cuda.is_available():
        raise SystemExit("GPU preflight requires CUDA")
    _, train_paths = core.read_manifest(args.manifest, "train")
    device = torch.device("cuda")
    rows = []
    for group in args.experiments:
        set_seed(args.seed)
        config = None if group == "baseline" else FeatureConfig(group)
        # Statistics do not affect tensor shapes. Avoid scanning the full train
        # split during a memory-only probe; formal training fits them afresh.
        normalizer = None if config is None else FeatureNormalizer()
        model = load_model(args.kit_root, args.checkpoint, group, config, normalizer, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        _, loader = core.loader(train_paths, args, shuffle=False)
        x, y, _, _ = next(iter(loader)); x, y = x.to(device), y.to(device)
        torch.cuda.reset_peak_memory_stats()
        for timed in (False, True):
            optimizer.zero_grad(set_to_none=True)
            if timed: torch.cuda.synchronize(); started = time.perf_counter()
            parts = core.loss_parts(model(x), y); loss = sum(N2_WEIGHTS[k] * parts[k] for k in N2_WEIGHTS)
            loss.backward(); optimizer.step()
            if timed:
                torch.cuda.synchronize(); elapsed = time.perf_counter() - started
        rows.append({"group": group, "batch_size": args.batch_size, "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                     "step_seconds": elapsed, "samples_per_second": args.batch_size / elapsed,
                     "residual_head_params": 0 if model.head is None else sum(p.numel() for p in model.head.parameters())})
        del model, optimizer, x, y
        torch.cuda.empty_cache()
    core.json_dump(args.out_dir / "preflight.json", {"gpu": torch.cuda.get_device_name(), "rows": rows})
    print(json.dumps(rows, indent=2), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--kit-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--mode", choices=("lock", "test", "preflight", "train", "analyze"), default="train")
    p.add_argument("--experiments", nargs="+", choices=FEATURE_GROUPS, default=list(FEATURE_GROUPS))
    p.add_argument("--seed", type=int, default=20260901); p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2); p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-5); p.add_argument("--max-train-seconds", type=float, default=1800.0)
    p.add_argument("--eval-interval", type=int, default=500); p.add_argument("--bootstrap-seed", type=int, default=20260901)
    p.add_argument("--bootstrap-draws", type=int, default=2000); p.add_argument("--resume", type=Path)
    p.add_argument("--restart-incomplete", action="store_true",
                   help="restart only an output directory containing at most config.json; refuses checkpoints/results")
    p.add_argument("--checkpoint-interval", type=int, default=100,
                   help="optimizer updates between atomic resumable checkpoints")
    p.add_argument("--max-session-seconds", type=float, default=0.0,
                   help="optional wall-time lease; exits resumably before the full optimizer budget")
    p.add_argument("--latency-warmup", type=int, default=10); p.add_argument("--latency-repeats", type=int, default=50)
    args = p.parse_args()
    if args.max_train_seconds <= 0 or args.max_session_seconds < 0 or args.batch_size < 1 or args.checkpoint_interval < 1:
        p.error("training/session seconds, checkpoint interval, and batch size must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "test": _feature_self_test(); return
    locked_path = args.out_dir / "fe_v2_1_locked_manifest.json"
    locked = create_locked_manifest(args) if args.mode == "lock" or not locked_path.exists() else json.loads(locked_path.read_text())
    if args.mode == "lock": return
    if args.mode == "preflight": preflight(args); return
    if args.mode == "train":
        completed = True
        for group in args.experiments:
            result = train_one(group, args, locked, args.resume if len(args.experiments) == 1 else None)
            completed = completed and result is not None
        if not completed:
            return
    # Analyze uses the last evaluation directories recorded by the summaries.
    if args.mode in ("train", "analyze"):
        for group in ("temporal", "spatial", "pixel"):
            summary = args.out_dir / FEATURE_LABELS[group] / "summary.json"
            raw = args.out_dir / FEATURE_LABELS["raw_control"] / "summary.json"
            if not (summary.exists() and raw.exists()): continue
            step_a = json.loads(raw.read_text())["metadata"]["actual_updates"]
            step_b = json.loads(summary.read_text())["metadata"]["actual_updates"]
            p_a = args.out_dir / FEATURE_LABELS["raw_control"] / f"eval_last_{step_a:05d}" / "trajectory_metrics.csv"
            p_b = args.out_dir / FEATURE_LABELS[group] / f"eval_last_{step_b:05d}" / "trajectory_metrics.csv"
            if p_a.exists() and p_b.exists(): paired_statistics(p_a, p_b, args.out_dir / f"paired_{group}_vs_raw_control.csv", args.bootstrap_seed, args.bootstrap_draws)
        if args.mode == "train":
            # Full wrapper latency is cheap relative to training; retain all five
            # local proxies instead of using timing as a selection criterion.
            for group in args.experiments:
                timing = local_wrapper_latency(group, args, locked)
                if timing is not None:
                    core.json_dump(args.out_dir / FEATURE_LABELS[group] / "local_submission_latency_3090.json", timing)
        write_summary_and_report(args, locked)


if __name__ == "__main__":
    main()
