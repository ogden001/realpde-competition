#!/usr/bin/env python3
"""Frozen FE-04 RawSpatial8 diagnostic and training runner.

The only new model is a 178-parameter residual head which receives
``u,v,p,du/dx,du/dy,dv/dx,dv/dy,derivative_valid``.  Gradient construction is
delegated to the already-run FE-02 implementation in ``realpde_fe_v21`` so a
future change cannot silently create a second SpatialPhysics definition.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

import realpde_fe_v21 as fe
import realpde_loss_official_v9 as core


LABEL = "FE-04-RawSpatial8"
VERSION = "FE-04-RawSpatial8-V1.1-r1"
N2 = fe.N2_WEIGHTS
C_MAX = 8


def json_dump(path: Path, value: object) -> None:
    core.json_dump(path, value)


def sha256(path: Path) -> str:
    return core.sha256(path)


def raw_spatial8(x: Tensor, config: fe.FeatureConfig) -> Tensor:
    """Raw channels plus the four FE-02 primitives and its validity channel."""
    if config.group != "spatial":
        raise ValueError("RawSpatial8 must use the frozen FE-02 spatial config")
    spatial = fe.build_features(x, config)  # exact FE-02 smoothing/mask/difference/clipping path
    result = torch.stack([
        x[..., 0], x[..., 1], x[..., 2], spatial[..., 0], spatial[..., 1],
        spatial[..., 2], spatial[..., 3], spatial[..., 7],
    ], dim=-1)
    if result.shape[-1] != C_MAX:
        raise AssertionError("RawSpatial8 channel count changed")
    return result


class RawSpatial8Model(nn.Module):
    def __init__(self, cno: nn.Module, config: fe.FeatureConfig, normalizer: fe.FeatureNormalizer):
        super().__init__()
        self.cno, self.config, self.normalizer = cno, config, normalizer
        self.head = fe.ResidualHead()

    def parts(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        base = self.cno(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        features = self.normalizer(raw_spatial8(x, self.config))
        delta = self.head(features.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        full = base.clone(); full[..., :2] += delta
        return base, delta, full

    def forward(self, x: Tensor) -> Tensor:
        return self.parts(x)[2]


def spatial_config() -> fe.FeatureConfig:
    return fe.FeatureConfig("spatial")


@torch.no_grad()
def fit_normalizer(paths: list[Path], args: argparse.Namespace, config: fe.FeatureConfig) -> fe.FeatureNormalizer:
    _, loader = core.loader(paths, args, shuffle=False)
    total = torch.zeros(C_MAX, dtype=torch.float64); total_sq = torch.zeros(C_MAX, dtype=torch.float64); count = 0
    for x, _, _, _ in loader:
        values = raw_spatial8(x, config).double()
        total += values.sum((0, 1, 2, 3)); total_sq += values.square().sum((0, 1, 2, 3)); count += int(np.prod(values.shape[:4]))
    mean = total / max(count, 1); variance = (total_sq / max(count, 1) - mean.square()).clamp_min(1e-12)
    return fe.FeatureNormalizer(mean.float(), variance.sqrt().float())


def load_model(kit_root: Path, initial_checkpoint: Path, config: fe.FeatureConfig,
               normalizer: fe.FeatureNormalizer, device: torch.device) -> RawSpatial8Model:
    sys.path.insert(0, str(kit_root))
    from rpde_baselines.model.cno import CNO3d
    cno = CNO3d(in_dim=3, out_dim=3, out_dim_mult=1, in_size=64, N_layers=3).to(device)
    state = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
    cno.load_state_dict(state.get("model_state_dict", state), strict=True)
    return RawSpatial8Model(cno, config, normalizer).to(device)


def normalizer_dict(normalizer: fe.FeatureNormalizer) -> dict:
    return {"mean": normalizer.mean.cpu().tolist(), "std": normalizer.std.cpu().tolist()}


def locked_spec(args: argparse.Namespace) -> dict:
    manifest = json.loads(args.manifest.read_text())
    spec = {
        "version": VERSION,
        "created_at": time.time(),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "fe02_source_runner_sha256": sha256(Path(fe.__file__).resolve()),
        "reference_id": "T1-ID-FE-N2-30M-S20260901",
        "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256(args.manifest),
        "split_counts": {name: len(manifest[name]) for name in ("train", "dev", "final")},
        "allowed_splits": ["train", "dev"], "locked_final_access": False,
        "initial_checkpoint": str(args.checkpoint.resolve()), "initial_checkpoint_sha256": sha256(args.checkpoint),
        "scorer": {"path": str((args.kit_root / "scoring.py").resolve()), "sha256": sha256(args.kit_root / "scoring.py")},
        "seed": args.seed, "batch_size": 18, "optimizer": {"name": "AdamW", "lr": args.lr}, "scheduler": None,
        "loss": N2, "active_optimizer_seconds": 1800, "checkpoint_rule": "last@30min active optimizer time",
        "evaluation_active_seconds": [600, 1200, 1800],
        "feature": {
            "channels": ["u", "v", "p", "du_dx", "du_dy", "dv_dx", "dv_dy", "derivative_valid"],
            "raw_policy": "identical raw u/v/p channels to FE-00R; no temporal aggregation",
            "spatial_source": "FE-02 realpde_fe_v21.build_features(group=spatial)",
            "feature_config": asdict(spatial_config()), "normalizer": "re-fit per channel on train windows only",
        },
        "head": {"architecture": "Conv3d(8,16,1) -> SiLU -> Conv3d(16,2,1)", "params": 178,
                 "last_layer_zero_init": True, "extra_normalization": None},
        "comparison": {"primary": "FE-00R-ResidualRaw-Control last@30min", "secondary": "FE-02-SpatialPhysics last@30min"},
        "hard_gate": {"rel_l2": "FE04 <= raw_control - 0.002", "mvpe": "FE04 <= raw_control + 0.002",
                      "tke": "FE04 <= raw_control + 0.002", "then": "paired trajectory bootstrap consistency"},
        "bootstrap": {"unit": "trajectory", "paired": True, "draws": args.bootstrap_draws, "seed": args.bootstrap_seed},
        "latency": "only if hard gate passes; local RTX 3090 wrapper proxy, not official A800 Time Score",
        "gpu_policy": "keep batch=18 under external contention; preempt/resume. Exclusive OOM aborts as LIMITED_COMPARABILITY, no automatic batch fallback.",
    }
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    spec["spec_sha256"] = hashlib.sha256(encoded).hexdigest()
    return spec


def get_locked(args: argparse.Namespace, create: bool = False) -> dict:
    path = args.out_dir / "fe04_locked_spec.json"
    if create:
        if path.exists():
            # A lock includes its creation time and is immutable by design.  Do
            # not regenerate it on a queue restart: returning it is the only
            # way to avoid an accidental post-diagnostic re-registration.
            return json.loads(path.read_text())
        value = locked_spec(args); json_dump(path, value); return value
    if not path.exists():
        raise FileNotFoundError("run --mode lock before diagnostics or training")
    return json.loads(path.read_text())


def checkpoint_payload(model: RawSpatial8Model, optimizer: torch.optim.Optimizer, elapsed: float,
                       step: int, batches_seen: int, config: dict, history: list[dict]) -> dict:
    return {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": fe.rng_state(), "elapsed_train_seconds": elapsed, "global_step": step,
            "batches_seen": batches_seen, "config": config, "history": history}


def atomic_save(payload: dict, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp"); torch.save(payload, temp); os.replace(temp, path)


def advance(loader, batches: int):
    iterator = iter(loader)
    for _ in range(batches):
        try: next(iterator)
        except StopIteration: iterator = iter(loader); next(iterator)
    return iterator


@torch.no_grad()
def evaluate(model: RawSpatial8Model, paths: list[Path], args: argparse.Namespace, device: torch.device,
             out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True); ds, loader = core.loader(paths, args, shuffle=False)
    prediction, target, elapsed = [], [], 0.0; model.eval()
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda": torch.cuda.synchronize()
        started = time.perf_counter(); value = model(x)
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        prediction.append(value.cpu().numpy().astype(np.float32)); target.append(y.numpy().astype(np.float32))
    pred, target_np = np.concatenate(prediction), np.concatenate(target)
    result = core.score_bundle(args.kit_root, pred, target_np, elapsed / len(ds), out)
    rows, anatomy = core.trajectory_rows(ds, pred, target_np, args.kit_root)
    result |= {"windows": len(ds), "trajectories": len(rows), "trajectory_anatomy": anatomy,
               "local_model_plus_features_s_per_window": elapsed / len(ds)}
    write_rows(out / "trajectory_metrics.csv", rows)
    return result


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows: return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def train(args: argparse.Namespace, spec: dict) -> None:
    out = args.out_dir / LABEL; out.mkdir(parents=True, exist_ok=True)
    progress, final = out / "progress.pth", out / "last.pth"
    if final.exists():
        print("FE-04 already complete", flush=True); return
    _, train_paths = core.read_manifest(args.manifest, "train"); _, dev_paths = core.read_manifest(args.manifest, "dev")
    config = spatial_config(); fe.set_seed(args.seed)
    # A preempted slice has already fitted and checkpointed the strictly
    # train-only normalizer.  Reusing those immutable moments avoids spending
    # several CPU-only minutes before every lease; it does not use dev/final
    # statistics and does not change the resumed model definition.
    resume_config = None
    if progress.exists():
        resume_config = torch.load(progress, map_location="cpu", weights_only=False)["config"]
        if resume_config.get("spec_sha256") != spec["spec_sha256"]:
            raise ValueError("resume checkpoint has another locked spec")
        normalizer = fe.FeatureNormalizer(torch.tensor(resume_config["normalizer"]["mean"]),
                                           torch.tensor(resume_config["normalizer"]["std"]))
    else:
        normalizer = fit_normalizer(train_paths, args, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.kit_root, args.checkpoint, config, normalizer, device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_ds, loader = core.loader(train_paths, args, shuffle=True); elapsed = 0.0; step = batches_seen = 0; history: list[dict] = []
    if progress.exists():
        saved = torch.load(progress, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state_dict"]); optimizer.load_state_dict(saved["optimizer_state_dict"])
        elapsed, step, batches_seen, history = saved["elapsed_train_seconds"], saved["global_step"], saved["batches_seen"], list(saved.get("history", []))
        iterator = advance(loader, batches_seen); fe.restore_rng(saved["rng_state"])
    else: iterator = iter(loader)
    run_config = {"label": LABEL, "spec_sha256": spec["spec_sha256"], "feature_config": asdict(config),
                  "normalizer": normalizer_dict(normalizer), "normalizer_fit_split": "train",
                  "normalizer_file_sha256": None, "residual_head_params": sum(x.numel() for x in model.head.parameters()),
                  "batch_size": args.batch_size, "seed": args.seed, "n2_weights": N2,
                  "initial_checkpoint_sha256": spec["initial_checkpoint_sha256"], "checkpoint_rule": spec["checkpoint_rule"]}
    json_dump(out / "config.json", run_config)
    json_dump(out / "feature_normalizer.json", run_config["normalizer"] | {"fit_split": "train", "channels": spec["feature"]["channels"], "clipping_abs": config.clipping_abs})
    run_config["normalizer_file_sha256"] = sha256(out / "feature_normalizer.json"); json_dump(out / "config.json", run_config)
    milestones = list(spec["evaluation_active_seconds"])
    completed_milestones = {int(x["scheduled_active_seconds"]) for x in history if "scheduled_active_seconds" in x}
    interrupted = False; started_session = time.monotonic()
    def on_signal(_signum, _frame):
        nonlocal interrupted; interrupted = True
    old = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGTERM, signal.SIGINT)}
    def save(reason: str):
        atomic_save(checkpoint_payload(model, optimizer, elapsed, step, batches_seen, run_config, history), progress)
        json_dump(out / "progress.json", {"reason": reason, "global_step": step, "elapsed_train_seconds": elapsed, "updated_at": time.time()})
    while elapsed < spec["active_optimizer_seconds"]:
        try: x, y, _, _ = next(iterator)
        except StopIteration: iterator = iter(loader); x, y, _, _ = next(iterator)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True); update_started = time.monotonic()
        model.train(); parts = core.loss_parts(model(x), y); loss = sum(N2[name] * parts[name] for name in N2)
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed += time.monotonic() - update_started; step += 1; batches_seen += 1
        if step % args.checkpoint_interval == 0: save("periodic")
        due = [value for value in milestones if elapsed >= value and value not in completed_milestones]
        for value in due:
            dev = evaluate(model, dev_paths, args, device, out / f"eval_active_{value // 60:02d}m")
            history.append({"iteration": step, "scheduled_active_seconds": value, "train_seconds": elapsed, **dev["raw_errors"]}); completed_milestones.add(value)
            save(f"evaluation_{value}s")
        if interrupted: save("preempted_by_queue"); return
        if args.max_session_seconds and time.monotonic() - started_session >= args.max_session_seconds: save("session_time_slice"); return
    dev = evaluate(model, dev_paths, args, device, out / "eval_last_30m")
    history.append({"iteration": step, "scheduled_active_seconds": 1800, "train_seconds": elapsed, **dev["raw_errors"]})
    atomic_save(checkpoint_payload(model, optimizer, elapsed, step, batches_seen, run_config, history), final)
    json_dump(out / "summary.json", {"metadata": run_config | {"actual_updates": step, "train_seconds": elapsed, "train_windows": len(train_ds), "samples_seen": batches_seen * args.batch_size, "stop_reason": "max_active_optimizer_seconds"}, "evaluation": dev, "history": history})
    for sig, handler in old.items(): signal.signal(sig, handler)


def load_v21(path: Path, kit_root: Path, initial_checkpoint: Path, device: torch.device) -> fe.FEModel:
    saved = torch.load(path, map_location=device, weights_only=False); cfg = saved["config"]
    feature_cfg = fe.FeatureConfig(**cfg["feature_config"])
    normalizer = fe.FeatureNormalizer(torch.tensor(cfg["normalizer"]["mean"]), torch.tensor(cfg["normalizer"]["std"]))
    model = fe.load_model(kit_root, initial_checkpoint, cfg["group"], feature_cfg, normalizer, device)
    model.load_state_dict(saved["model_state_dict"]); return model.eval()


@torch.no_grad()
def collect_diagnostic(raw_model: fe.FEModel, spatial_model: fe.FEModel, paths: list[Path], args: argparse.Namespace,
                       device: torch.device) -> tuple[object, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ds, loader = core.loader(paths, args, shuffle=False); raws=[]; bases=[]; deltas=[]; targets=[]
    for x, y, _, _ in loader:
        x = x.to(device)
        raw = raw_model(x)
        base = spatial_model.cno(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        f = spatial_model.normalizer(fe.build_features(x, spatial_model.config))
        delta = spatial_model.head(f.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        raws.append(raw.cpu().numpy().astype(np.float32)); bases.append(base.cpu().numpy().astype(np.float32)); deltas.append(delta.cpu().numpy().astype(np.float32)); targets.append(y.numpy().astype(np.float32))
    return ds, np.concatenate(raws), np.concatenate(bases), np.concatenate(deltas), np.concatenate(targets)


def decomposed_windows(ds, base: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the same ordered-window concatenation as core.trajectory_rows.

    v9 itself aggregates independent scored windows.  This expansion is solely
    the existing V2.1 trajectory diagnostic convention, and deliberately does
    not pretend to be an undisclosed official prediction stitcher.
    """
    mean, fluct, full = np.empty_like(base), np.empty_like(base), base.copy(); full[..., :2] += delta
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index, ref in enumerate(ds.refs): groups[ref.path.name].append((ref.start, index))
    for refs in groups.values():
        order = [i for _, i in sorted(refs)]; joined = np.concatenate([delta[i] for i in order], axis=0)
        correction_mean = joined.mean(axis=0, keepdims=True); cursor = 0
        for index in order:
            n = delta[index].shape[0]; d_mean = np.broadcast_to(correction_mean, delta[index].shape); d_fluc = delta[index] - d_mean
            mean[index] = base[index]; mean[index, ..., :2] += d_mean
            fluct[index] = base[index]; fluct[index, ..., :2] += d_fluc
            cursor += n
    return full, mean, fluct


def metric_rows(ds, pred: np.ndarray, target: np.ndarray, kit_root: Path) -> list[dict]:
    return core.trajectory_rows(ds, pred, target, kit_root)[0]


def paired(control: list[dict], candidate: list[dict], seed: int, draws: int) -> list[dict]:
    a = {r["trajectory_id"]: r for r in control}; b = {r["trajectory_id"]: r for r in candidate}; ids = sorted(a.keys() & b.keys()); rng = np.random.default_rng(seed); rows=[]
    for metric in ("rel_l2", "tke", "mvpe"):
        values = np.array([float(a[i][metric]) - float(b[i][metric]) for i in ids]); boot = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
        rows.append({"metric": metric, "trajectories": len(ids), "mean_delta_control_minus_candidate": float(values.mean()), "win_rate_candidate_lower": float((values > 0).mean()), "bootstrap_95_low": float(np.percentile(boot, 2.5)), "bootstrap_95_high": float(np.percentile(boot, 97.5))})
    return rows


def tke_replay(ds, pred: np.ndarray, target: np.ndarray, kit_root: Path, expected: float) -> tuple[dict, list[dict]]:
    sys.path.insert(0, str(kit_root)); import scoring
    # Keep NumPy's float32 accumulation for the replay: that is exactly what
    # scoring.py's ``float(np.mean(...))`` does.  Cast only after the official
    # aggregate has been reproduced, then use float64 for contribution tables.
    raw_per_window = scoring.tke_rel_l2_per_sample(pred, target, scoring.measured_channels(target))
    replay = float(np.mean(raw_per_window))
    per_window = raw_per_window.astype(float)
    if not np.isclose(replay, expected, rtol=0, atol=1e-10): raise RuntimeError(f"TKE replay mismatch: {replay} != {expected}")
    grouped: dict[str, list[float]] = defaultdict(list)
    for ref, value in zip(ds.refs, per_window): grouped[ref.path.name].append(float(value))
    total_n, total_sum = len(per_window), float(per_window.sum()); rows=[]
    for name, values in sorted(grouped.items()):
        subtotal, count = float(sum(values)), len(values); loo = (total_sum - subtotal) / (total_n - count)
        rows.append({"trajectory_id": name, "num_windows": count, "equivalent_term": "sum_of_per_window_tke_relative_l2", "aggregate_numerator_contrib": subtotal, "aggregate_denominator_contrib": count, "aggregate_mean_contrib": subtotal / total_n, "aggregate_without_trajectory": loo, "delta_loo_vs_full": loo - replay})
    return {"aggregate": replay, "tolerance": 1e-10, "formula": "mean over all scored windows of tke_rel_l2_per_sample; each sample is a norm ratio, so no global physical numerator/denominator exists", "windows": total_n}, rows


def diagnose(args: argparse.Namespace, spec: dict) -> None:
    source = args.v21_out; required = {"raw": source / "FE-00R-ResidualRaw-Control/last.pth", "spatial": source / "FE-02-SpatialPhysics/last.pth"}
    if not all(path.exists() for path in required.values()): raise FileNotFoundError(required)
    _, dev_paths = core.read_manifest(args.manifest, "dev"); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_model = load_v21(required["raw"], args.kit_root, args.checkpoint, device); spatial_model = load_v21(required["spatial"], args.kit_root, args.checkpoint, device)
    ds, raw, base, delta, target = collect_diagnostic(raw_model, spatial_model, dev_paths, args, device)
    full, mean, fluct = decomposed_windows(ds, base, delta); out = args.out_dir / "diagnostic_spatial"; out.mkdir(parents=True, exist_ok=True)
    values = {"raw_control": raw, "spatial_full": full, "spatial_mean_only": mean, "spatial_fluctuation_only": fluct}
    results={}; trajectories={}
    for name, value in values.items():
        case = out / name; case.mkdir(exist_ok=True); results[name] = core.score_bundle(args.kit_root, value, target, 0.0, case); trajectories[name] = metric_rows(ds, value, target, args.kit_root); write_rows(case / "trajectory_metrics.csv", trajectories[name])
    pair_rows=[]
    for name in ("spatial_full", "spatial_mean_only", "spatial_fluctuation_only"):
        rows=paired(trajectories["raw_control"], trajectories[name], args.bootstrap_seed, args.bootstrap_draws)
        for row in rows: row["candidate"] = name
        pair_rows += rows
    write_rows(out / "paired_vs_raw_control.csv", pair_rows)
    replay={};
    for name in ("raw_control", "spatial_full"):
        replay[name], rows = tke_replay(ds, values[name], target, args.kit_root, results[name]["raw_errors"]["tke"]); write_rows(out / f"tke_aggregate_contribution_{name}.csv", rows)
    json_dump(out / "diagnostic.json", {"locked_spec_sha256": spec["spec_sha256"], "official_tke_replay": replay, "results": results,
        "trajectory_reconstruction": "V2.1 ordered concatenation of overlapping windows per trajectory; v9 official aggregate remains per-window mean", "base_component": "FE-02 trained CNO output before its residual head", "delta": "FE-02 residual-head correction to u,v"})
    lines=["# FE-02 SpatialPhysics no-training diagnostic", "", f"Locked FE-04 spec: `{spec['spec_sha256']}`.", "", "## Official TKE aggregation", "", "v9 computes a TKE relative-L2 independently for every scored 20-frame window, then takes their arithmetic mean. It is not a single global physical numerator/denominator ratio. `tke_aggregate_contribution_*.csv` records the equivalent sum-of-window-ratios contribution and leave-one-trajectory-out replay.", "", "## Raw metric replay", "", "| Variant | Rel-L2 | TKE | MVPE |", "|---|---:|---:|---:|"]
    for name, result in results.items():
        e=result["raw_errors"]; lines.append(f"| {name} | {e['rel_l2']:.6f} | {e['tke']:.6f} | {e['mvpe']:.6f} |")
    lines += ["", "Mean-only and fluctuation-only use the FE-02 CNO output plus the trajectory-level residual mean or zero-mean fluctuation. The trajectory construction is explicitly the pre-existing V2.1 ordered-window diagnostic convention; it is not represented as a hidden scorer stitch."]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def preflight(args: argparse.Namespace, spec: dict) -> None:
    if not torch.cuda.is_available(): raise SystemExit("GPU preflight requires CUDA")
    _, train_paths = core.read_manifest(args.manifest, "train"); device=torch.device("cuda"); config=spatial_config(); model=load_model(args.kit_root, args.checkpoint, config, fe.FeatureNormalizer(), device); optimizer=torch.optim.AdamW(model.parameters(), lr=args.lr)
    _, loader=core.loader(train_paths, args, shuffle=False); x,y,_,_=next(iter(loader)); x,y=x.to(device),y.to(device); torch.cuda.reset_peak_memory_stats()
    for timed in (False, True):
        optimizer.zero_grad(set_to_none=True)
        if timed: torch.cuda.synchronize(); started=time.perf_counter()
        loss=sum(N2[k]*v for k,v in core.loss_parts(model(x),y).items()); loss.backward(); optimizer.step()
        if timed: torch.cuda.synchronize(); elapsed=time.perf_counter()-started
    json_dump(args.out_dir / "preflight_b18.json", {"spec_sha256":spec["spec_sha256"], "batch_size":18, "gpu":torch.cuda.get_device_name(), "peak_gib":torch.cuda.max_memory_allocated()/2**30, "step_seconds":elapsed, "samples_per_second":18/elapsed})


def analyze(args: argparse.Namespace, spec: dict) -> None:
    candidate_path=args.out_dir/LABEL/"summary.json"; raw_path=args.v21_out/"FE-00R-ResidualRaw-Control/summary.json"; spatial_path=args.v21_out/"FE-02-SpatialPhysics/summary.json"
    if not candidate_path.exists(): raise FileNotFoundError(candidate_path)
    candidate=json.loads(candidate_path.read_text()); raw=json.loads(raw_path.read_text()); spatial=json.loads(spatial_path.read_text()); e=candidate["evaluation"]["raw_errors"]; r=raw["evaluation"]["raw_errors"]
    raw_csv=args.v21_out/"FE-00R-ResidualRaw-Control/eval_last_01271/trajectory_metrics.csv"; spatial_csv=args.v21_out/"FE-02-SpatialPhysics/eval_last_01269/trajectory_metrics.csv"; cand_csv=args.out_dir/LABEL/"eval_last_30m/trajectory_metrics.csv"
    read=lambda p:list(csv.DictReader(p.open()))
    pairs={"vs_raw_control":paired(read(raw_csv),read(cand_csv),args.bootstrap_seed,args.bootstrap_draws),"vs_spatial":paired(read(spatial_csv),read(cand_csv),args.bootstrap_seed,args.bootstrap_draws)}
    pass_gate=e["rel_l2"] <= r["rel_l2"]-.002 and e["mvpe"] <= r["mvpe"]+.002 and e["tke"] <= r["tke"]+.002
    result={"locked_spec_sha256":spec["spec_sha256"],"raw_control":r,"spatial":spatial["evaluation"]["raw_errors"],"fe04":e,"hard_gate_passed":pass_gate,"paired":pairs,"comparability":"FULL" if candidate["metadata"]["batch_size"]==18 else "LIMITED_COMPARABILITY"}
    json_dump(args.out_dir/"fe04_result.json",result)
    report=["# FE-04 RawSpatial8 result", "", f"Locked spec: `{spec['spec_sha256']}`.", "", "| Model | Rel-L2 | TKE | MVPE |", "|---|---:|---:|---:|"]
    for name,row in (("FE-00R Raw-Control",r),("FE-02 Spatial",spatial["evaluation"]["raw_errors"]),("FE-04 RawSpatial8",e)): report.append(f"| {name} | {row['rel_l2']:.6f} | {row['tke']:.6f} | {row['mvpe']:.6f} |")
    report += ["", f"Hard gate: **{'PASS' if pass_gate else 'NO-GO'}**. It requires Rel-L2 improvement ≥0.002 versus Raw-Control, with MVPE/TKE each no worse than +0.002. Second seed or longer training is not launched automatically.", "", "FE-04 holds the 8-channel/178-parameter head budget fixed but replaces FE-00R's five zero channels with spatial information; it does not eliminate the active-input-information effect."]
    (args.out_dir/"report_fe04.md").write_text("\n".join(report)+"\n")


def self_test() -> None:
    x=torch.zeros(2,20,32,64,3); x[...,0]=torch.linspace(0,1,64).view(1,1,1,64); x[...,1]=torch.linspace(0,1,32).view(1,1,32,1)
    value=raw_spatial8(x,spatial_config()); assert value.shape==(2,20,32,64,8) and torch.isfinite(value).all(); assert torch.equal(value[...,:3],x)
    assert raw_spatial8(torch.zeros_like(x),spatial_config())[...,7].eq(0).all(); print("FE-04 feature self-test: PASS")


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--kit-root",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--out-dir",type=Path,required=True); p.add_argument("--v21-out",type=Path,required=True)
    p.add_argument("--mode",choices=("lock","test","diagnose","preflight","train","analyze"),required=True); p.add_argument("--seed",type=int,default=20260901); p.add_argument("--batch-size",type=int,default=18); p.add_argument("--workers",type=int,default=2); p.add_argument("--max-windows",type=int); p.add_argument("--lr",type=float,default=1e-5); p.add_argument("--bootstrap-seed",type=int,default=20260901); p.add_argument("--bootstrap-draws",type=int,default=2000); p.add_argument("--checkpoint-interval",type=int,default=100); p.add_argument("--max-session-seconds",type=float,default=0.0)
    args=p.parse_args()
    if args.batch_size != 18: p.error("FE-04 frozen batch size is 18; do not lower it automatically")
    args.out_dir.mkdir(parents=True,exist_ok=True)
    if args.mode=="test": self_test(); return
    spec=get_locked(args,create=args.mode=="lock")
    if args.mode=="lock": print(spec["spec_sha256"]); return
    if args.mode=="diagnose": diagnose(args,spec)
    elif args.mode=="preflight": preflight(args,spec)
    elif args.mode=="train": train(args,spec)
    elif args.mode=="analyze": analyze(args,spec)


if __name__ == "__main__": main()
