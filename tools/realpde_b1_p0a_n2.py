#!/usr/bin/env python3
"""Train submission-compatible B1: P0-A features plus calibrated N2 loss.

P0-A is built entirely from ``input_array`` history and therefore works with
the Track 1 v9 ``predict(input_array, metadata={})`` contract.  This runner
uses the supplied v9 scorer for dev and locked-final audits, but never reads
locked-final targets while selecting a checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

import realpde_loss_official_v9 as core
from realpde_p0_data import H5WindowDataset, RandomPhaseWindowSampler, read_grid
from realpde_p0_features import P0FeatureBuilder, P0FeatureConfig


N2_WEIGHTS = {
    "fluct": 0.0,
    "mean": 0.0,
    "mse": 1.0,
    "mvpe": 0.009757,
    "rel": 0.027514,
    "tke": 0.05,
}


def parse_eval_updates(value: str) -> tuple[int, ...]:
    updates = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not updates or any(update < 1 for update in updates) or tuple(sorted(set(updates))) != updates:
        raise ValueError("eval updates must be a non-empty strictly increasing list of positive integers")
    return updates


def summarize_window_audit(records: list[dict[str, object]], *, seed: int) -> dict[str, object]:
    if not records:
        raise ValueError("window audit must contain at least one record")
    epochs = sorted({int(record["epoch"]) for record in records})
    trajectories = sorted({str(record["trajectory"]) for record in records})
    windows_per_epoch = {
        str(epoch): sum(int(record["window_count"]) for record in records if int(record["epoch"]) == epoch)
        for epoch in epochs
    }
    phase_counts = Counter(int(record["phase"]) for record in records)
    counts = [int(record["window_count"]) for record in records]
    invalid = sum(int(record.get("invalid_window_count", 0)) for record in records)
    non_stride20 = sum(
        int(record.get("min_stride") != 20 or record.get("max_stride") != 20)
        for record in records
    )
    return {
        "seed": int(seed),
        "epoch_count": len(epochs),
        "epochs": epochs,
        "trajectory_count": len(trajectories),
        "windows_per_epoch": windows_per_epoch,
        "phase_counts": {str(phase): int(phase_counts.get(phase, 0)) for phase in range(20)},
        "min_window_count_per_trajectory": min(counts),
        "max_window_count_per_trajectory": max(counts),
        "invalid_window_count": invalid,
        "non_stride20_count": non_stride20,
    }


def early_screen_gate(baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, object]:
    degradation = {
        key: (float(candidate[key]) / max(float(baseline[key]), 1e-12) - 1.0) * 100.0
        for key in ("rel_l2", "tke", "mvpe")
    }
    stop = (degradation["rel_l2"] > 3.0 and degradation["mvpe"] > 3.0) or sum(value > 3.0 for value in degradation.values()) >= 3
    return {"status": "STOP_EARLY" if stop else "CONTINUE", "degradation_percent": degradation}


def matched_phase_severe_early_gate(baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, object]:
    """Apply the registered RW-MB @1000 severe-negative screen."""
    degradation = {
        key: (float(candidate[key]) / max(float(baseline[key]), 1e-12) - 1.0) * 100.0
        for key in ("rel_l2", "tke", "mvpe")
    }
    epsilon = 1e-9
    stop = ((degradation["rel_l2"] > 10.0 + epsilon and degradation["mvpe"] > 10.0 + epsilon)
            or all(value > 10.0 + epsilon for value in degradation.values())
            or degradation["tke"] > 20.0 + epsilon)
    return {"status": "STOP_EARLY" if stop else "CONTINUE", "degradation_percent": degradation}


def make_train_loader(paths: list[Path], args: argparse.Namespace):
    if args.train_window_mode == "fixed":
        dataset, loader = core.loader(paths, args, shuffle=True)
        return dataset, loader, None
    dataset = H5WindowDataset(paths, max_windows_per_trajectory=args.max_windows, window_mode="random_phase")
    sampler = RandomPhaseWindowSampler(dataset, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                        pin_memory=True, persistent_workers=False)
    return dataset, loader, sampler


def forced_random_phase_overrides(paths: list[Path], forced_phase: int | None) -> dict[str, int] | None:
    """Return a fixed phase map only for the registered sampler-path control."""
    if forced_phase is None:
        return None
    return {path.name: int(forced_phase) for path in paths}


def adapt_input_weight(model: torch.nn.Module, checkpoint: dict, in_channels: int) -> None:
    """Expand the input lift while preserving the 3-channel model at step zero."""
    state = checkpoint.get("model_state_dict", checkpoint)
    key = "lift.inter_CNOBlock.convolution.weight"
    if key not in state:
        raise KeyError(f"checkpoint lacks {key}")
    old = state[key]
    expanded = model.state_dict()[key].clone()
    if expanded.shape[1] != in_channels:
        raise ValueError(f"unexpected input weights: checkpoint={tuple(old.shape)}, model={tuple(expanded.shape)}")
    if old.shape[1] == in_channels:
        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        return
    if old.shape[1] != 3:
        raise ValueError(f"checkpoint has {old.shape[1]} inputs; expected 3 or {in_channels}")
    expanded.zero_()
    expanded[:, :3].copy_(old)
    state = dict(state)
    state[key] = expanded
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def load_model(kit_root: Path, checkpoint: Path, builder: P0FeatureBuilder, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(kit_root))
    from rpde_baselines.model.cno import CNO3d

    model = CNO3d(in_dim=len(builder.feature_names), out_dim=3, out_dim_mult=1, in_size=64, N_layers=3).to(device)
    adapt_input_weight(model, torch.load(checkpoint, map_location="cpu"), len(builder.feature_names))
    return model


def forward(model: torch.nn.Module, builder: P0FeatureBuilder, x: Tensor) -> Tensor:
    features = builder(x)
    return model(features.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)


@torch.no_grad()
def evaluate(model: torch.nn.Module, builder: P0FeatureBuilder, paths: list[Path], args: argparse.Namespace,
             device: torch.device, out: Path) -> dict:
    """Score model outputs with v9; timing includes P0-A construction."""
    out.mkdir(parents=True, exist_ok=True)
    ds, data = core.loader(paths, args, shuffle=False)
    model.eval()
    predictions, targets, elapsed = [], [], 0.0
    for x, y, _, _ in data:
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
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    result = core.score_bundle(args.kit_root, prediction, target, elapsed / len(ds), out)
    rows, anatomy = core.trajectory_rows(ds, prediction, target, args.kit_root)
    result.update({"windows": len(ds), "trajectories": len(rows), "trajectory_anatomy": anatomy})
    import csv
    with (out / "trajectory_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def save_checkpoint(path: Path, model: torch.nn.Module, *, iteration: int, config: P0FeatureConfig,
                    manifest_sha256: str) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "iteration": iteration,
            "feature_set": "P0-A",
            "feature_names": list(P0FeatureBuilder(config).feature_names),
            "feature_config": vars(config),
            "loss_weights": N2_WEIGHTS,
            "manifest_sha256": manifest_sha256,
        },
        path,
    )


def run(args: argparse.Namespace) -> None:
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if not (args.kit_root / "scoring.py").is_file():
        raise FileNotFoundError(args.kit_root / "scoring.py")
    core.set_seed(args.seed)
    eval_updates = parse_eval_updates(args.eval_updates) if args.eval_updates else None
    args.out_dir.mkdir(parents=True)
    _, train_paths = core.read_manifest(args.manifest, "train")
    _, dev_paths = core.read_manifest(args.manifest, "dev")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_grid, y_grid = read_grid(train_paths[0], sub_sample=2)
    config = P0FeatureConfig(
        include_p0_a=True,
        include_p0_b=False,
        dx=float(x_grid[0, 1] - x_grid[0, 0]),
        dy=float(y_grid[1, 0] - y_grid[0, 0]),
    )
    builder = P0FeatureBuilder(config).to(device)
    manifest_sha256 = core.sha256(args.manifest)
    metadata = {
        "run": "B1_P0A_N2",
        "submission_compatibility": "P0-A only; no metadata required by predict(input_array, metadata={})",
        "feature_names": list(builder.feature_names),
        "feature_config": vars(config),
        "loss_weights": N2_WEIGHTS,
        "seed": args.seed,
        "updates": args.updates,
        "max_train_seconds": args.max_train_seconds,
        "eval_interval": args.eval_interval,
        "eval_updates": eval_updates,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "train_window_mode": args.train_window_mode,
        "random_phase_forced_phase": args.random_phase_forced_phase,
        "execution_commit": args.execution_commit,
        "early_gate_mode": args.early_gate_mode,
        "lr": args.lr,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha256,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": core.sha256(args.checkpoint),
        "kit_root": str(args.kit_root.resolve()),
        "kit_scorer_sha256": core.sha256(args.kit_root / "scoring.py"),
        "device": str(device),
        "start_time": time.time(),
    }
    core.json_dump(args.out_dir / "run_metadata.json", metadata)
    model = load_model(args.kit_root, args.checkpoint, builder, device)
    baseline = evaluate(model, builder, dev_paths, args, device, args.out_dir / "eval_baseline")
    train_ds, train_loader, train_sampler = make_train_loader(train_paths, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = [{"iteration": 0, **baseline["raw_errors"]}]
    save_checkpoint(args.out_dir / "model_latest.pth", model, iteration=0, config=config, manifest_sha256=manifest_sha256)
    save_checkpoint(args.out_dir / "model_update_00000.pth", model, iteration=0, config=config, manifest_sha256=manifest_sha256)

    audit_records: list[dict[str, object]] = []
    audit_handle = None
    if train_sampler is not None:
        audit_handle = (args.out_dir / "window_audit.jsonl").open("w", encoding="utf-8")
        phase_overrides = forced_random_phase_overrides(train_paths, args.random_phase_forced_phase)

        def prepare_random_epoch(epoch: int, *, announce: bool = False) -> None:
            train_sampler.set_epoch(epoch, phases=phase_overrides)
            records = train_sampler.audit_records()
            audit_records.extend(records)
            for record in records:
                audit_handle.write(json.dumps(record, sort_keys=True) + "\n")
            audit_handle.flush()
            if announce:
                for record in records[:3]:
                    starts = list(range(int(record["first_start"]), int(record["last_start"]) + 1, 20))
                    print(f"traj {record['trajectory']} phase={record['phase']} starts={starts}", flush=True)

        prepare_random_epoch(0, announce=True)

    interrupted = False

    def request_stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True
        print("STOP_REQUESTED: will save checkpoint after current update", flush=True)

    previous_int = signal.signal(signal.SIGINT, request_stop)
    previous_term = signal.signal(signal.SIGTERM, request_stop)
    train_started = time.monotonic()
    iterator = iter(train_loader)
    current_epoch = 0
    latest_parts: dict[str, float] = {}
    stop_reason = "max_updates"
    actual_updates = 0
    early_gate_result: dict[str, object] | None = None
    try:
        for step in range(1, args.updates + 1):
            if interrupted:
                stop_reason = "signal"
                break
            if args.max_train_seconds is not None and time.monotonic() - train_started >= args.max_train_seconds:
                stop_reason = "max_train_seconds"
                break
            try:
                x, y, _, _ = next(iterator)
            except StopIteration:
                current_epoch += 1
                if train_sampler is not None:
                    prepare_random_epoch(current_epoch)
                iterator = iter(train_loader)
                x, y, _, _ = next(iterator)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            model.train()
            parts = core.loss_parts(forward(model, builder, x), y)
            loss = sum(N2_WEIGHTS[name] * value for name, value in parts.items())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            actual_updates = step
            latest_parts = {key: float(value.detach().cpu()) for key, value in parts.items()} | {"total": float(loss.detach().cpu())}
            should_eval = (step in eval_updates) if eval_updates is not None else step % args.eval_interval == 0
            if should_eval or step == args.updates:
                dev = evaluate(model, builder, dev_paths, args, device, args.out_dir / f"eval_{step:05d}")
                raw = dev["raw_errors"]
                row = {"iteration": step, **raw, "official_v9_subscores": dev["official_v9_subscores"],
                       "train_seconds": time.monotonic() - train_started}
                history.append(row)
                save_checkpoint(args.out_dir / "model_latest.pth", model, iteration=step,
                                config=config, manifest_sha256=manifest_sha256)
                save_checkpoint(args.out_dir / f"model_update_{step:05d}.pth", model, iteration=step,
                                config=config, manifest_sha256=manifest_sha256)
                print(json.dumps({"B1_DEV": row}, sort_keys=True), flush=True)
                if step == args.early_gate_update and args.early_gate_summary:
                    reference = json.loads(args.early_gate_summary.read_text(encoding="utf-8"))
                    reference_row = next(item for item in reference["history"] if int(item["iteration"]) == step)
                    early_gate_result = (
                        matched_phase_severe_early_gate(reference_row, raw)
                        if args.early_gate_mode == "matched_phase_severe"
                        else early_screen_gate(reference_row, raw)
                    )
                    print(json.dumps({"RW_EARLY_GATE": early_gate_result}, sort_keys=True), flush=True)
                    if early_gate_result["status"] == "STOP_EARLY":
                        stop_reason = "STOP_EARLY"
                        break
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)
        if audit_handle is not None:
            audit_handle.close()

    if not history or history[-1]["iteration"] != actual_updates:
        dev = evaluate(model, builder, dev_paths, args, device, args.out_dir / f"eval_{actual_updates:05d}")
        raw = dev["raw_errors"]
        row = {"iteration": actual_updates, **raw, "official_v9_subscores": dev["official_v9_subscores"],
               "train_seconds": time.monotonic() - train_started}
        history.append(row)
        print(json.dumps({"B1_DEV": row}, sort_keys=True), flush=True)
    save_checkpoint(args.out_dir / "model_latest.pth", model, iteration=actual_updates,
                    config=config, manifest_sha256=manifest_sha256)
    save_checkpoint(args.out_dir / f"model_update_{actual_updates:05d}.pth", model, iteration=actual_updates,
                    config=config, manifest_sha256=manifest_sha256)
    audit_summary = None
    if audit_records:
        audit_summary = summarize_window_audit(audit_records, seed=args.seed)
        core.json_dump(args.out_dir / "window_audit_summary.json", audit_summary)
    summary = {
        "metadata": metadata | {"end_time": time.time(), "actual_updates": actual_updates,
                                 "train_seconds": time.monotonic() - train_started, "stop_reason": stop_reason,
                                 "train_windows": len(train_sampler) if train_sampler is not None else len(train_ds),
                                 "train_dataset_refs": len(train_ds), "last_train_loss_parts": latest_parts,
                                 "early_gate": early_gate_result},
        "baseline": baseline,
        "history": history,
        "window_audit_summary": audit_summary,
    }
    core.json_dump(args.out_dir / "summary.json", summary)
    print(json.dumps({"B1_DONE": {"out_dir": str(args.out_dir), "stop_reason": stop_reason,
                                   "actual_updates": actual_updates}}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--updates", type=int, default=20000)
    parser.add_argument("--max-train-seconds", type=float, default=7200.0)
    parser.add_argument("--eval-interval", type=int, default=820)
    parser.add_argument("--eval-updates", type=str, default=None,
                        help="Comma-separated absolute updates to evaluate, e.g. 1000,2000,3000,5000,7500")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train-window-mode", choices=("fixed", "random_phase"), default="fixed")
    parser.add_argument("--random-phase-forced-phase", type=int, default=None,
                        help="Optional fixed phase for the random-phase sampler-path control")
    parser.add_argument("--early-gate-summary", type=Path, default=None,
                        help="RW-00 summary used for the RW-01 @3000 early screen")
    parser.add_argument("--early-gate-update", type=int, default=3000)
    parser.add_argument("--early-gate-mode", choices=("legacy", "matched_phase_severe"), default="legacy")
    parser.add_argument("--execution-commit", default=None,
                        help="Frozen source commit recorded in run metadata")
    args = parser.parse_args()
    if args.updates < 1 or args.eval_interval < 1 or args.max_train_seconds <= 0 or args.early_gate_update < 1:
        parser.error("updates, eval-interval, max-train-seconds, and early-gate-update must be positive")
    if args.random_phase_forced_phase is not None and not 0 <= args.random_phase_forced_phase < 20:
        parser.error("random-phase-forced-phase must be in [0, 20)")
    if args.random_phase_forced_phase is not None and args.train_window_mode != "random_phase":
        parser.error("random-phase-forced-phase requires train-window-mode=random_phase")
    run(args)


if __name__ == "__main__":
    main()
