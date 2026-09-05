#!/usr/bin/env python3
"""Low-memory matched Direct-vs-MF long convergence for concurrent GPU use.

This runner keeps the historical effective training batch at 8, but slices each
CPU batch into GPU microbatches of 2 and accumulates the correctly weighted
loss before one AdamW step. The final partial effective batch is normalized by
its actual sample count. Evaluation also uses batch 2. Only one model is
resident on GPU at a time.

The scientific contract is inherited from realpde_mf_long_convergence:
Direct@3000 vs MF@3000, fixed 50/16, P0-A, N2, seed 20260901, AdamW 1e-5,
12,000 additional optimizer updates per arm, checkpoints at absolute
6000/9000/12000/15000. No locked-final, Codabench, SPS or full-data access.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import realpde_mf_long_convergence as base  # noqa: E402

SEED = base.SEED
START_ABSOLUTE_UPDATE = base.START_ABSOLUTE_UPDATE
ADDITIONAL_UPDATES = base.ADDITIONAL_UPDATES
MILESTONE_INTERVAL = base.MILESTONE_INTERVAL
EXPECTED_MANIFEST_SHA256 = base.EXPECTED_MANIFEST_SHA256
EXPECTED_DIRECT3000_SHA256 = base.EXPECTED_DIRECT3000_SHA256
EXPECTED_MF1500_SHA256 = base.EXPECTED_MF1500_SHA256
EXPECTED_SCORER_SHA256 = base.EXPECTED_SCORER_SHA256
N2_WEIGHTS = base.N2_WEIGHTS
EFFECTIVE_BATCH_SIZE = 8
MICRO_BATCH_SIZE = 2
MAX_GPU_MEMORY_GIB = 9.0

absolute_milestones = base.absolute_milestones
validate_mf3000_metadata = base.validate_mf3000_metadata
validate_direct3000_metadata = base.validate_direct3000_metadata
mf_improvement_pct = base.mf_improvement_pct
restore_optimizer_state = base.restore_optimizer_state
sha256 = base.sha256
save_json = base.save_json
set_seed = base.set_seed


def microbatch_plan(batch_size: int, micro_batch_size: int) -> list[tuple[int, int, float]]:
    if batch_size <= 0 or micro_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    plan = []
    for start in range(0, batch_size, micro_batch_size):
        end = min(start + micro_batch_size, batch_size)
        plan.append((start, end, (end - start) / batch_size))
    return plan


def eval_args(args: argparse.Namespace) -> argparse.Namespace:
    values = dict(vars(args))
    values["batch_size"] = args.micro_batch_size
    return argparse.Namespace(**values)


def assert_memory_cap(device: torch.device, cap_gib: float, stage: str) -> float:
    if device.type != "cuda":
        return 0.0
    peak = float(torch.cuda.max_memory_reserved(device) / (1024 ** 3))
    if peak > cap_gib:
        raise RuntimeError(f"GPU memory cap exceeded during {stage}: {peak:.3f} GiB > {cap_gib:.3f} GiB")
    return peak


@torch.no_grad()
def evaluate_lowmem(model, builder, paths, args, device, core, mf01, out: Path) -> dict:
    local_args = eval_args(args)
    out.mkdir(parents=True, exist_ok=True)
    ds, loader = core.loader(paths, local_args, shuffle=False)
    predictions, targets, elapsed = [], [], 0.0
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        prediction = base.forward(model, builder, x, mf01)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        predictions.append(prediction.cpu().numpy().astype("float32"))
        targets.append(y.numpy().astype("float32"))
    import numpy as np
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    peak = assert_memory_cap(device, args.max_gpu_memory_gib, f"eval:{out.name}")
    scored = core.score_bundle(args.kit_root, prediction, target, elapsed / len(ds), out)
    rows, _ = core.trajectory_rows(ds, prediction, target, args.kit_root)
    with (out / "trajectory_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return scored | {
        "windows": len(ds), "trajectories": len(rows), "trajectory_rows": rows,
        "peak_gpu_reserved_gib": peak,
    }


def effective_batch_step(model, builder, x_cpu, y_cpu, optimizer, args, device, core, mf01) -> dict:
    full_batch = int(x_cpu.shape[0])
    optimizer.zero_grad(set_to_none=True)
    weighted_loss = 0.0
    micro_count = 0
    for start, end, weight in microbatch_plan(full_batch, args.micro_batch_size):
        x = x_cpu[start:end].to(device, non_blocking=True)
        y = y_cpu[start:end].to(device, non_blocking=True)
        prediction = base.forward(model, builder, x, mf01)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("non-finite prediction during low-memory training")
        parts = core.loss_parts(prediction, y)
        loss = sum(N2_WEIGHTS[name] * parts[name] for name in N2_WEIGHTS)
        (loss * weight).backward()
        weighted_loss += float(loss.detach()) * weight
        micro_count += 1
        del x, y, prediction, parts, loss
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    if not torch.isfinite(torch.tensor(grad_norm)) or grad_norm <= 0:
        raise RuntimeError(f"invalid gradient norm: {grad_norm}")
    optimizer.step()
    return {
        "loss": weighted_loss,
        "effective_batch": full_batch,
        "microbatches": micro_count,
        "grad_norm_before_clip": grad_norm,
    }


def one_step_trainability_check(arm, model, builder, payload, train_paths, args, device, core, mf01) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    restore_optimizer_state(optimizer, payload)
    loader_args = argparse.Namespace(**vars(args))
    loader_args.batch_size = EFFECTIVE_BATCH_SIZE
    loader_args.workers = 0
    loader_args.max_windows = None
    _, loader = core.loader(train_paths, loader_args, shuffle=False)
    x_cpu, y_cpu, _, _ = next(iter(loader))
    first_parameter = next(model.parameters())
    before = first_parameter.detach().clone()
    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    step = effective_batch_step(model, builder, x_cpu, y_cpu, optimizer, args, device, core, mf01)
    delta = float((first_parameter.detach() - before).norm())
    peak = assert_memory_cap(device, args.max_gpu_memory_gib, f"preflight:{arm}:train")
    if delta <= 0:
        raise RuntimeError(f"{arm} first parameter did not update")
    return step | {"first_parameter_delta": delta, "peak_gpu_reserved_gib": peak,
                   "optimizer_state_entries": len(optimizer.state)}


def run_preflight(args: argparse.Namespace) -> None:
    if args.preflight_out is None:
        raise ValueError("--preflight-out is required with --preflight")
    if args.preflight_out.exists() and any(args.preflight_out.iterdir()):
        raise FileExistsError(args.preflight_out)
    args.preflight_out.mkdir(parents=True, exist_ok=True)
    core, mf01 = base.runtime_modules()
    set_seed(args.seed)
    train_paths, dev_paths, direct_payload, mf_payload = base.validate_contract(args, core, mf01)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    builder, feature_config = mf01.build_features(train_paths, device)
    evidence = {
        "passed": False,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "micro_batch_size": args.micro_batch_size,
        "max_gpu_memory_gib": args.max_gpu_memory_gib,
        "manifest_sha256": sha256(args.manifest),
        "direct3000_sha256": sha256(args.direct_checkpoint),
        "mf3000_sha256": sha256(args.mf_checkpoint),
        "scorer_sha256": sha256(args.kit_root / "scoring.py"),
        "feature_names": list(builder.feature_names),
        "feature_config": vars(feature_config),
        "train_trajectories": 50, "dev_trajectories": 16,
        "locked_final_accessed": False, "codabench": False, "device": str(device),
    }
    for arm, payload in (("direct", direct_payload), ("mf", mf_payload)):
        set_seed(args.seed)
        model = base.build_model(arm, args.kit_root, builder, payload, device, mf01)
        start_eval = evaluate_lowmem(model, builder, dev_paths, args, device, core, mf01,
                                     args.preflight_out / f"eval_{arm}_3000")
        base.assert_expected_start_metrics(arm, start_eval["raw_errors"])
        set_seed(args.seed)
        model_for_step = base.build_model(arm, args.kit_root, builder, payload, device, mf01)
        trainability = one_step_trainability_check(
            arm, model_for_step, builder, payload, train_paths, args, device, core, mf01
        )
        evidence[arm] = {
            "start_raw_errors": start_eval["raw_errors"],
            "official_v9_subscores": start_eval["official_v9_subscores"],
            "eval_peak_gpu_reserved_gib": start_eval["peak_gpu_reserved_gib"],
            "trainability": trainability,
        }
        del model, model_for_step
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
    evidence["passed"] = True
    save_json(args.preflight_out / "preflight.json", evidence)


def train_arm_lowmem(arm, payload, builder, train_paths, dev_paths, args, device, core, mf01, out_root: Path) -> dict:
    arm_out = out_root / arm
    arm_out.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    model = base.build_model(arm, args.kit_root, builder, payload, device, mf01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    restore_optimizer_state(optimizer, payload)

    start_eval = evaluate_lowmem(model, builder, dev_paths, args, device, core, mf01, arm_out / "eval_03000")
    base.assert_expected_start_metrics(arm, start_eval["raw_errors"])
    history = [{"absolute_update": START_ABSOLUTE_UPDATE, "arm": arm, **start_eval["raw_errors"],
                "peak_gpu_reserved_gib": start_eval["peak_gpu_reserved_gib"]}]
    evals = {START_ABSOLUTE_UPDATE: start_eval}

    loader_args = argparse.Namespace(**vars(args))
    loader_args.batch_size = EFFECTIVE_BATCH_SIZE
    train_ds, loader = core.loader(train_paths, loader_args, shuffle=True)
    iterator = iter(loader)
    milestones = set(absolute_milestones(START_ABSOLUTE_UPDATE, ADDITIONAL_UPDATES, MILESTONE_INTERVAL))
    started = time.monotonic()
    training_peak = 0.0
    for relative in range(1, ADDITIONAL_UPDATES + 1):
        try:
            x_cpu, y_cpu, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x_cpu, y_cpu, _, _ = next(iterator)
        model.train()
        if relative == 1 and device.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        step = effective_batch_step(model, builder, x_cpu, y_cpu, optimizer, args, device, core, mf01)
        if relative == 1:
            training_peak = assert_memory_cap(device, args.max_gpu_memory_gib, f"formal:{arm}:train")

        absolute = START_ABSOLUTE_UPDATE + relative
        if absolute in milestones:
            result = evaluate_lowmem(model, builder, dev_paths, args, device, core, mf01,
                                     arm_out / f"eval_{absolute:05d}")
            row = {
                "absolute_update": absolute, "arm": arm, **result["raw_errors"],
                "mean_t_neural_s": result["mean_t_neural_s"],
                "elapsed_seconds": time.monotonic() - started,
                "eval_peak_gpu_reserved_gib": result["peak_gpu_reserved_gib"],
                "training_peak_gpu_reserved_gib": training_peak,
            }
            history.append(row); evals[absolute] = result
            checkpoint_metadata = {
                "experiment_id": args.experiment_id, "arm": arm,
                "start_absolute_update": START_ABSOLUTE_UPDATE, "absolute_update": absolute,
                "seed": args.seed, "lr": args.lr,
                "effective_batch_size": EFFECTIVE_BATCH_SIZE,
                "micro_batch_size": args.micro_batch_size,
                "workers": args.workers,
                "manifest_sha256": sha256(args.manifest),
                "source_checkpoint_sha256": sha256(args.direct_checkpoint if arm == "direct" else args.mf_checkpoint),
                "scorer_sha256": sha256(args.kit_root / "scoring.py"),
                "loss_weights": N2_WEIGHTS, "locked_final_accessed": False, "codabench": False,
            }
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "iteration": absolute, "metadata": checkpoint_metadata},
                       arm_out / f"model_update_{absolute:05d}.pth")
            print(json.dumps(row, sort_keys=True), flush=True)

    with (arm_out / "update_curve.csv").open("w", newline="") as handle:
        fields = list(dict.fromkeys(key for row in history for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(history)
    save_json(arm_out / "summary.json", {
        "arm": arm, "train_windows": len(train_ds), "elapsed_seconds": time.monotonic() - started,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE, "micro_batch_size": args.micro_batch_size,
        "training_peak_gpu_reserved_gib": training_peak, "history": history,
    })
    final_rows = evals[START_ABSOLUTE_UPDATE + ADDITIONAL_UPDATES]["trajectory_rows"]
    result = {"history": history, "evals": evals, "final_rows": final_rows}
    del model, optimizer
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    return result


def run(args: argparse.Namespace) -> None:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(args.out_dir)
    if args.preflight_evidence is None:
        raise ValueError("--preflight-evidence is required for formal low-memory training")
    preflight = json.loads(args.preflight_evidence.read_text(encoding="utf-8"))
    if not preflight.get("passed"):
        raise RuntimeError("refusing low-memory long convergence: preflight did not pass")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    core, mf01 = base.runtime_modules()
    set_seed(args.seed)
    train_paths, dev_paths, direct_payload, mf_payload = base.validate_contract(args, core, mf01)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    builder, feature_config = mf01.build_features(train_paths, device)
    metadata = {
        "experiment_id": args.experiment_id,
        "purpose": "matched Direct-vs-MF convergence 3000->15000 with GPU microbatching",
        "start_absolute_update": START_ABSOLUTE_UPDATE,
        "additional_updates_per_arm": ADDITIONAL_UPDATES,
        "final_absolute_update": START_ABSOLUTE_UPDATE + ADDITIONAL_UPDATES,
        "absolute_milestones": absolute_milestones(START_ABSOLUTE_UPDATE, ADDITIONAL_UPDATES, MILESTONE_INTERVAL),
        "arm_order": ["direct", "mf"],
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "micro_batch_size": args.micro_batch_size,
        "max_gpu_memory_gib": args.max_gpu_memory_gib,
        "gradient_accumulation": "microbatch losses weighted by micro_size/effective_batch_actual before one AdamW step",
        "seed": args.seed, "lr": args.lr, "workers": args.workers,
        "loss_weights": N2_WEIGHTS,
        "manifest_sha256": sha256(args.manifest),
        "direct3000_sha256": sha256(args.direct_checkpoint),
        "mf3000_sha256": sha256(args.mf_checkpoint),
        "scorer_sha256": sha256(args.kit_root / "scoring.py"),
        "feature_names": list(builder.feature_names), "feature_config": vars(feature_config),
        "train_trajectories": len(train_paths), "dev_trajectories": len(dev_paths),
        "preflight_evidence": str(args.preflight_evidence),
        "locked_final_accessed": False, "codabench": False, "device": str(device),
        "runner_sha256": sha256(Path(__file__)),
    }
    save_json(args.out_dir / "run_metadata.json", metadata)

    direct_result = train_arm_lowmem("direct", direct_payload, builder, train_paths, dev_paths,
                                     args, device, core, mf01, args.out_dir)
    mf_result = train_arm_lowmem("mf", mf_payload, builder, train_paths, dev_paths,
                                 args, device, core, mf01, args.out_dir)

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
        writer.writeheader(); writer.writerows(paired_rows)

    trajectory_rows, wins = base.trajectory_wins(direct_result["final_rows"], mf_result["final_rows"])
    with (args.out_dir / "final_trajectory_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectory_rows[0]))
        writer.writeheader(); writer.writerows(trajectory_rows)
    status = base.campaign_status(paired_rows)
    status["final_trajectory_wins"] = wins
    status["execution_mode"] = "low_memory_microbatch2_effective8"
    save_json(args.out_dir / "gate_result.json", status)
    save_json(args.out_dir / "summary.json", {"metadata": metadata, "paired_convergence": paired_rows, "gate": status})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", default="T1-ID-MF-LONG-CONVERGENCE-LOWMEM-S20260906")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--direct-checkpoint", type=Path, required=True)
    parser.add_argument("--mf-checkpoint", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=EFFECTIVE_BATCH_SIZE)
    parser.add_argument("--micro-batch-size", type=int, default=MICRO_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--start-update", type=int, default=START_ABSOLUTE_UPDATE)
    parser.add_argument("--additional-updates", type=int, default=ADDITIONAL_UPDATES)
    parser.add_argument("--milestone-interval", type=int, default=MILESTONE_INTERVAL)
    parser.add_argument("--max-gpu-memory-gib", type=float, default=MAX_GPU_MEMORY_GIB)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-out", type=Path)
    parser.add_argument("--preflight-evidence", type=Path)
    args = parser.parse_args()
    if args.seed != SEED or args.lr != 1e-5 or args.batch_size != EFFECTIVE_BATCH_SIZE or args.workers != 2:
        raise ValueError("seed/lr/effective batch/workers are frozen at 20260901/1e-5/8/2")
    if args.micro_batch_size != MICRO_BATCH_SIZE:
        raise ValueError("low-memory campaign micro batch is frozen at 2")
    if args.max_gpu_memory_gib > MAX_GPU_MEMORY_GIB:
        raise ValueError("GPU memory cap cannot exceed 9.0 GiB")
    absolute_milestones(args.start_update, args.additional_updates, args.milestone_interval)
    if args.preflight:
        run_preflight(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
