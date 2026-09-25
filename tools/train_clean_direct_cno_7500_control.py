#!/usr/bin/env python3
"""Clean 7.5k matched control: Strong recipe with Direct-CNO output.

Scientific variable versus Exp1 Strong Backbone @7,500:
    MF mean/fluctuation output parameterization -> direct [u,v,p] output.

Frozen:
    Clean Train51 / Seen-Dev12
    Dense-All + P0-A
    official sim_real direct-CNO initialization
    N2 + vorticity objective
    seed=41, batch=8, AdamW, lr=1e-5
    7,500 optimizer updates

This runner is intentionally derived from train_clean_strong_backbone.py.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import realpde_loss_official_v9 as core
import realpde_sota_v2_integrated as sota
from dw01_by_horizon import aggregate_by_horizon, compute_window_horizon_metrics
from diagnose_tail_fluctuation_coherence import fluctuation_horizon_rows, mean_field_metrics

SEED = 41
FINAL_UPDATE = 7_500
MILESTONES = (0, 2_500, 5_000, 7_500)
EXPECTED_TRAIN = 51
EXPECTED_DEV = 12
EXPECTED_DENSE = 41_317
EXPECTED_DEV_WINDOWS = 491
BATCH_SIZE = 8
LR = 1e-5
MAX_PEAK_RESERVED_BYTES = 12 * 1024**3
SIM_PRETRAIN_SHA256 = "82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61"


def point_score(raw: dict[str, float]) -> float:
    def s(x: float) -> float:
        return 100.0 / (1.0 + 0.5 * max(float(x), 0.0))
    return float(np.mean([s(raw["rel_l2"]), s(raw["tke"]), s(raw["mvpe"])]))


def current_peak_reserved(device: torch.device) -> int:
    return int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0


def enforce_memory_cap(device: torch.device, where: str) -> int:
    peak = current_peak_reserved(device)
    if peak > MAX_PEAK_RESERVED_BYTES:
        raise RuntimeError(
            f"Direct-CNO control exceeded 12 GiB process CUDA peak reserved at {where}: "
            f"{peak / 1024**3:.3f} GiB"
        )
    return peak


def save_checkpoint(path: Path, model, optimizer, update: int, config, metadata: dict) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": int(update),
            "feature_set": "P0-A",
            "feature_config": vars(config),
            "output_parameterization": "direct_uvp",
            "loss_weights": sota.N2,
            "lambda_vort": sota.LAMBDA_VORT,
            "stage": "A",
            "stage_a_lr": LR,
            "metadata": metadata,
        },
        path,
    )


def aggregate_by_trajectory_horizon(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["trajectory"]), int(row["horizon"]))
        grouped.setdefault(key, []).append(row)
    fields = (
        "frame_rel_l2",
        "frame_rmse",
        "tke_contrib_rel_l2",
        "tke_contrib_ratio",
        "mvpe_probe_rel_l2",
    )
    out: list[dict[str, object]] = []
    for (trajectory, horizon), group in sorted(grouped.items()):
        out.append(
            {
                "trajectory": trajectory,
                "horizon": horizon,
                "windows": len(group),
                **{
                    field: float(np.mean([float(row[field]) for row in group]))
                    for field in fields
                },
            }
        )
    return out


@torch.inference_mode()
def evaluate_direct(
    model,
    builder,
    dev_paths: list[Path],
    *,
    kit_root: Path,
    device: torch.device,
    workers: int,
    eval_batch_size: int,
    out_dir: Path,
    update: int,
) -> tuple[dict[str, float], float]:
    out_dir.mkdir(parents=True)
    ds, loader = sota.dev_loader(
        dev_paths,
        argparse.Namespace(eval_batch_size=eval_batch_size, workers=workers),
    )
    if len(ds) != EXPECTED_DEV_WINDOWS:
        raise ValueError(f"expected {EXPECTED_DEV_WINDOWS} dev windows, got {len(ds)}")

    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    elapsed = 0.0
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        pred = sota.forward_direct(model, builder, x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - started
        preds.append(pred.cpu().numpy().astype(np.float32))
        targets.append(y.numpy().astype(np.float32))

    prediction = np.concatenate(preds, axis=0)
    target = np.concatenate(targets, axis=0)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("non-finite direct-CNO dev prediction")
    if float(np.abs(prediction[..., 2]).max()) != 0.0:
        raise FloatingPointError("pressure channel is not identically zero")

    scored = core.score_bundle(
        kit_root,
        prediction,
        target,
        elapsed / len(ds),
        out_dir,
    )
    raw = {k: float(scored["raw_errors"][k]) for k in ("rel_l2", "tke", "mvpe")}

    trajectory_rows, anatomy = core.trajectory_rows(ds, prediction, target, kit_root)
    sota.rows(out_dir / "by_trajectory.csv", trajectory_rows)

    names = [ref.path.name for ref in ds.refs]
    starts = [int(ref.start) for ref in ds.refs]
    window_rows = compute_window_horizon_metrics(prediction, target, names, starts)
    horizon_rows = aggregate_by_horizon(
        window_rows,
        experiment=f"direct_cno_{update}",
        trajectories=EXPECTED_DEV,
    )
    sota.rows(out_dir / "by_horizon.csv", horizon_rows)
    sota.rows(out_dir / "by_trajectory_horizon.csv", aggregate_by_trajectory_horizon(window_rows))

    fluct_rows = fluctuation_horizon_rows(prediction, target)
    sota.rows(out_dir / "fluctuation_by_horizon.csv", fluct_rows)
    mean_stats = mean_field_metrics(prediction, target)

    f = {int(row["horizon"]): row for row in horizon_rows}
    tail = {
        "f18_rel": float(f[18]["frame_rel_l2"]),
        "f19_rel": float(f[19]["frame_rel_l2"]),
        "f20_rel": float(f[20]["frame_rel_l2"]),
        "f18_f20_growth_pct": 100.0
        * (float(f[20]["frame_rel_l2"]) - float(f[18]["frame_rel_l2"]))
        / max(abs(float(f[18]["frame_rel_l2"])), 1e-30),
    }
    summary = {
        "update": update,
        "raw_errors": raw,
        "point_score": point_score(raw),
        "tail": tail,
        "trajectories": len(trajectory_rows),
        "windows": len(ds),
        "mean_t_neural_s": float(elapsed / len(ds)),
        "runtime_valid_for_comparison": False,
        "runtime_note": "Shared GPU execution is allowed; timing is not a scientific comparison metric.",
        "trajectory_anatomy": anatomy,
        "mean_field": mean_stats,
        "tail_fluctuation": {
            "f18": fluct_rows[17],
            "f19": fluct_rows[18],
            "f20": fluct_rows[19],
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return tail | raw, float(summary["point_score"])


def run(args: argparse.Namespace) -> None:
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)

    for path in (args.manifest, args.data_root, args.init_checkpoint, args.kit_root):
        if not path.exists():
            raise FileNotFoundError(path)

    if sota.sha256(args.init_checkpoint) != SIM_PRETRAIN_SHA256:
        raise RuntimeError("official sim_real checkpoint SHA mismatch")

    core.set_seed(SEED)
    train = sota.split_paths(args.manifest, "train", args.data_root)
    dev = sota.split_paths(args.manifest, "dev", args.data_root)
    if (len(train), len(dev)) != (EXPECTED_TRAIN, EXPECTED_DEV):
        raise ValueError(
            f"clean split must be {EXPECTED_TRAIN}/{EXPECTED_DEV}, got {len(train)}/{len(dev)}"
        )
    if set(p.name for p in train) & set(p.name for p in dev):
        raise ValueError("train/dev overlap")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")

    builder, feature_config = sota.build_features(train, device)
    payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    model = sota.build_direct(args.kit_root, builder, payload, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    loader_args = argparse.Namespace(seed=SEED, micro_batch=BATCH_SIZE, workers=args.workers)
    train_ds, sampler, train_loader = sota.dense_loader(train, loader_args)
    dev_ds, dev_loader = sota.dev_loader(
        dev,
        argparse.Namespace(eval_batch_size=args.eval_batch_size, workers=args.workers),
    )
    if len(train_ds) != EXPECTED_DENSE or len(dev_ds) != EXPECTED_DEV_WINDOWS:
        raise ValueError(
            f"window audit mismatch dense/dev={len(train_ds)}/{len(dev_ds)}"
        )

    metadata = {
        "status": "REVIEW_REQUIRED",
        "protocol": "CLEAN_BASELINE_FINAL_CAMPAIGN/DIRECT_CNO_7500_MATCHED_CONTROL",
        "scientific_variable": "MF-CNO output parameterization -> Direct-CNO output parameterization",
        "recipe": "Dense-All+P0-A+Direct-CNO+N2+Vorticity",
        "matched_reference": "Exp1 Strong Backbone MF-CNO @ update 7500",
        "seed": SEED,
        "train_trajectories": len(train),
        "dev_trajectories": len(dev),
        "dense_train_windows": len(train_ds),
        "dev_windows": len(dev_ds),
        "batch_size": BATCH_SIZE,
        "final_update": FINAL_UPDATE,
        "lr": LR,
        "loss_weights": sota.N2,
        "lambda_vort": sota.LAMBDA_VORT,
        "stage_b_used": False,
        "official_init_sha256": SIM_PRETRAIN_SHA256,
        "max_process_cuda_peak_reserved_bytes": MAX_PEAK_RESERVED_BYTES,
        "shared_gpu_allowed": True,
        "runtime_valid_for_comparison": False,
        "holdout_accessed": False,
        "locked_final_accessed": False,
        "full_data_accessed": False,
        "sps_used_for_selection": False,
        "runtime_used_for_selection": False,
        "codabench_accessed": False,
    }
    (args.out_dir / "run_config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Reproduce the historical Exp1 preflight side effect exactly:
    # one train-mode forward on the first two Seen-Dev windows before update 1.
    # CNO contains BatchNorm3d, so this detail is part of the matched protocol.
    _hist_ds, hist_loader = sota.dev_loader(
        dev,
        argparse.Namespace(eval_batch_size=8, workers=args.workers),
    )
    hist_x, _hist_y, _, _ = next(iter(hist_loader))
    hist_x = hist_x[:2].to(device)
    model.train()
    with torch.no_grad():
        hist_pred = sota.forward_direct(model, builder, hist_x)
    if not torch.isfinite(hist_pred).all():
        raise FloatingPointError("non-finite historical-parity preflight prediction")

    # Preserve that exact post-parity state. The batch-8 backward below is only
    # a memory/gradient smoke and must not alter BatchNorm running statistics
    # or any other model state used by formal update 1.
    formal_start_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    # Memory/gradient preflight uses the exact frozen batch=8. Because CNO has
    # BatchNorm3d, reducing training micro-batch would change science.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _preflight_ds, preflight_loader = sota.dev_loader(
        dev,
        argparse.Namespace(eval_batch_size=BATCH_SIZE, workers=args.workers),
    )
    probe_x, probe_y, _, _ = next(iter(preflight_loader))
    if probe_x.shape[0] < BATCH_SIZE:
        raise RuntimeError("preflight did not receive frozen batch=8")
    probe_x = probe_x[:BATCH_SIZE].to(device)
    probe_y = probe_y[:BATCH_SIZE].to(device)
    model.train()
    pred = sota.forward_direct(model, builder, probe_x)
    loss, parts = sota.integrated_loss(pred, probe_y, 1)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite preflight loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_max = max(
        float(parameter.grad.abs().max())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    optimizer.zero_grad(set_to_none=True)
    preflight_peak = enforce_memory_cap(device, "preflight")

    # Roll back the smoke-only BN/statistical mutation before formal training.
    model.load_state_dict(formal_start_state, strict=True)
    model.to(device)
    optimizer.zero_grad(set_to_none=True)

    preflight = {
        "passed": bool(torch.isfinite(pred).all() and grad_max > 0),
        "batch_size": BATCH_SIZE,
        "historical_parity_forward_windows": 2,
        "smoke_state_restored_before_training": True,
        "pressure_max_abs": float(pred[..., 2].abs().max()),
        "loss": float(loss.detach()),
        "loss_parts": {k: float(v.detach()) for k, v in parts.items()},
        "gradient_max_abs": grad_max,
        "dense_train_windows": len(train_ds),
        "dev_windows": len(dev_ds),
        "feature_count": len(builder.feature_names),
        "process_peak_gpu_memory_reserved_bytes": preflight_peak,
        "process_peak_gpu_memory_reserved_gib": preflight_peak / 1024**3,
        "memory_cap_gib": 12.0,
    }
    (args.out_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not preflight["passed"]:
        raise RuntimeError("preflight failed")
    if args.preflight_only:
        return

    # Fresh peak counter for the formal run.
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    checkpoints = args.out_dir / "checkpoints"
    checkpoints.mkdir()
    history: list[dict[str, float | int]] = []

    def eval_and_record(update: int) -> None:
        metrics, score = evaluate_direct(
            model,
            builder,
            dev,
            kit_root=args.kit_root,
            device=device,
            workers=args.workers,
            eval_batch_size=args.eval_batch_size,
            out_dir=args.out_dir / f"eval_{update:05d}",
            update=update,
        )
        row = {"update": update, "point_score": score, **metrics}
        history.append(row)
        sota.rows(args.out_dir / "aggregate_metrics.csv", history)

    eval_and_record(0)

    iterator = iter(train_loader)
    epoch = 0
    started = time.monotonic()
    for update in range(1, FINAL_UPDATE + 1):
        try:
            x, y, _, _ = next(iterator)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            iterator = iter(train_loader)
            x, y, _, _ = next(iterator)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        model.train()
        pred = sota.forward_direct(model, builder, x)
        loss, parts = sota.integrated_loss(pred, y, update)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss @ {update}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if update % args.log_every == 0:
            peak = enforce_memory_cap(device, f"update {update}")
            print(
                json.dumps(
                    {
                        "update": update,
                        "lr": LR,
                        "loss": float(loss.detach()),
                        "process_peak_reserved_gib": peak / 1024**3,
                    }
                ),
                flush=True,
            )

        if update in MILESTONES[1:]:
            eval_and_record(update)
            save_checkpoint(
                checkpoints / f"model_update_{update:05d}.pth",
                model,
                optimizer,
                update,
                feature_config,
                metadata,
            )

    final_peak = enforce_memory_cap(device, "final")
    runtime = {
        "training_wall_seconds": time.monotonic() - started,
        "runtime_valid_for_comparison": False,
        "shared_gpu_allowed": True,
        "process_peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "process_peak_gpu_memory_reserved_bytes": final_peak,
        "process_peak_gpu_memory_reserved_gib": final_peak / 1024**3,
        "memory_cap_gib": 12.0,
        "dense_epochs_completed": epoch + 1,
    }
    (args.out_dir / "runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "REVIEW_REQUIRED",
                "scientific_variable": metadata["scientific_variable"],
                "final_update": FINAL_UPDATE,
                "history": history,
                "holdout_accessed": False,
                "locked_final_accessed": False,
                "codabench_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "DONE").touch()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--init-checkpoint", type=Path, required=True)
    p.add_argument("--kit-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
