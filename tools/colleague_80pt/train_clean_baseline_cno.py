#!/usr/bin/env python3
"""Train the clean-baseline Stage-1 CNO with fixed dense-window sampling.

This preserves the colleague-80 CNO architecture and Stage-1 loss, while using
an explicit clean train/dev split and a reproducible stride-1 training pool.
Checkpoint selection is point-only: the mean of the official-v9 Rel-L2, TKE,
and MVPE subscores. SPS and runtime never participate in model selection.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from dw01_by_horizon import aggregate_by_horizon, compute_window_horizon_metrics, write_csv  # noqa: E402
from realpde_h5_feature_adapter_train import (  # noqa: E402
    H5WindowDataset,
    load_cno_checkpoint,
    measured_channels,
    mvpe_rel_l2_per_sample,
    paths_from_split_manifest,
    rel_l2_per_sample,
    score_error,
    tke_rel_l2_per_sample,
)
from residual_multi import build_cno  # noqa: E402
from train_cno_heldout_aoa import colleague_stage1_loss, wake_ramp_weights  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def point_score(metrics: dict[str, float]) -> float:
    return float(np.mean([
        score_error(float(metrics["rel_l2_raw"])),
        score_error(float(metrics["tke_raw"])),
        score_error(float(metrics["mvpe_raw"])),
    ]))


def _rel_l2_np(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    denom = np.linalg.norm(t, axis=1).clip(min=1e-8)
    return np.linalg.norm(p - t, axis=1) / denom


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    out_dir: Path,
    experiment: str,
) -> dict[str, float | int]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        pred = model(x)
        predictions.append(pred.cpu().numpy().astype(np.float32))
        targets.append(y.numpy().astype(np.float32))

    pred = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    pred[..., 2] = 0.0
    channels = measured_channels(target)
    rel = rel_l2_per_sample(pred, target, channels)
    tke = tke_rel_l2_per_sample(pred, target, channels)
    mvpe = mvpe_rel_l2_per_sample(pred, target)

    pred_uv = pred[..., :2]
    target_uv = target[..., :2]
    pred_mean = pred_uv.mean(axis=1, keepdims=True)
    target_mean = target_uv.mean(axis=1, keepdims=True)
    mean_rel = _rel_l2_np(pred_mean, target_mean)
    fluct_rel = _rel_l2_np(pred_uv - pred_mean, target_uv - target_mean)

    metrics: dict[str, float | int] = {
        "windows": int(pred.shape[0]),
        "trajectories": len(loader.dataset.paths),
        "rel_l2_raw": float(np.mean(rel)),
        "tke_raw": float(np.mean(tke)),
        "mvpe_raw": float(np.mean(mvpe)),
        "mean_field_rel_l2": float(np.mean(mean_rel)),
        "fluctuation_rel_l2": float(np.mean(fluct_rel)),
    }
    metrics["rel_l2_score"] = score_error(float(metrics["rel_l2_raw"]))
    metrics["tke_score"] = score_error(float(metrics["tke_raw"]))
    metrics["mvpe_score"] = score_error(float(metrics["mvpe_raw"]))
    metrics["point_score"] = point_score(metrics)  # type: ignore[arg-type]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    refs = loader.dataset.refs
    names = [ref.path.name for ref in refs]
    starts = [ref.start for ref in refs]
    horizon_rows = aggregate_by_horizon(
        compute_window_horizon_metrics(pred, target, names, starts),
        experiment=experiment,
        trajectories=len(set(names)),
    )
    write_csv(out_dir / "by_horizon.csv", horizon_rows, list(horizon_rows[0]))
    return metrics


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    iteration: int,
    metrics: dict[str, object],
    run_config: dict[str, object],
    train_log: list[dict[str, float]],
    eval_log: list[dict[str, object]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "iteration": int(iteration),
            "selection_metric": "point_score",
            "selection_score": float(metrics["point_score"]),
            "dev_metrics": metrics,
            "training_config": run_config,
            "train_log": train_log,
            "eval_log": eval_log,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--realpdebench-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=8723)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--preload-to-ram", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--benchmark-mode", action="store_true")
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    args = parser.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_paths, dev_paths = paths_from_split_manifest(args.real_root, args.split_manifest)
    train_dataset = H5WindowDataset(
        train_paths,
        in_steps=20,
        out_steps=20,
        stride=1,
        sub_sample=2,
        include_pressure=False,
        window_mode="fixed",
        preload_to_ram=args.preload_to_ram,
    )
    dev_dataset = H5WindowDataset(
        dev_paths,
        in_steps=20,
        out_steps=20,
        stride=20,
        sub_sample=2,
        include_pressure=False,
        window_mode="fixed",
        preload_to_ram=args.preload_to_ram,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    worker_kwargs: dict[str, object] = {}
    if args.workers > 0:
        worker_kwargs = {
            "persistent_workers": True,
            "prefetch_factor": args.prefetch_factor,
        }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        **worker_kwargs,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        **worker_kwargs,
    )

    model = build_cno(args.realpdebench_root, device)
    load_cno_checkpoint(model, args.init_checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.updates)
    )
    w, wk = wake_ramp_weights(device)

    run_config: dict[str, object] = {
        "protocol": "REALPDE_CLEAN_BASELINE_V1",
        "stage": "stage1_cno",
        "train_trajectories": len(train_paths),
        "dev_trajectories": len(dev_paths),
        "train_dev_overlap": False,
        "in_steps": 20,
        "out_steps": 20,
        "train_stride": 1,
        "eval_stride": 20,
        "sub_sample": 2,
        "sampling_policy": "all_legal_stride1_windows_global_shuffle_without_replacement_per_epoch",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "test_batch_size": args.test_batch_size,
        "candidate_train_windows": len(train_dataset),
        "dev_windows": len(dev_dataset),
        "preload_to_ram": bool(args.preload_to_ram),
        "train_ram_cache": train_dataset.cache_summary(),
        "dev_ram_cache": dev_dataset.cache_summary(),
        "workers": args.workers,
        "persistent_workers": bool(args.workers > 0),
        "prefetch_factor": args.prefetch_factor if args.workers > 0 else None,
        "benchmark_mode": bool(args.benchmark_mode),
        "updates": args.updates,
        "eval_interval": args.eval_interval,
        "lr": args.lr,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "selection_metric": "point_score_mean_v9_rel_tke_mvpe_subscores",
        "sps_used_for_selection": False,
        "runtime_used_for_selection": False,
        "loss": "colleague80_stage1_decomp_exact",
    }
    (args.out_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.benchmark_mode:
        if device.type != "cuda":
            raise RuntimeError("--benchmark-mode requires CUDA")
        if args.benchmark_warmup < 0 or args.benchmark_warmup >= args.updates:
            raise ValueError("--benchmark-warmup must be in [0, updates)")
        torch.cuda.reset_peak_memory_stats(device)
        iterator = iter(train_loader)
        epoch = 0
        measured_seconds = 0.0
        measured_updates = 0
        for step in range(1, args.updates + 1):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            try:
                x, y = next(iterator)
            except StopIteration:
                epoch += 1
                iterator = iter(train_loader)
                x, y = next(iterator)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss, _ = colleague_stage1_loss(pred, y, w=w, wk=wk)
            loss.backward()
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize(device)
            if step > args.benchmark_warmup:
                measured_seconds += time.perf_counter() - started
                measured_updates += 1
        total_memory = int(torch.cuda.get_device_properties(device).total_memory)
        runtime = {
            "benchmark_mode": True,
            "batch_size": int(args.batch_size),
            "updates": int(args.updates),
            "warmup_updates": int(args.benchmark_warmup),
            "measured_updates": int(measured_updates),
            "measured_seconds": float(measured_seconds),
            "samples_per_second": float(
                measured_updates * args.batch_size / max(measured_seconds, 1e-12)
            ),
            "peak_gpu_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
            "peak_gpu_memory_reserved": int(torch.cuda.max_memory_reserved(device)),
            "gpu_total_memory": total_memory,
            "peak_allocated_fraction": float(
                torch.cuda.max_memory_allocated(device) / total_memory
            ),
            "peak_reserved_fraction": float(
                torch.cuda.max_memory_reserved(device) / total_memory
            ),
            "train_ram_cache": train_dataset.cache_summary(),
            "dev_ram_cache": dev_dataset.cache_summary(),
            "workers": args.workers,
            "prefetch_factor": args.prefetch_factor if args.workers > 0 else None,
        }
        (args.out_dir / "runtime.json").write_text(
            json.dumps(runtime, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.out_dir / "DONE").touch()
        print(json.dumps(runtime, indent=2, sort_keys=True), flush=True)
        return

    train_log: list[dict[str, float]] = []
    eval_log: list[dict[str, object]] = []
    init_metrics = evaluate(
        model, dev_loader, device, out_dir=args.out_dir / "eval_step_00000", experiment="cno_step0"
    )
    eval_log.append({"iteration": 0, **init_metrics})
    best_score = float(init_metrics["point_score"])
    best_iter = 0
    save_checkpoint(
        args.out_dir / "model_init.pth",
        model,
        iteration=0,
        metrics=init_metrics,
        run_config=run_config,
        train_log=train_log,
        eval_log=eval_log,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    save_checkpoint(
        args.out_dir / "model_best.pth",
        model,
        iteration=0,
        metrics=init_metrics,
        run_config=run_config,
        train_log=train_log,
        eval_log=eval_log,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    iterator = iter(train_loader)
    epoch = 0
    latest_metrics: dict[str, object] = dict(init_metrics)
    for step in range(1, args.updates + 1):
        model.train()
        try:
            x, y = next(iterator)
        except StopIteration:
            epoch += 1
            iterator = iter(train_loader)
            x, y = next(iterator)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss, parts = colleague_stage1_loss(pred, y, w=w, wk=wk)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0 or step == args.updates:
            row = {
                "step": float(step),
                "epoch": float(epoch),
                "lr": float(optimizer.param_groups[0]["lr"]),
                **parts,
            }
            train_log.append(row)
            print("TRAIN " + json.dumps(row, sort_keys=True), flush=True)

        if step % args.eval_interval == 0 or step == args.updates:
            latest_metrics = evaluate(
                model,
                dev_loader,
                device,
                out_dir=args.out_dir / f"eval_step_{step:05d}",
                experiment=f"cno_step{step}",
            )
            eval_row = {"iteration": step, **latest_metrics}
            eval_log.append(eval_row)
            score = float(latest_metrics["point_score"])
            print("EVAL " + json.dumps(eval_row, sort_keys=True), flush=True)
            save_checkpoint(
                args.out_dir / "model_latest.pth",
                model,
                iteration=step,
                metrics=latest_metrics,
                run_config=run_config,
                train_log=train_log,
                eval_log=eval_log,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            if score > best_score:
                best_score = score
                best_iter = step
                save_checkpoint(
                    args.out_dir / "model_best.pth",
                    model,
                    iteration=step,
                    metrics=latest_metrics,
                    run_config=run_config,
                    train_log=train_log,
                    eval_log=eval_log,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

            (args.out_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "REVIEW_REQUIRED",
                        "best_iteration": best_iter,
                        "best_point_score": best_score,
                        "latest_iteration": step,
                        "latest_metrics": latest_metrics,
                    },
                    indent=2,
                    sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )

    save_checkpoint(
        args.out_dir / "model_final.pth",
        model,
        iteration=args.updates,
        metrics=latest_metrics,
        run_config=run_config,
        train_log=train_log,
        eval_log=eval_log,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    (args.out_dir / "sampling_audit.json").write_text(
        json.dumps(
            {
                "candidate_legal_windows": len(train_dataset),
                "samples_per_complete_epoch_after_drop_last": (
                    len(train_dataset) // args.batch_size
                ) * args.batch_size,
                "updates": args.updates,
                "samples_consumed": args.updates * args.batch_size,
                "sampling_policy": run_config["sampling_policy"],
                "seed": args.seed,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "REVIEW_REQUIRED", "best_iteration": best_iter, "best_point_score": best_score}, indent=2), flush=True)


if __name__ == "__main__":
    main()
