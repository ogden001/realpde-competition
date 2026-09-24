#!/usr/bin/env python3
"""Clean-split validation of the frozen SOTA-V2 strong-backbone recipe.

Only released real-PIV Train51 / Seen-Dev12 are used.  The model is initialized
from the official sim_real direct-CNO checkpoint; historical all82/full-data
strong-backbone weights are forbidden.  The strong recipe itself is frozen as
Dense-All + P0-A + MF-CNO + N2 + vorticity + Stage-B extra Rel.

Checkpoint selection is point-only (Rel-L2/TKE/MVPE subscores).  SPS, runtime,
AoA10 holdout, full-data and Codabench never participate in selection here.
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
from realpde_mf01 import MF01CNO

SEED = 41
FINAL_UPDATE = 35_000
STAGE_A_END = 30_000
MILESTONES = (0, 7_500, 15_000, 20_000, 25_000, 30_000, 31_000, 32_500, 35_000)
EXPECTED_TRAIN = 51
EXPECTED_DEV = 12
EXPECTED_DENSE = 41_317
EXPECTED_DEV_WINDOWS = 491
BATCH_SIZE = 8


def point_score(raw: dict[str, float]) -> float:
    def s(x: float) -> float:
        return 100.0 / (1.0 + 0.5 * max(float(x), 0.0))
    return float(np.mean([s(raw["rel_l2"]), s(raw["tke"]), s(raw["mvpe"])]))


def save(path: Path, model, optimizer, update: int, config, metadata, score: float) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": int(update),
        "feature_set": "P0-A",
        "feature_config": vars(config),
        "loss_weights": sota.N2,
        "lambda_vort": sota.LAMBDA_VORT,
        "extra_rel_stage_b": sota.EXTRA_REL,
        "selection_metric": "point_score",
        "selection_score": float(score),
        "metadata": metadata,
    }
    torch.save(payload, path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--init-checkpoint", type=Path, required=True)
    p.add_argument("--kit-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--require-cuda", action="store_true")
    args = p.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    for path in (args.manifest, args.data_root, args.init_checkpoint, args.kit_root):
        if not path.exists():
            raise FileNotFoundError(path)

    core.set_seed(SEED)
    train = sota.split_paths(args.manifest, "train", args.data_root)
    dev = sota.split_paths(args.manifest, "dev", args.data_root)
    if (len(train), len(dev)) != (EXPECTED_TRAIN, EXPECTED_DEV):
        raise ValueError(f"clean split must be {EXPECTED_TRAIN}/{EXPECTED_DEV}, got {len(train)}/{len(dev)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")

    builder, feature_config = sota.build_features(train, device)
    payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    model = MF01CNO(args.kit_root, len(builder.feature_names), device)
    sota.init_mf_from_direct(model, payload, len(builder.feature_names))

    loader_args = argparse.Namespace(seed=SEED, micro_batch=BATCH_SIZE, workers=args.workers)
    train_ds, sampler, train_loader = sota.dense_loader(train, loader_args)
    dev_ds, dev_loader = sota.dev_loader(dev, argparse.Namespace(eval_batch_size=8, workers=args.workers))
    if len(train_ds) != EXPECTED_DENSE or len(dev_ds) != EXPECTED_DEV_WINDOWS:
        raise ValueError(f"window audit mismatch dense/dev={len(train_ds)}/{len(dev_ds)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=sota.STAGE_A_LR)
    metadata = {
        "protocol": "CLEAN_BASELINE_FINAL_CAMPAIGN/STRONG_BACKBONE_CLEAN",
        "recipe": "Dense-All+P0-A+MF-CNO+N2+Vorticity+StageB-extra-Rel",
        "seed": SEED,
        "train_trajectories": len(train),
        "dev_trajectories": len(dev),
        "dense_train_windows": len(train_ds),
        "dev_windows": len(dev_ds),
        "batch_size": BATCH_SIZE,
        "stage_a_end": STAGE_A_END,
        "final_update": FINAL_UPDATE,
        "stage_a_lr": sota.STAGE_A_LR,
        "stage_b_lr": sota.STAGE_B_LR,
        "historical_all82_checkpoint_used": False,
        "holdout_accessed": False,
        "sps_used_for_selection": False,
        "runtime_used_for_selection": False,
        "codabench_accessed": False,
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    probe_x, _, _, _ = next(iter(dev_loader))
    probe_x = probe_x[:2].to(device)
    with torch.no_grad():
        mf_pred = sota.forward_mf(model, builder, probe_x)
        direct = sota.build_direct(args.kit_root, builder, payload, device)
        direct_pred = sota.forward_direct(direct, builder, probe_x)
    parity = float((mf_pred[..., :2] - direct_pred[..., :2]).abs().max())
    if parity > 1e-6:
        raise RuntimeError(f"Direct->MF parity failed: {parity}")
    (args.out_dir / "preflight.json").write_text(json.dumps({
        "direct_to_mf_uv_max_abs_diff": parity,
        "train_windows": len(train_ds),
        "dev_windows": len(dev_ds),
        "passed": True,
    }, indent=2, sort_keys=True) + "\n")
    del direct, direct_pred

    checkpoints = args.out_dir / "checkpoints"
    checkpoints.mkdir()
    history: list[dict[str, float | int | str]] = []

    def evaluate(update: int) -> tuple[dict, float]:
        ev = sota.evaluate(model, builder, dev, argparse.Namespace(
            kit_root=args.kit_root, eval_batch_size=8, workers=args.workers
        ), device, args.out_dir / f"eval_{update:05d}", update)
        raw = {k: float(ev["raw_errors"][k]) for k in ("rel_l2", "tke", "mvpe")}
        score = point_score(raw)
        row = {"update": update, "point_score": score, **raw}
        history.append(row)
        sota.rows(args.out_dir / "aggregate_metrics.csv", history)
        return ev, score

    _, best_score = evaluate(0)
    best_update = 0
    save(checkpoints / "model_best.pth", model, optimizer, 0, feature_config, metadata, best_score)

    iterator = iter(train_loader)
    epoch = 0
    current_lr = sota.STAGE_A_LR
    started = time.monotonic()
    for update in range(1, FINAL_UPDATE + 1):
        cfg = sota.stage_config(update)
        lr = float(cfg["lr"])
        if lr != current_lr:
            current_lr = lr
            for group in optimizer.param_groups:
                group["lr"] = current_lr
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
        pred = sota.forward_mf(model, builder, x)
        loss, _ = sota.integrated_loss(pred, y, update)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss @ {update}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if update in MILESTONES[1:]:
            _, score = evaluate(update)
            save(checkpoints / f"model_update_{update:05d}.pth", model, optimizer, update, feature_config, metadata, score)
            save(checkpoints / "model_latest.pth", model, optimizer, update, feature_config, metadata, score)
            if score > best_score:
                best_score, best_update = score, update
                save(checkpoints / "model_best.pth", model, optimizer, update, feature_config, metadata, score)

    save(checkpoints / "model_final.pth", model, optimizer, FINAL_UPDATE, feature_config, metadata, history[-1]["point_score"])
    (args.out_dir / "summary.json").write_text(json.dumps({
        "status": "REVIEW_REQUIRED",
        "best_update": best_update,
        "best_point_score": best_score,
        "history": history,
        "training_wall_seconds": time.monotonic() - started,
        "holdout_accessed": False,
        "codabench_accessed": False,
    }, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "DONE").touch()


if __name__ == "__main__":
    main()
