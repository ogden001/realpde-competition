#!/usr/bin/env python3
"""Train the colleague Stage-1 CNO on an explicit split, optionally with heldout-AoA bridge augmentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from aoa_meanfield_augmentation import AoAMeanFieldShiftDataset  # noqa: E402
from realpde_h5_feature_adapter_train import (  # noqa: E402
    H5WindowDataset,
    load_cno_checkpoint,
    paths_from_split_manifest,
)
from residual_multi import build_cno  # noqa: E402


def wake_ramp_weights(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    h, w, t = 32, 64, 20
    wake = np.zeros((h, w), np.float32)
    wake[6:26, 10:45] = 1.0
    ramp = 1.0 + np.linspace(0.0, 1.0, t, dtype=np.float32)
    w_np = ramp[None, :, None, None] * (1.0 + wake[None, None, :, :])
    w_np /= w_np.mean()
    w = torch.from_numpy(w_np).to(device).unsqueeze(-1)
    wk = torch.from_numpy(1.0 + wake).to(device)
    return w, wk


def colleague_stage1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    w: torch.Tensor,
    wk: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    p = pred[..., :2]
    t = target[..., :2]
    pm = p.mean(1, keepdim=True)
    tm = t.mean(1, keepdim=True)
    pf = p - pm
    tf = t - tm
    mean_loss = (w * (pm - tm).square()).mean()
    fluct_loss = (w * (pf - tf).square()).mean()
    kp = (0.5 * (pf**2).mean(1)).sum(-1)
    kt = (0.5 * (tf**2).mean(1)).sum(-1)
    tke = (wk * (kp - kt).square()).mean()
    pzero = (pred[..., 2] ** 2).mean()
    loss = mean_loss + fluct_loss + 0.05 * tke + 0.01 * pzero
    return loss, {
        "loss": float(loss.detach().cpu()),
        "mean_loss": float(mean_loss.detach().cpu()),
        "fluct_loss": float(fluct_loss.detach().cpu()),
        "tke_loss": float(tke.detach().cpu()),
        "pzero": float(pzero.detach().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--realpdebench-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=8723)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--aoa-bridge", action="store_true")
    parser.add_argument("--bridge-low", type=float, default=5.0)
    parser.add_argument("--bridge-high", type=float, default=15.0)
    parser.add_argument("--aug-prob", type=float, default=0.5)
    parser.add_argument("--lambda-min", type=float, default=0.35)
    parser.add_argument("--lambda-max", type=float, default=0.65)
    parser.add_argument("--max-gap-deg", type=float, default=10.1)
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(args.out)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_paths, dev_paths = paths_from_split_manifest(args.real_root, args.split_manifest)
    base_dataset = H5WindowDataset(
        train_paths,
        in_steps=20,
        out_steps=20,
        stride=1,
        sub_sample=2,
        include_pressure=False,
        window_mode="fixed",
    )
    aoa_dataset = None
    train_dataset = base_dataset
    if args.aoa_bridge:
        aoa_dataset = AoAMeanFieldShiftDataset(
            base_dataset,
            probability=args.aug_prob,
            lambda_min=args.lambda_min,
            lambda_max=args.lambda_max,
            seed=args.seed,
            max_gap_deg=args.max_gap_deg,
            min_eligible_fraction=0.0,
            bridge_pair=(args.bridge_low, args.bridge_high),
        )
        train_dataset = aoa_dataset
        audit_path = args.out.with_suffix(".aoa_audit.json")
        audit_path.write_text(
            json.dumps(aoa_dataset.audit(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    model = build_cno(args.realpdebench_root, device)
    load_cno_checkpoint(model, args.init_checkpoint, device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.updates),
    )
    w, wk = wake_ramp_weights(device)

    train_log: list[dict[str, float]] = []
    iterator = iter(loader)
    epoch = 0
    for step in range(1, args.updates + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            if aoa_dataset is not None:
                aoa_dataset.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)

        if len(batch) == 3:
            x, y, meta = batch
        else:
            x, y = batch
            meta = None
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss, parts = colleague_stage1_loss(pred, y, w=w, wk=wk)
        if meta is not None:
            parts["aug_fraction"] = float(meta["applied"].float().mean().item())
            parts["effective_shift_abs_deg"] = float(
                meta["effective_shift_deg"].float().abs().mean().item()
            )
        else:
            parts["aug_fraction"] = 0.0
            parts["effective_shift_abs_deg"] = 0.0
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0 or step == args.updates:
            row = {
                "step": step,
                "lr": float(optimizer.param_groups[0]["lr"]),
                **parts,
            }
            train_log.append(row)
            print("TRAIN " + json.dumps(row, sort_keys=True), flush=True)

    payload = {
        "state_dict": model.state_dict(),
        "training_config": {
            "train_trajectories": len(train_paths),
            "inner_dev_trajectories_not_used_for_training": len(dev_paths),
            "updates": args.updates,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "aoa_bridge": bool(args.aoa_bridge),
            "bridge_low": args.bridge_low,
            "bridge_high": args.bridge_high,
            "aug_prob": args.aug_prob if args.aoa_bridge else 0.0,
            "lambda_min": args.lambda_min if args.aoa_bridge else 0.0,
            "lambda_max": args.lambda_max if args.aoa_bridge else 0.0,
            "max_gap_deg": args.max_gap_deg if args.aoa_bridge else 0.0,
        },
        "train_log": train_log,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    args.out.with_suffix(".json").write_text(
        json.dumps(payload["training_config"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "DONE", **payload["training_config"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
