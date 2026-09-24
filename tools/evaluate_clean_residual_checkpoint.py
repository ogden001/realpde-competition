#!/usr/bin/env python3
"""Evaluate a frozen residual checkpoint on a manifest without training.

Supports both the direct-CNO clean baseline and the P0-A/MF strong-backbone
adapter.  This is an evaluation-only tool used by CLEAN_BASELINE_FINAL_CAMPAIGN.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from colleague_80pt.realpde_h5_feature_adapter_train import H5WindowDataset, paths_from_split_manifest
from colleague_80pt.residual_multi import (
    CorrectorConfig,
    ResidualCorrectionModel,
    ResidualCorrector3D,
    load_frozen_base,
    load_residual_checkpoint,
    write_final_evidence,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--real-root", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--backbone-checkpoint", type=Path, required=True)
    p.add_argument("--residual-checkpoint", type=Path, required=True)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--base-model", choices=("cno", "sota_v2_mf"), required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--require-cuda", action="store_true")
    args = p.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    train_paths, dev_paths = paths_from_split_manifest(args.real_root, args.split_manifest)
    if set(p.name for p in train_paths) & set(p.name for p in dev_paths):
        raise ValueError("evaluation manifest has train/dev overlap")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")

    payload = torch.load(args.residual_checkpoint, map_location="cpu", weights_only=False)
    cfg_raw = payload.get("corrector_config")
    if not isinstance(cfg_raw, dict):
        raise ValueError("residual checkpoint missing corrector_config")
    base = load_frozen_base(args.base_model, args.backbone_checkpoint, args.model_root, device)
    model = ResidualCorrectionModel(base, ResidualCorrector3D(CorrectorConfig(**cfg_raw))).to(device)
    load_residual_checkpoint(model, args.residual_checkpoint, device)
    model.eval()

    ds = H5WindowDataset(
        dev_paths,
        in_steps=20,
        out_steps=20,
        stride=20,
        sub_sample=2,
        include_pressure=False,
        window_mode="fixed",
        preload_to_ram=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda")
    result = write_final_evidence(model, loader, device, out_dir=args.out_dir,
                                  experiment=args.out_dir.name, alpha=1.0)
    (args.out_dir / "manifest.json").write_text(json.dumps({
        "status": "REVIEW_REQUIRED",
        "base_model": args.base_model,
        "backbone_checkpoint": str(args.backbone_checkpoint),
        "residual_checkpoint": str(args.residual_checkpoint),
        "train_trajectories_in_manifest": len(train_paths),
        "eval_trajectories": len(dev_paths),
        "eval_windows": len(ds),
        "result": result,
        "optimizer_steps": 0,
        "codabench_accessed": False,
        "locked_final_accessed": False,
    }, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "DONE").touch()


if __name__ == "__main__":
    main()
