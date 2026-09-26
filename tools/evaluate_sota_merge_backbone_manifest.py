#!/usr/bin/env python3
"""Evaluate any frozen P0-A MF backbone checkpoint on a manifest.

Evaluation only. No optimizer step, SPS tuning, full-data refit, locked-final
access, private data access, packaging, or Codabench action is performed.
"""
from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

import realpde_sota_v2_integrated as strong
import realpde_teammate_residual_transfer as transfer
import train_sota_merge_backbone as merge
from realpde_adaptive_probe import feature_config_from_checkpoint
from realpde_p0_features import P0FeatureBuilder

STATUS = "REVIEW_REQUIRED"


def manifest_paths(data_root: Path, manifest: Path) -> list[Path]:
    merge.assert_safe_path(data_root)
    merge.assert_safe_path(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = payload.get("dev")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest must contain a non-empty dev list")
    names = [str(row["file"] if isinstance(row, dict) else row) for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("duplicate dev filenames")
    train_rows = payload.get("train", [])
    train_names = {
        str(row["file"] if isinstance(row, dict) else row)
        for row in train_rows
    }
    if train_names & set(names):
        raise ValueError("manifest train/dev overlap")
    paths = [data_root / name for name in names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing evaluation trajectories: {missing[:10]}")
    return paths


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, object]:
    for path in (args.data_root, args.manifest, args.kit_root, args.checkpoint):
        merge.assert_safe_path(path)
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)

    paths = manifest_paths(args.data_root, args.manifest)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("feature_set") != "P0-A":
        raise ValueError("checkpoint is not a P0-A MF checkpoint")
    if "model_state_dict" not in payload:
        raise ValueError("checkpoint missing model_state_dict")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")

    cfg = feature_config_from_checkpoint(payload)
    builder = P0FeatureBuilder(cfg).to(device)
    model = strong.MF01CNO(args.kit_root, len(builder.feature_names), device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    ds, loader = strong.dev_loader(
        paths,
        Namespace(eval_batch_size=args.eval_batch_size, workers=args.workers),
    )
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        pred = strong.forward_mf(model, builder, x)
        preds.append(pred.cpu().numpy().astype(np.float32))
        targets.append(y.numpy().astype(np.float32))

    prediction = np.concatenate(preds)
    target = np.concatenate(targets)
    raw = transfer.raw_physical_errors(args.kit_root, prediction, target)
    result = {
        "status": STATUS,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": strong.sha256(args.checkpoint),
        "stage": payload.get("stage"),
        "iteration": payload.get("iteration"),
        "source_stage_a_update": payload.get("metadata", {}).get("source_stage_a_update"),
        "split": args.split_name,
        "trajectories": len(paths),
        "windows": len(ds),
        "rel_l2": float(raw["rel_l2"]),
        "tke": float(raw["tke"]),
        "mvpe": float(raw["mvpe"]),
        "point_score": merge.point_score(raw),
        "optimizer_steps": 0,
        "selection_allowed": False,
        "locked_final_accessed": False,
        "private_accessed": False,
        "codabench_accessed": False,
    }
    merge.dump(args.out_dir / "metrics.json", result)
    merge.dump(
        args.out_dir / "horizon_error_summary.json",
        strong.horizon_error_summary(prediction, target),
    )
    (args.out_dir / "DONE").touch()
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--kit-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--split-name", default="eval")
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--require-cuda", action="store_true")
    args = p.parse_args()
    print(json.dumps(evaluate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
