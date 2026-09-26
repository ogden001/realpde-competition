#!/usr/bin/env python3
"""Runtime helpers for REALPDE_SOTA_MERGE_JOINT_V1.

This module contains only glue around existing Stage-A/Stage-B/Residual
components. It intentionally defines no new model architecture.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

import realpde_residual_corrector_projection as projection
import realpde_sota_v2_integrated as strong
import realpde_teammate_residual_transfer as transfer
import train_sota_merge_backbone as merge
from realpde_adaptive_probe import ResidualCorrector3D, feature_config_from_checkpoint
from realpde_p0_features import P0FeatureBuilder

PROTOCOL = "REALPDE_SOTA_MERGE_JOINT_V1"
SOURCE_STAGE_A_UPDATE = 57_000


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def atomic_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_backbone(path: Path, kit_root: Path, device: torch.device):
    merge.assert_safe_path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("iteration", -1)) != SOURCE_STAGE_A_UPDATE:
        raise ValueError(f"requires Stage-A @{SOURCE_STAGE_A_UPDATE}")
    if payload.get("stage") != "A" or payload.get("feature_set") != "P0-A":
        raise ValueError("source must be a P0-A Stage-A checkpoint")
    cfg = feature_config_from_checkpoint(payload)
    builder = P0FeatureBuilder(cfg).to(device)
    model = strong.MF01CNO(kit_root, len(builder.feature_names), device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return payload, cfg, builder, model


def load_corrector(path: Path | None, source_backbone_sha: str, device: torch.device):
    model = ResidualCorrector3D(in_channels=42, hidden=64, blocks=2, max_delta=0.04).to(device)
    if path is None:
        return model, {"mode": "fresh_init", "checkpoint": None, "sha256": None}
    merge.assert_safe_path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    bound = payload.get("backbone_sha256")
    if bound is not None and bound != source_backbone_sha:
        raise ValueError("corrector is bound to a different backbone; use fresh init")
    arch = payload.get("architecture", {})
    expected = {"in_channels": 42, "hidden": 64, "blocks": 2, "max_delta": 0.04}
    for key, value in expected.items():
        if key in arch and arch[key] != value:
            raise ValueError(f"corrector architecture mismatch: {key}")
    model.load_state_dict(payload["corrector_state_dict"], strict=True)
    return model, {
        "mode": "checkpoint",
        "checkpoint": str(path),
        "sha256": strong.sha256(path),
        "bound_backbone_sha256": bound,
        "updates": payload.get("updates"),
    }


def forward_final(backbone, builder, corrector, x):
    base_pred = strong.forward_mf(backbone, builder, x)
    delta = transfer._delta_from_corrector(corrector, x, base_pred)
    corrected = transfer.apply_scaled_correction(base_pred, delta, 1.0)
    final_pred = projection.spatial_tke_map_projection(base_pred, corrected)
    return base_pred, final_pred


@torch.no_grad()
def evaluate_pair(backbone, builder, corrector, paths, *, kit_root, workers,
                  eval_batch_size, device, out_dir, update, split_name):
    out_dir.mkdir(parents=True, exist_ok=True)
    ds, loader = strong.dev_loader(
        paths, argparse.Namespace(eval_batch_size=eval_batch_size, workers=workers)
    )
    backbone.eval()
    corrector.eval()
    bp, fp, ys = [], [], []
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        base_pred, final_pred = forward_final(backbone, builder, corrector, x)
        bp.append(base_pred.cpu().numpy().astype(np.float32))
        fp.append(final_pred.cpu().numpy().astype(np.float32))
        ys.append(y.numpy().astype(np.float32))
    base_np, final_np, target_np = np.concatenate(bp), np.concatenate(fp), np.concatenate(ys)
    base_raw = transfer.raw_physical_errors(kit_root, base_np, target_np)
    final_raw = transfer.raw_physical_errors(kit_root, final_np, target_np)
    dump(out_dir / "base_horizon_error_summary.json", strong.horizon_error_summary(base_np, target_np))
    dump(out_dir / "final_horizon_error_summary.json", strong.horizon_error_summary(final_np, target_np))
    row = {
        "split": split_name,
        "update": int(update),
        "windows": len(ds),
        "base_rel_l2": float(base_raw["rel_l2"]),
        "base_tke": float(base_raw["tke"]),
        "base_mvpe": float(base_raw["mvpe"]),
        "base_point": merge.point_score(base_raw),
        "final_rel_l2": float(final_raw["rel_l2"]),
        "final_tke": float(final_raw["tke"]),
        "final_mvpe": float(final_raw["mvpe"]),
        "final_point": merge.point_score(final_raw),
    }
    dump(out_dir / "metrics.json", row)
    return row


def save_recovery(path: Path, *, backbone, corrector, optimizer, scheduler, cfg,
                  source_backbone_sha, corrector_source, update,
                  stage_a_epoch, stage_a_offset, stage_b_epoch, stage_b_offset, history):
    payload = {
        "protocol": PROTOCOL,
        "joint_update": int(update),
        "source_stage_a_update": SOURCE_STAGE_A_UPDATE,
        "source_backbone_sha256": source_backbone_sha,
        "corrector_source": corrector_source,
        "backbone_state_dict": backbone.state_dict(),
        "corrector_state_dict": corrector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "feature_config": vars(cfg),
        "stage_a_sampler_epoch": int(stage_a_epoch),
        "stage_a_sampler_offset": int(stage_a_offset),
        "stage_b_sampler_epoch": int(stage_b_epoch),
        "stage_b_sampler_offset": int(stage_b_offset),
        "history": history,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    atomic_save(payload, path)


def restore_recovery(path: Path, backbone, corrector, optimizer, scheduler):
    p = torch.load(path, map_location="cpu", weights_only=False)
    if p.get("protocol") != PROTOCOL:
        raise ValueError("recovery protocol mismatch")
    backbone.load_state_dict(p["backbone_state_dict"], strict=True)
    corrector.load_state_dict(p["corrector_state_dict"], strict=True)
    optimizer.load_state_dict(p["optimizer_state_dict"])
    scheduler.load_state_dict(p["scheduler_state_dict"])
    torch.set_rng_state(p["torch_rng_state"])
    np.random.set_state(p["numpy_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in p:
        torch.cuda.set_rng_state_all(p["cuda_rng_state_all"])
    return p


def save_model_pair(out_dir: Path, *, update, backbone, corrector, cfg,
                    source_backbone_sha, corrector_source):
    ck = out_dir / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    bp = ck / f"backbone_joint_{update:06d}.pth"
    atomic_save({
        "model_state_dict": {k: v.detach().cpu() for k, v in backbone.state_dict().items()},
        "iteration": SOURCE_STAGE_A_UPDATE,
        "joint_update": int(update),
        "stage": "JOINT",
        "scope": "clean",
        "feature_set": "P0-A",
        "feature_config": vars(cfg),
        "source_stage_a_update": SOURCE_STAGE_A_UPDATE,
        "source_backbone_sha256": source_backbone_sha,
        "protocol": PROTOCOL,
    }, bp)
    cp = ck / f"corrector_joint_{update:06d}.pth"
    atomic_save({
        "corrector_state_dict": {k: v.detach().cpu() for k, v in corrector.state_dict().items()},
        "architecture": {"in_channels": 42, "hidden": 64, "blocks": 2, "max_delta": 0.04},
        "backbone_sha256": strong.sha256(bp),
        "source_backbone_sha256": source_backbone_sha,
        "joint_update": int(update),
        "source_corrector": corrector_source,
        "protocol": PROTOCOL,
    }, cp)
    return bp, cp
