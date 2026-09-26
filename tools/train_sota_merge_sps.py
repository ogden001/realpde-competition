#!/usr/bin/env python3
"""SPS optimization for a frozen REALPDE_SOTA_MERGE_JOINT_V1 checkpoint pair.

Thin glue only:
- reuses the existing colleague-80 uncertainty Head3D;
- reuses Exp3 static/adaptive SPS calibration and width-constrained selector;
- reuses the joint point-model forward path;
- never updates the point predictor.

Selection is Seen-Dev only. AoA10 is evaluated once, unchanged, only after the
Seen-Dev gate is GO. No locked-final/private/Codabench access is performed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import realpde_sota_merge_joint_runtime as joint
import realpde_sota_v2_integrated as strong
import train_clean_residual_aware_sps as exp3
from colleague_80pt.train_head_fast import Head3D, HeadConfig, initialize_coupled
from realpde_adaptive_probe import feature_config_from_checkpoint
from realpde_mf01 import MF01CNO
from realpde_p0_features import P0FeatureBuilder
from realpde_p0_data import H5WindowDataset
from realpde_adaptive_probe import ResidualCorrector3D

STATUS = "REVIEW_REQUIRED"
PROTOCOL = "REALPDE_SOTA_MERGE_SPS_V1"
SEED = 41
UPDATES = 5000
EVAL_EVERY = 500
BATCH = 16
LR = 1e-3
WEIGHT_DECAY = 1e-5
WARMUP = 200
EPS = 1e-6


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def manifest_paths(manifest: Path, split: str, root: Path) -> list[Path]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = payload.get(split)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"manifest missing {split}")
    paths = [root / str(row["file"] if isinstance(row, dict) else row) for row in rows]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing[:5])
    return paths


def fixed_dataset(paths: list[Path], stride: int) -> H5WindowDataset:
    return H5WindowDataset(paths, in_steps=20, out_steps=20, stride=stride,
                           sub_sample=2, include_pressure=False, window_mode="fixed")


def load_joint_pair(backbone_path: Path, corrector_path: Path, kit_root: Path, device: torch.device):
    for p in (backbone_path, corrector_path, kit_root):
        joint.merge.assert_safe_path(p)
    bp = torch.load(backbone_path, map_location="cpu", weights_only=False)
    cp = torch.load(corrector_path, map_location="cpu", weights_only=False)
    if bp.get("protocol") != joint.PROTOCOL or cp.get("protocol") != joint.PROTOCOL:
        raise ValueError("checkpoint pair is not REALPDE_SOTA_MERGE_JOINT_V1")
    if int(bp.get("joint_update", -1)) != int(cp.get("joint_update", -2)):
        raise ValueError("backbone/corrector joint_update mismatch")
    expected_backbone_sha = strong.sha256(backbone_path)
    if cp.get("backbone_sha256") != expected_backbone_sha:
        raise ValueError("corrector is not bound to this joint backbone")
    cfg = feature_config_from_checkpoint(bp)
    builder = P0FeatureBuilder(cfg).to(device)
    backbone = MF01CNO(kit_root, len(builder.feature_names), device)
    backbone.load_state_dict(bp["model_state_dict"], strict=True)
    arch = cp.get("architecture", {})
    corrector = ResidualCorrector3D(
        in_channels=int(arch.get("in_channels", 42)),
        hidden=int(arch.get("hidden", 64)),
        blocks=int(arch.get("blocks", 2)),
        max_delta=float(arch.get("max_delta", 0.04)),
    ).to(device)
    corrector.load_state_dict(cp["corrector_state_dict"], strict=True)
    backbone.eval(); corrector.eval()
    for p in backbone.parameters(): p.requires_grad_(False)
    for p in corrector.parameters(): p.requires_grad_(False)
    return bp, cp, cfg, builder, backbone, corrector


@torch.no_grad()
def point_forward(backbone, builder, corrector, x):
    return joint.forward_final(backbone, builder, corrector, x)


@torch.no_grad()
def collect(backbone, builder, corrector, head, paths, device, workers):
    ds = fixed_dataset(paths, 20)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")
    preds, sigmas, targets = [], [], []
    backbone.eval(); corrector.eval(); head.eval()
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        base, final = point_forward(backbone, builder, corrector, x)
        log_std = head(x, base)
        preds.append(final.cpu().numpy().astype(np.float32))
        sigmas.append(torch.exp(log_std).cpu().numpy().astype(np.float32))
        targets.append(y.numpy().astype(np.float32))
    return ds, np.concatenate(preds), np.concatenate(sigmas), np.concatenate(targets)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True, help="clean Train51/Seen-Dev12 manifest")
    p.add_argument("--aoa10-manifest", type=Path, required=True, help="manifest whose dev/holdout list is AoA10")
    p.add_argument("--aoa10-split", default="dev")
    p.add_argument("--kit-root", type=Path, required=True)
    p.add_argument("--backbone-checkpoint", type=Path, required=True)
    p.add_argument("--corrector-checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--require-cuda", action="store_true")
    args = p.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")
    torch.manual_seed(SEED); np.random.seed(SEED)

    train_paths = manifest_paths(args.manifest, "train", args.data_root)
    dev_paths = manifest_paths(args.manifest, "dev", args.data_root)
    aoa_paths = manifest_paths(args.aoa10_manifest, args.aoa10_split, args.data_root)
    if (len(train_paths), len(dev_paths)) != (51, 12):
        raise ValueError(f"expected clean 51/12, got {len(train_paths)}/{len(dev_paths)}")
    if {p.name for p in aoa_paths} & ({p.name for p in train_paths} | {p.name for p in dev_paths}):
        raise ValueError("AoA10 overlaps Train51/Seen-Dev12")

    bp, cp, cfg, builder, backbone, corrector = load_joint_pair(
        args.backbone_checkpoint, args.corrector_checkpoint, args.kit_root, device
    )

    # Exp3 architecture/semantics: uncertainty sees Past20 + pre-residual base;
    # supervision is the error of the post-residual final point prediction.
    hcfg = HeadConfig(hidden=64, blocks=2, dropout=0.0, include_pressure=True,
                      history_context=False, sigma0=0.02, min_sigma=1e-4,
                      max_sigma=1.0, include_delta=False)
    head = Head3D(hcfg).to(device)
    initialize_coupled(head, seed=SEED)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARMUP) *
        (0.5 * (1 + np.cos(np.pi * min(1.0, s / UPDATES))))
    )

    train_ds = fixed_dataset(train_paths, 5)
    gen = torch.Generator(); gen.manual_seed(SEED)
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, generator=gen,
                        drop_last=True, num_workers=args.workers,
                        pin_memory=device.type == "cuda")
    iterator = iter(loader)

    _, pred_before, _, target_before = collect(
        backbone, builder, corrector, head, dev_paths, device, args.workers
    )
    static_grid, static_best = exp3.calibrate_static(pred_before, target_before)
    dump(args.out_dir / "static_calibration_grid.json", static_grid)

    best = None; best_state = None; evals = []; losses = []
    started = time.monotonic()
    for step in range(1, UPDATES + 1):
        try:
            x, y, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader); x, y, _, _ = next(iterator)
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with torch.no_grad():
            base, final = point_forward(backbone, builder, corrector, x)
        log_std = head(x, base)
        err = (final[..., :2] - y[..., :2]).abs()
        mask = (y[..., :2] != 0.0).to(log_std.dtype)
        # Preserve the validated Exp3 log-MAE uncertainty objective.
        loss = ((log_std - torch.log(err + EPS)).abs() * mask).sum() / mask.sum().clamp_min(1.0)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite SPS loss @{step}")
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step(); sched.step()
        if step == 1 or step % 100 == 0:
            losses.append({"step": step, "loss": float(loss.detach().cpu()),
                           "lr": float(opt.param_groups[0]["lr"])})
        if step % EVAL_EVERY == 0:
            _, pred_d, sigma_d, target_d = collect(
                backbone, builder, corrector, head, dev_paths, device, args.workers
            )
            grid, unconstrained = exp3.calibrate_adaptive(pred_d, target_d, sigma_d)
            chosen = exp3.select_adaptive_under_width_cap(grid, static_best)
            rec = {"step": step, "best": chosen, "unconstrained_best": unconstrained}
            evals.append(rec)
            dump(args.out_dir / f"calibration_grid_{step:05d}.json", grid)
            if best is None or chosen["sps"] > best["best"]["sps"]:
                best = rec
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    if best is None or best_state is None:
        raise RuntimeError("no SPS checkpoint selected")
    head.load_state_dict(best_state, strict=True); head.eval()
    _, pred_after, sigma_after, target_after = collect(
        backbone, builder, corrector, head, dev_paths, device, args.workers
    )
    parity = float(np.max(np.abs(pred_after - pred_before)))
    selected = best["best"]
    candidate_dev = exp3.adaptive_score(
        pred_after, target_after, sigma_after, selected["floor"],
        selected["mult_u"], selected["mult_v"], selected["rel"]
    )
    dev_gain = candidate_dev["sps"] - static_best["sps"]
    width_ratio = candidate_dev["mean_width_uv"] / max(static_best["mean_width_uv"], 1e-12)
    checks = {
        "sps_gain_ge_min": dev_gain >= exp3.MIN_DEV_SPS_GAIN,
        "width_ratio_le_max": width_ratio <= exp3.MAX_DEV_WIDTH_RATIO + 1e-12,
        "point_parity_le_tol": parity <= exp3.POINT_PARITY_TOL,
    }
    gate = "GO" if all(checks.values()) else "NO_GO"

    summary = {
        "status": STATUS, "protocol": PROTOCOL,
        "joint_update": int(bp["joint_update"]),
        "backbone_sha256": strong.sha256(args.backbone_checkpoint),
        "corrector_sha256": strong.sha256(args.corrector_checkpoint),
        "point_model_frozen": True, "optimizer_updates_point_model": 0,
        "uncertainty_updates": UPDATES, "selected_step": best["step"],
        "selected_calibration": selected, "static_seen_dev": static_best,
        "adaptive_seen_dev": candidate_dev, "seen_dev_sps_gain": dev_gain,
        "seen_dev_width_ratio": width_ratio, "point_prediction_parity_max_abs": parity,
        "checks": checks, "gate": gate, "training_wall_seconds": time.monotonic() - started,
        "aoa10_accessed": False, "aoa10_used_for_selection": False,
        "locked_final_accessed": False, "private_accessed": False,
        "codabench_accessed": False,
    }

    # AoA10 is a one-shot generalization audit, never a selector/calibrator.
    if gate == "GO":
        _, pred_h, sigma_h, target_h = collect(
            backbone, builder, corrector, head, aoa_paths, device, args.workers
        )
        static_h = exp3.static_score(
            pred_h, target_h, static_best["abs"], static_best["rel"]
        )
        adaptive_h = exp3.adaptive_score(
            pred_h, target_h, sigma_h, selected["floor"],
            selected["mult_u"], selected["mult_v"], selected["rel"]
        )
        summary.update({
            "aoa10_accessed": True, "static_aoa10": static_h,
            "adaptive_aoa10": adaptive_h,
            "aoa10_sps_gain": adaptive_h["sps"] - static_h["sps"],
        })

    torch.save({
        "head_state_dict": best_state, "head_config": hcfg.__dict__,
        "selected_step": best["step"], "calibration": selected,
        "protocol": PROTOCOL, "joint_update": int(bp["joint_update"]),
        "backbone_sha256": summary["backbone_sha256"],
        "corrector_sha256": summary["corrector_sha256"],
    }, args.out_dir / "head_best.pth")
    dump(args.out_dir / "training_progress.json", {"losses": losses, "evals": evals})
    dump(args.out_dir / "summary.json", summary)
    (args.out_dir / "DONE").touch()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
