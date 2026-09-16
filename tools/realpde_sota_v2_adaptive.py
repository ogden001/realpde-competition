#!/usr/bin/env python3
"""Train/calibrate a fresh adaptive uncertainty head for frozen SOTA-V2 @32500."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from sota_v2_adaptive_runtime import AdaptiveUncertaintyHead, flow_features

SEED = 20260905
HEAD_UPDATES = 1400
HEAD_LR = 1e-3
HEAD_WEIGHT_DECAY = 1e-5
EXPECTED_ITERATION = 32_500
EXPECTED_MANIFEST_SHA = "42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347"
EXPECTED_TRAIN_WINDOWS = 2052
EXPECTED_DEV_WINDOWS = 659
EXPECTED_DEV_RAW = {
    "rel_l2": 0.09993461519479752,
    "tke": 0.4692927300930023,
    "mvpe": 0.07577798515558243,
}
FLOORS = (0.0, 0.0025, 0.005, 0.0075)
MULTS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
FIXED_GRID = tuple((floor, mult) for floor in FLOORS for mult in MULTS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def uncertainty_features(past: Tensor, prediction: Tensor) -> Tensor:
    if past.ndim != 5 or prediction.ndim != 5 or past.shape[-1] != 3 or prediction.shape[-1] != 3:
        raise ValueError("past/prediction must be [B,T,H,W,3]")
    if past.shape[:4] != prediction.shape[:4]:
        raise ValueError("past/prediction geometry mismatch")
    return torch.cat([past, flow_features(prediction[..., :2])], dim=-1)


def gaussian_nll(target_uv: Tensor, prediction_uv: Tensor, sigma_uv: Tensor) -> Tensor:
    if target_uv.shape != prediction_uv.shape or sigma_uv.shape != target_uv.shape:
        raise ValueError("target/prediction/sigma shapes are incompatible")
    return (torch.log(sigma_uv) + 0.5 * ((target_uv - prediction_uv) / sigma_uv).square()).mean()


def validate_replay_metrics(raw: dict[str, float], *, atol: float = 5e-6) -> None:
    for name, expected in EXPECTED_DEV_RAW.items():
        actual = float(raw[name])
        if abs(actual - expected) > atol:
            raise ValueError(f"replay metric mismatch for {name}: {actual} vs {expected}")


def choose_best_calibration(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("calibration rows are empty")
    return max(rows, key=lambda row: (float(row["sps"]), -float(row["mean_width_uv"])))


def _feature_config(raw: object):
    from realpde_p0_features import P0FeatureConfig
    if not isinstance(raw, dict) or "dx" not in raw or "dy" not in raw:
        raise ValueError("checkpoint lacks P0-A feature_config")
    return P0FeatureConfig(
        include_p0_a=bool(raw.get("include_p0_a", True)),
        include_p0_b=bool(raw.get("include_p0_b", False)),
        dx=float(raw["dx"]), dy=float(raw["dy"]),
        dt=None if raw.get("dt") is None else float(raw["dt"]),
        re_center=float(raw.get("re_center", 0.0)), re_scale=float(raw.get("re_scale", 1.0)),
    )


def _split_paths(manifest: Path, data_root: Path) -> tuple[list[Path], list[Path]]:
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    train = [data_root / row["file"] for row in spec["train"]]
    dev = [data_root / row["file"] for row in spec["dev"]]
    if (len(train), len(dev)) != (50, 16) or any(not path.is_file() for path in train + dev):
        raise ValueError("frozen 50/16 manifest/data mismatch")
    return train, dev


def _load_backbone(checkpoint: Path, kit_root: Path, device: torch.device):
    from realpde_mf01 import MF01CNO
    from realpde_p0_features import P0FeatureBuilder
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("iteration") != EXPECTED_ITERATION or payload.get("feature_set") != "P0-A":
        raise ValueError("validation checkpoint is not SOTA-V2 @32500 P0-A")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not state or not all(name.startswith("cno.") for name in state):
        raise ValueError("validation checkpoint is not MF01CNO state")
    config = _feature_config(payload.get("feature_config"))
    builder = P0FeatureBuilder(config).to(device)
    model = MF01CNO(kit_root, len(builder.feature_names), device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return payload, config, builder, model


def _forward(model, builder, x: Tensor) -> Tensor:
    prediction = model(builder(x)).clone()
    prediction[..., 2] = 0.0
    return prediction


def _collect(model, builder, head, paths: list[Path], *, batch_size: int, workers: int, device: torch.device):
    from realpde_p0_data import H5WindowDataset
    ds = H5WindowDataset(paths, in_steps=20, out_steps=20, stride=20, sub_sample=2,
                         include_pressure=False, window_mode="fixed")
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    predictions, sigmas, targets = [], [], []
    with torch.inference_mode():
        for x, y, _, _ in loader:
            x = x.to(device, non_blocking=True)
            pred = _forward(model, builder, x)
            features = uncertainty_features(x, pred).permute(0, 4, 1, 2, 3)
            sigma = head(features).permute(0, 2, 3, 4, 1)
            predictions.append(pred.cpu().numpy().astype(np.float32))
            sigmas.append(sigma.cpu().numpy().astype(np.float32))
            targets.append(y.numpy().astype(np.float32))
    return ds, np.concatenate(predictions), np.concatenate(sigmas), np.concatenate(targets)


def _raw_metrics(pred: np.ndarray, target: np.ndarray, scoring) -> dict[str, float]:
    channels = scoring.measured_channels(target)
    return {
        "rel_l2": float(np.mean(scoring.rel_l2_per_sample(pred, target, channels))),
        "tke": float(np.mean(scoring.tke_rel_l2_per_sample(pred, target, channels))),
        "mvpe": float(scoring.mvpe_rel_l2(pred, target)),
    }


def _score_sps(raw_sps: float, scoring) -> float:
    return float(scoring.score_sps(raw_sps))


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if sha256(args.manifest) != EXPECTED_MANIFEST_SHA:
        raise ValueError("frozen manifest SHA mismatch")
    torch.manual_seed(SEED); np.random.seed(SEED)
    train_paths, dev_paths = _split_paths(args.manifest, args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")
    payload, config, builder, model = _load_backbone(args.validation_checkpoint, args.kit_root, device)

    sys.path.insert(0, str(args.kit_root.resolve()))
    import scoring
    replay_head = AdaptiveUncertaintyHead().to(device).eval()
    dev_ds, dev_pred, _, dev_target = _collect(model, builder, replay_head, dev_paths,
                                                batch_size=args.batch_size, workers=args.workers, device=device)
    raw = _raw_metrics(dev_pred, dev_target, scoring)
    validate_replay_metrics(raw, atol=args.replay_tolerance)
    if len(dev_ds) != EXPECTED_DEV_WINDOWS:
        raise ValueError(f"dev windows {len(dev_ds)} != {EXPECTED_DEV_WINDOWS}")

    from realpde_p0_data import H5WindowDataset
    train_ds = H5WindowDataset(train_paths, in_steps=20, out_steps=20, stride=20, sub_sample=2,
                               include_pressure=False, window_mode="fixed")
    if len(train_ds) != EXPECTED_TRAIN_WINDOWS:
        raise ValueError(f"train windows {len(train_ds)} != {EXPECTED_TRAIN_WINDOWS}")
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                         pin_memory=True, drop_last=True)
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "validation_backbone_replay.json").write_text(json.dumps({
        "validation_checkpoint": str(args.validation_checkpoint),
        "validation_checkpoint_sha256": sha256(args.validation_checkpoint),
        "iteration": payload.get("iteration"), "raw_errors": raw,
        "windows": len(dev_ds), "trajectories": len(dev_paths),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    head = AdaptiveUncertaintyHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=HEAD_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HEAD_UPDATES)
    iterator = iter(loader); losses = []; started = time.monotonic()
    head.train()
    for update in range(1, HEAD_UPDATES + 1):
        try:
            x, y, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader); x, y, _, _ = next(iterator)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.no_grad():
            pred = _forward(model, builder, x)
            features = uncertainty_features(x, pred).permute(0, 4, 1, 2, 3)
        sigma = head(features).permute(0, 2, 3, 4, 1)
        loss = gaussian_nll(y[..., :2], pred[..., :2], sigma)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite head loss @ {update}")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        if update == 1 or update % 100 == 0 or update == HEAD_UPDATES:
            losses.append({"update": update, "loss": float(loss.detach().cpu()), "lr": float(optimizer.param_groups[0]["lr"])})

    head.eval()
    head_path = args.out_dir / "adaptive_head_1400.pth"
    torch.save({
        "head_state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        "metadata": {
            "seed": SEED, "updates": HEAD_UPDATES, "lr": HEAD_LR, "weight_decay": HEAD_WEIGHT_DECAY,
            "architecture": {"in_channels": 15, "hidden": 32, "blocks": 2},
            "validation_checkpoint_sha256": sha256(args.validation_checkpoint),
            "validation_checkpoint_iteration": EXPECTED_ITERATION,
            "manifest_sha256": EXPECTED_MANIFEST_SHA,
            "train_trajectories": len(train_paths), "train_windows": len(train_ds),
            "feature_config": vars(config),
        },
    }, head_path)
    (args.out_dir / "head_training_summary.json").write_text(json.dumps({
        "head": str(head_path), "head_sha256": sha256(head_path), "loss_curve": losses,
        "elapsed_seconds": time.monotonic() - started,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    dev_ds, dev_pred2, dev_sigma, dev_target2 = _collect(model, builder, head, dev_paths,
                                                         batch_size=args.batch_size, workers=args.workers, device=device)
    raw2 = _raw_metrics(dev_pred2, dev_target2, scoring)
    validate_replay_metrics(raw2, atol=args.replay_tolerance)
    if not np.array_equal(dev_target, dev_target2) or np.max(np.abs(dev_pred - dev_pred2)) > 1e-7:
        raise RuntimeError("uncertainty head altered or reordered backbone predictions")
    channels = scoring.measured_channels(dev_target2)
    rows = []
    for floor, mult in FIXED_GRID:
        half_uv = floor + mult * dev_sigma
        half = np.concatenate([half_uv, np.zeros(half_uv.shape[:-1] + (1,), dtype=np.float32)], axis=-1)
        raw_sps, coverage = scoring.aggregate_sps(dev_pred2, dev_target2, channels, dev_pred2 - half, dev_pred2 + half)
        rows.append({
            "floor": floor, "mult": mult, "sps": _score_sps(float(raw_sps), scoring),
            "coverage": float(coverage), "mean_width_uv": float(np.mean(2.0 * half_uv)),
        })
    best = choose_best_calibration(rows)
    static_half = (0.0075 + 0.02 * np.abs(dev_pred2)).astype(np.float32)
    static_raw, static_coverage = scoring.aggregate_sps(dev_pred2, dev_target2, channels,
                                                        dev_pred2 - static_half, dev_pred2 + static_half)
    static = {"abs": 0.0075, "rel": 0.02, "sps": _score_sps(float(static_raw), scoring),
              "coverage": float(static_coverage), "mean_width_uv": float(np.mean(2.0 * static_half[..., :2]))}
    gate = "ADAPTIVE_GO" if float(best["sps"]) > float(static["sps"]) else "ADAPTIVE_NO_GO"
    with (args.out_dir / "calibration_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.out_dir / "calibration_grid.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "status": "REVIEW_REQUIRED", "gate": gate, "best": best, "static_reference": static,
        "delta_sps": float(best["sps"]) - float(static["sps"]), "raw_errors": raw2,
        "head": str(head_path), "head_sha256": sha256(head_path), "grid_rows": len(rows),
        "dev_windows": len(dev_ds), "dev_trajectories": len(dev_paths),
        "validation_checkpoint_sha256": sha256(args.validation_checkpoint),
        "scorer_sha256": sha256(args.kit_root / "scoring.py"),
    }
    (args.out_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--replay-tolerance", type=float, default=5e-6)
    parser.add_argument("--require-cuda", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
