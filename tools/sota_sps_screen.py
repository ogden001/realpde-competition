#!/usr/bin/env python3
"""Small official-v9 SPS bounds screen on the frozen 50/16 validation split."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import realpde_loss_official_v9 as core
from realpde_p0_features import P0FeatureBuilder, P0FeatureConfig


EXPECTED_SCORER_SHA256 = "a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39"
CANDIDATES = (
    {"name": "fallback", "abs": None, "rel": None},
    {"name": "abs0050_rel000", "abs": 0.0050, "rel": 0.0},
    {"name": "abs0075_rel000", "abs": 0.0075, "rel": 0.0},
    {"name": "abs0100_rel000", "abs": 0.0100, "rel": 0.0},
    {"name": "abs0125_rel000", "abs": 0.0125, "rel": 0.0},
    {"name": "abs0075_rel001", "abs": 0.0075, "rel": 0.01},
    {"name": "abs0075_rel002", "abs": 0.0075, "rel": 0.02},
)


def build_bounds(
    prediction: np.ndarray, *, abs_width: float, rel_width: float
) -> tuple[np.ndarray, np.ndarray]:
    if abs_width < 0 or rel_width < 0:
        raise ValueError("SPS widths must be non-negative")
    half = np.asarray(abs_width + rel_width * np.abs(prediction), dtype=np.float32)
    lower = np.asarray(prediction - half, dtype=np.float32)
    upper = np.asarray(prediction + half, dtype=np.float32)
    return lower, upper


def p0a_only_config(feature_config: dict) -> P0FeatureConfig:
    """Reconstruct historical P0-A semantics without inheriting P0-B defaults."""
    if not isinstance(feature_config, dict) or not all(key in feature_config for key in ("dx", "dy")):
        raise ValueError("validation checkpoint lacks P0-A feature_config")
    if feature_config.get("include_p0_a", True) is not True:
        raise ValueError("validation checkpoint feature_config explicitly disables P0-A")
    if feature_config.get("include_p0_b", False) is True:
        raise ValueError("validation checkpoint feature_config unexpectedly enables P0-B")
    allowed = {
        key: feature_config[key]
        for key in ("dx", "dy", "dt", "re_center", "re_scale")
        if key in feature_config
    }
    return P0FeatureConfig(include_p0_a=True, include_p0_b=False, **allowed)


def summarize_candidates(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("SPS screen produced no candidates")
    fallback = next((row for row in rows if row["name"] == "fallback"), None)
    if fallback is None:
        raise ValueError("SPS screen lacks fallback candidate")
    best = max(rows, key=lambda row: float(row["sps_score"]))
    return {
        "decision": "REVIEW_REQUIRED",
        "best_local_sps_candidate": best["name"],
        "best_local_sps_score": float(best["sps_score"]),
        "fallback_sps_score": float(fallback["sps_score"]),
        "best_local_sps_delta_vs_fallback": float(best["sps_score"] - fallback["sps_score"]),
    }


def load_validation_model(
    checkpoint: Path, *, expected_iteration: int, kit_root: Path, device: torch.device
) -> tuple[torch.nn.Module, P0FeatureBuilder, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("iteration") != expected_iteration:
        raise ValueError(
            f"validation checkpoint iteration {payload.get('iteration')!r} != expected {expected_iteration}"
        )
    if payload.get("feature_set") != "P0-A":
        raise ValueError("validation checkpoint is not P0-A")
    config = p0a_only_config(payload.get("feature_config"))
    builder = P0FeatureBuilder(config).to(device)
    sys.path.insert(0, str(kit_root))
    from rpde_baselines.model.cno import CNO3d

    model = CNO3d(
        in_dim=len(builder.feature_names), out_dim=3, out_dim_mult=1, in_size=64, N_layers=3
    ).to(device)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("validation checkpoint lacks model_state_dict")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, builder, payload


def infer_dev(
    *, model: torch.nn.Module, builder: P0FeatureBuilder, dev_paths: list[Path],
    batch_size: int, workers: int, seed: int, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    args = argparse.Namespace(
        max_windows=None,
        batch_size=batch_size,
        workers=workers,
        seed=seed,
    )
    dataset, loader = core.loader(dev_paths, args, shuffle=False)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    elapsed = 0.0
    with torch.inference_mode():
        for x, y, _, _ in loader:
            x = x.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            features = builder(x)
            pred = model(features.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - started
            pred[..., 2] = 0.0
            predictions.append(pred.cpu().numpy().astype(np.float32))
            targets.append(y.numpy().astype(np.float32))
    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        elapsed / max(len(dataset), 1),
        len(dataset),
    )


def official_sps_rows(prediction: np.ndarray, target: np.ndarray, kit_root: Path) -> tuple[list[dict], dict]:
    sys.path.insert(0, str(kit_root))
    import scoring

    c = scoring.measured_channels(target)
    raw_errors = {
        "rel_l2": float(np.mean(scoring.rel_l2_per_sample(prediction, target, c))),
        "tke": float(np.mean(scoring.tke_rel_l2_per_sample(prediction, target, c))),
        "mvpe": float(scoring.mvpe_rel_l2(prediction, target)),
    }
    rows: list[dict] = []
    for candidate in CANDIDATES:
        if candidate["name"] == "fallback":
            lower = upper = None
        else:
            lower, upper = build_bounds(
                prediction,
                abs_width=float(candidate["abs"]),
                rel_width=float(candidate["rel"]),
            )
        raw_sps, coverage = scoring.aggregate_sps(
            prediction,
            target,
            c,
            lower=lower,
            upper=upper,
        )
        rows.append(
            {
                **candidate,
                "sps_raw": float(raw_sps),
                "sps_score": float(scoring.score_sps(raw_sps)),
                "coverage": float(coverage),
            }
        )
    return rows, raw_errors


def run(args: argparse.Namespace) -> dict:
    scorer = args.kit_root / "scoring.py"
    if not scorer.is_file():
        raise FileNotFoundError(scorer)
    scorer_sha = core.sha256(scorer)
    if scorer_sha != args.expected_scorer_sha256:
        raise ValueError(
            f"scorer SHA256 {scorer_sha} != frozen {args.expected_scorer_sha256}"
        )
    manifest, dev_paths = core.read_manifest(args.manifest, "dev")
    if len(dev_paths) != args.expected_dev_trajectories:
        raise ValueError(
            f"dev trajectories {len(dev_paths)} != expected {args.expected_dev_trajectories}"
        )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, builder, payload = load_validation_model(
        args.checkpoint,
        expected_iteration=args.expected_iteration,
        kit_root=args.kit_root,
        device=device,
    )
    prediction, target, mean_time, windows = infer_dev(
        model=model,
        builder=builder,
        dev_paths=dev_paths,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        device=device,
    )
    rows, raw_errors = official_sps_rows(prediction, target, args.kit_root)
    summary = summarize_candidates(rows)
    result = {
        "decision": "REVIEW_REQUIRED",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": core.sha256(args.checkpoint),
        "checkpoint_iteration": int(payload["iteration"]),
        "feature_config": payload["feature_config"],
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": core.sha256(args.manifest),
        "scorer_sha256": scorer_sha,
        "dev_trajectories": len(dev_paths),
        "dev_windows": windows,
        "mean_t_neural_s": mean_time,
        "raw_errors": raw_errors,
        "candidates": rows,
        "summary": summary,
        "note": "Ranked only by official-v9 SPS on frozen dev; no Codabench/private-test use and no auto-submit.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-iteration", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-scorer-sha256", default=EXPECTED_SCORER_SHA256)
    parser.add_argument("--expected-dev-trajectories", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
