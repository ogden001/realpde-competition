#!/usr/bin/env python3
"""Run the frozen 4x7 adaptive-SPS calibration grid on validation dev only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import realpde_b1_p0a_n2 as base_api
from realpde_adaptive_probe import (AdaptiveUncertaintyHead, ResidualCorrector3D, adaptive_features,
                                    fixed_calibration_grid, flow_features, feature_config_from_checkpoint,
                                    assert_feature_config_matches_checkpoint)
from realpde_p0_data import H5WindowDataset
from realpde_p0_features import P0FeatureBuilder


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def calibration_model_names(corrected_probe: Path | None) -> tuple[str, ...]:
    return ("base", "corrected") if corrected_probe is not None else ("base",)


def score_sps_percent(raw_sps: float, scoring) -> float:
    """Convert official scorer's normalized SPS to the displayed 0–100 scale."""
    return float(scoring.score_sps(raw_sps))


def run(*, data_root: Path, kit_root: Path, checkpoint: Path, manifest: Path,
        validation_probe: Path, corrected_probe: Path | None, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(manifest.read_text(encoding="utf-8")); root = data_root
    train = [root / row["file"] for row in spec["train"]]; dev = [root / row["file"] for row in spec["dev"]]
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = feature_config_from_checkpoint(checkpoint_payload)
    assert_feature_config_matches_checkpoint(cfg, checkpoint_payload)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); builder = P0FeatureBuilder(cfg).to(device)
    base = base_api.load_model(kit_root, checkpoint, builder, device).eval()
    vp = torch.load(validation_probe, map_location="cpu", weights_only=False)
    cp = None if corrected_probe is None else torch.load(corrected_probe, map_location="cpu", weights_only=False)
    corrector = None
    corrected_head = None
    if cp is not None:
        corrector = ResidualCorrector3D(hidden=64, blocks=2).to(device)
        corrector.load_state_dict(cp["corrector_state_dict"] or vp["corrector_state_dict"])
        corrector.eval()
        corrected_head = AdaptiveUncertaintyHead(hidden=32, blocks=2).to(device)
        corrected_head.load_state_dict(cp["corrected_head_state_dict"])
        corrected_head.eval()
    base_head = AdaptiveUncertaintyHead(hidden=32, blocks=2).to(device); base_head.load_state_dict(vp["base_head_state_dict"]); base_head.eval()
    ds = H5WindowDataset(dev, include_pressure=True); loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    base_pred, final_pred, base_sigma, corrected_sigma, targets = [], [], [], [], []
    with torch.no_grad():
        for x, y, _, _ in loader:
            x = x.to(device); bp = base_api.forward(base, builder, x)
            features = torch.cat([x, flow_features(bp[..., :2])], dim=-1).permute(0, 4, 1, 2, 3)
            base_pred.append(bp.cpu().numpy()); base_sigma.append(base_head(features).permute(0, 2, 3, 4, 1).cpu().numpy())
            if corrector is not None and corrected_head is not None:
                d = corrector(adaptive_features(x, bp).permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
                final_pred.append((bp + d).cpu().numpy())
                corrected_sigma.append(corrected_head(features).permute(0, 2, 3, 4, 1).cpu().numpy())
            targets.append(y.numpy())
    base_pred, base_sigma, targets = [np.concatenate(v).astype(np.float32) for v in (base_pred, base_sigma, targets)]
    if corrector is not None:
        final_pred, corrected_sigma = [np.concatenate(v).astype(np.float32) for v in (final_pred, corrected_sigma)]
    sys.path.insert(0, str(kit_root.resolve())); import scoring
    rows = []
    model_data = {"base": (base_pred, base_sigma)}
    if corrector is not None:
        model_data["corrected"] = (final_pred, corrected_sigma)
    for model_name in calibration_model_names(corrected_probe):
        pred, sigma = model_data[model_name]
        c = scoring.measured_channels(targets)
        raw = {"rel_l2": float(np.mean(scoring.rel_l2_per_sample(pred, targets, c))), "tke": float(np.mean(scoring.tke_rel_l2_per_sample(pred, targets, c))), "mvpe": float(scoring.mvpe_rel_l2(pred, targets))}
        for floor, mult in fixed_calibration_grid():
            uv_half = (floor + mult * sigma).astype(np.float32)
            half = np.concatenate([uv_half, np.zeros(uv_half.shape[:-1] + (1,), dtype=np.float32)], axis=-1)
            lo, hi = pred - half, pred + half
            sps, coverage = scoring.aggregate_sps(pred, targets, c, lo, hi)
            rows.append({"model": model_name, "floor": floor, "mult": mult, "sps": 100.0 * float(sps), "coverage": float(coverage), "mean_width_uv": float(np.mean(2.0 * half[..., :2])), **raw})
    rows.sort(key=lambda row: (row["model"], -row["sps"]))
    half = (0.0075 + 0.02 * np.abs(base_pred)).astype(np.float32)
    c = scoring.measured_channels(targets)
    static_raw, static_coverage = scoring.aggregate_sps(base_pred, targets, c, base_pred - half, base_pred + half)
    result = {
        "status": "REVIEW_REQUIRED",
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_iteration": checkpoint_payload.get("iteration"),
        "feature_config": checkpoint_payload.get("feature_config"),
        "manifest": str(manifest.resolve()), "manifest_sha256": _sha256(manifest),
        "validation_probe": str(validation_probe.resolve()), "validation_probe_sha256": _sha256(validation_probe),
        "corrected_probe": None if corrected_probe is None else str(corrected_probe.resolve()),
        "corrected_probe_sha256": None if corrected_probe is None else _sha256(corrected_probe),
        "scorer_sha256": _sha256(kit_root / "scoring.py"),
        "rows": rows, "grid_size": len(rows), "windows": len(ds), "trajectories": len(dev),
        "static_reference": {"abs": 0.0075, "rel": 0.02, "sps": score_sps_percent(static_raw, scoring), "coverage": float(static_coverage)},
        "note": "Base-head-only fixed calibration; no retraining, corrected-head training, full refit, package, locked/private, or Codabench.",
    }
    (out_dir / "calibration_grid.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (out_dir / "calibration_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    best = {name: max((r for r in rows if r["model"] == name), key=lambda r: r["sps"]) for name in calibration_model_names(corrected_probe)}
    print(json.dumps({"status": "REVIEW_REQUIRED", "rows": len(rows), "windows": len(ds), "best": best, "static_reference": result["static_reference"]}, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("data-root", "kit-root", "checkpoint", "manifest", "validation-probe", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--corrected-probe", type=Path)
    args = parser.parse_args(); run(data_root=args.data_root, kit_root=args.kit_root, checkpoint=args.checkpoint, manifest=args.manifest, validation_probe=args.validation_probe, corrected_probe=args.corrected_probe, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
