#!/usr/bin/env python3
"""Run the frozen 4x7 adaptive-SPS calibration grid on validation dev only."""
from __future__ import annotations

import argparse
import csv
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


def run(*, data_root: Path, kit_root: Path, checkpoint: Path, manifest: Path, validation_probe: Path, corrected_probe: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(manifest.read_text(encoding="utf-8")); root = data_root
    train = [root / row["file"] for row in spec["train"]]; dev = [root / row["file"] for row in spec["dev"]]
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = feature_config_from_checkpoint(checkpoint_payload)
    assert_feature_config_matches_checkpoint(cfg, checkpoint_payload)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); builder = P0FeatureBuilder(cfg).to(device)
    base = base_api.load_model(kit_root, checkpoint, builder, device).eval()
    vp = torch.load(validation_probe, map_location="cpu", weights_only=False); cp = torch.load(corrected_probe, map_location="cpu", weights_only=False)
    corrector = ResidualCorrector3D(hidden=64, blocks=2).to(device); corrector.load_state_dict(cp["corrector_state_dict"] or vp["corrector_state_dict"]); corrector.eval()
    base_head = AdaptiveUncertaintyHead(hidden=32, blocks=2).to(device); base_head.load_state_dict(vp["base_head_state_dict"]); base_head.eval()
    corrected_head = AdaptiveUncertaintyHead(hidden=32, blocks=2).to(device); corrected_head.load_state_dict(cp["corrected_head_state_dict"]); corrected_head.eval()
    ds = H5WindowDataset(dev, include_pressure=True); loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    base_pred, final_pred, base_sigma, corrected_sigma, targets = [], [], [], [], []
    with torch.no_grad():
        for x, y, _, _ in loader:
            x = x.to(device); bp = base_api.forward(base, builder, x); d = corrector(adaptive_features(x, bp).permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
            features = torch.cat([x, flow_features(bp[..., :2])], dim=-1).permute(0, 4, 1, 2, 3)
            base_pred.append(bp.cpu().numpy()); final_pred.append((bp + d).cpu().numpy()); base_sigma.append(base_head(features).permute(0, 2, 3, 4, 1).cpu().numpy()); corrected_sigma.append(corrected_head(features).permute(0, 2, 3, 4, 1).cpu().numpy()); targets.append(y.numpy())
    base_pred, final_pred, base_sigma, corrected_sigma, targets = [np.concatenate(v).astype(np.float32) for v in (base_pred, final_pred, base_sigma, corrected_sigma, targets)]
    sys.path.insert(0, str(kit_root.resolve())); import scoring
    rows = []
    for model_name, pred, sigma in (("base", base_pred, base_sigma), ("corrected", final_pred, corrected_sigma)):
        c = scoring.measured_channels(targets)
        raw = {"rel_l2": float(np.mean(scoring.rel_l2_per_sample(pred, targets, c))), "tke": float(np.mean(scoring.tke_rel_l2_per_sample(pred, targets, c))), "mvpe": float(scoring.mvpe_rel_l2(pred, targets))}
        for floor, mult in fixed_calibration_grid():
            uv_half = (floor + mult * sigma).astype(np.float32)
            half = np.concatenate([uv_half, np.zeros(uv_half.shape[:-1] + (1,), dtype=np.float32)], axis=-1)
            lo, hi = pred - half, pred + half
            sps, coverage = scoring.aggregate_sps(pred, targets, c, lo, hi)
            rows.append({"model": model_name, "floor": floor, "mult": mult, "sps": 100.0 * float(sps), "coverage": float(coverage), "mean_width_uv": float(np.mean(2.0 * half[..., :2])), **raw})
    rows.sort(key=lambda row: (row["model"], -row["sps"]))
    (out_dir / "calibration_grid.json").write_text(json.dumps({"rows": rows, "grid_size": len(rows), "windows": len(ds), "trajectories": len(dev)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (out_dir / "calibration_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"status": "CALIBRATION_COMPLETE", "rows": len(rows), "windows": len(ds), "best": {name: max((r for r in rows if r["model"] == name), key=lambda r: r["sps"]) for name in ("base", "corrected")}}, indent=2, sort_keys=True))
    return {"rows": rows, "windows": len(ds), "trajectories": len(dev)}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("data-root", "kit-root", "checkpoint", "manifest", "validation-probe", "corrected-probe", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(); run(data_root=args.data_root, kit_root=args.kit_root, checkpoint=args.checkpoint, manifest=args.manifest, validation_probe=args.validation_probe, corrected_probe=args.corrected_probe, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
