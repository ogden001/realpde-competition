#!/usr/bin/env python3
"""Evaluate the preregistered corrector Gate on the frozen 16-trajectory dev split."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import realpde_loss_official_v9 as core
import realpde_b1_p0a_n2 as base_api
from realpde_adaptive_probe import ResidualCorrector3D, adaptive_features, evaluate_corrector_gate
from realpde_p0_data import H5WindowDataset, read_grid
from realpde_p0_features import P0FeatureBuilder, P0FeatureConfig


def evaluate(*, data_root: Path, kit_root: Path, checkpoint: Path, manifest: Path, probe: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    train = [data_root / row["file"] for row in payload["train"]]
    dev = [data_root / row["file"] for row in payload["dev"]]
    x_grid, y_grid = read_grid(train[0], sub_sample=2)
    config = P0FeatureConfig(True, False, float(x_grid[0, 1] - x_grid[0, 0]), float(y_grid[1, 0] - y_grid[0, 0]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    builder = P0FeatureBuilder(config).to(device)
    base = base_api.load_model(kit_root, checkpoint, builder, device).eval()
    saved = torch.load(probe, map_location="cpu", weights_only=False)
    corrector = ResidualCorrector3D(hidden=64, blocks=2).to(device)
    corrector.load_state_dict(saved["corrector_state_dict"], strict=True); corrector.eval()
    dataset = H5WindowDataset(dev, include_pressure=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    base_pred, candidate, targets = [], [], []
    with torch.no_grad():
        for x, y, _, _ in loader:
            x = x.to(device)
            frozen = base_api.forward(base, builder, x)
            delta = corrector(adaptive_features(x, frozen).permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
            base_pred.append(frozen.cpu().numpy()); candidate.append((frozen + delta).cpu().numpy()); targets.append(y.numpy())
    base_pred, candidate, targets = [np.concatenate(v).astype(np.float32) for v in (base_pred, candidate, targets)]
    import scoring
    def raw(pred: np.ndarray) -> dict[str, float]:
        channels = scoring.measured_channels(targets)
        return {"rel_l2": float(np.mean(scoring.rel_l2_per_sample(pred, targets, channels))), "tke": float(np.mean(scoring.tke_rel_l2_per_sample(pred, targets, channels))), "mvpe": float(scoring.mvpe_rel_l2(pred, targets))}
    baseline, cand = raw(base_pred), raw(candidate)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, ref in enumerate(dataset.refs): grouped[ref.path.name].append(index)
    trajectory_rows = []
    degradation = []
    for name, indices in sorted(grouped.items()):
        y, bp, cp = targets[indices], base_pred[indices], candidate[indices]
        y1, b1, c1 = y[None].reshape(1, -1, *y.shape[2:]), bp[None].reshape(1, -1, *bp.shape[2:]), cp[None].reshape(1, -1, *cp.shape[2:])
        channels = scoring.measured_channels(y1)
        bt = float(scoring.tke_rel_l2_per_sample(b1, y1, channels)[0]); ct = float(scoring.tke_rel_l2_per_sample(c1, y1, channels)[0]); d = ct / max(bt, 1e-12) - 1.0
        degradation.append(d); trajectory_rows.append({"trajectory_id": name, "base_tke": bt, "candidate_tke": ct, "tke_degradation": d})
    gate = evaluate_corrector_gate(baseline=baseline, candidate=cand, trajectory_tke_degradations=degradation)
    result = {"status": gate["status"], "baseline": baseline, "candidate": cand, "gate": gate, "windows": len(dataset), "trajectories": len(trajectory_rows), "execution_probe": str(probe)}
    (out_dir / "gate_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (out_dir / "trajectory_gate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectory_rows[0])); writer.writeheader(); writer.writerows(trajectory_rows)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("data-root", "kit-root", "checkpoint", "manifest", "probe", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    evaluate(data_root=args.data_root, kit_root=args.kit_root, checkpoint=args.checkpoint, manifest=args.manifest, probe=args.probe, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
