#!/usr/bin/env python3
"""GPU smoke test for the P0 feature path and an expanded CNO checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from realpde_p0_data import H5WindowDataset, read_grid
from realpde_p0_features import P0FeatureBuilder, P0FeatureConfig
from realpde_p0_finetune import adapt_input_weight, cno_forward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--example-h5", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--realpdebench-root", type=Path, required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this smoke test")
    sys.path.insert(0, str(args.realpdebench_root))
    from realpdebench.model.cno import CNO3d

    dataset = H5WindowDataset([args.example_h5], max_windows_per_trajectory=1)
    x, y, condition, _ = dataset[0]
    x, y, condition = x.unsqueeze(0).cuda(), y.unsqueeze(0).cuda(), condition.unsqueeze(0).cuda()
    grid_x, grid_y = read_grid(args.example_h5, sub_sample=2)
    builder = P0FeatureBuilder(
        P0FeatureConfig(dx=float(grid_x[0, 1] - grid_x[0, 0]), dy=float(grid_y[1, 0] - grid_y[0, 0]),
                        re_center=float(condition[0, 0]), re_scale=1.0), grid_x, grid_y,
    ).cuda()
    features = builder(x, {"re": condition[:, 0], "aoa": condition[:, 1]})
    model = CNO3d(in_dim=features.shape[-1], out_dim=3, out_dim_mult=1, in_size=64, N_layers=3).cuda()
    adapt_input_weight(model, torch.load(args.checkpoint, map_location="cpu"), features.shape[-1])
    model.eval()
    with torch.no_grad():
        output = cno_forward(model, features)
    if output.shape != y.shape:
        raise RuntimeError(f"output {tuple(output.shape)} != target {tuple(y.shape)}")
    print({"input": tuple(features.shape), "output": tuple(output.shape), "peak_bytes": torch.cuda.max_memory_allocated()})


if __name__ == "__main__":
    main()
