#!/usr/bin/env python3
"""Measure P0-A/B CNO training-memory and throughput for one batch size."""

from __future__ import annotations

import argparse
import sys
import time
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
    ap.add_argument("--batch-size", type=int, required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available(): raise SystemExit("CUDA required")
    sys.path.insert(0, str(args.realpdebench_root))
    from realpdebench.model.cno import CNO3d
    dataset = H5WindowDataset([args.example_h5], max_windows_per_trajectory=1)
    x, y, condition, _ = dataset[0]
    x, y = x.unsqueeze(0).repeat(args.batch_size, 1, 1, 1, 1).cuda(), y.unsqueeze(0).repeat(args.batch_size, 1, 1, 1, 1).cuda()
    condition = condition.unsqueeze(0).repeat(args.batch_size, 1).cuda()
    gx, gy = read_grid(args.example_h5, 2)
    builder = P0FeatureBuilder(P0FeatureConfig(dx=float(gx[0,1]-gx[0,0]), dy=float(gy[1,0]-gy[0,0]), re_center=float(condition[0,0]), re_scale=1.), gx, gy).cuda()
    model = CNO3d(in_dim=25, out_dim=3, out_dim_mult=1, in_size=64, N_layers=3).cuda()
    adapt_input_weight(model, torch.load(args.checkpoint, map_location="cpu"), 25)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    torch.cuda.reset_peak_memory_stats()
    # One warmup, then one timed full training update.
    for timed in (False, True):
        optimizer.zero_grad(set_to_none=True)
        f = builder(x, {"re": condition[:,0], "aoa": condition[:,1]})
        if timed: torch.cuda.synchronize(); start=time.perf_counter()
        pred = cno_forward(model, f); loss=torch.mean((pred[...,:2]-y[...,:2]).square()); loss.backward(); optimizer.step()
        if timed:
            torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    print({"batch_size":args.batch_size,"peak_gib":round(torch.cuda.max_memory_allocated()/2**30,3),"step_s":round(elapsed,4),"samples_per_s":round(args.batch_size/elapsed,2)})

if __name__ == "__main__": main()
