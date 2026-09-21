#!/usr/bin/env python3
"""Evaluate a residual-corrector checkpoint on the fixed 640-window clean protocol.

Uses the same 65/16 split (seed 41), stride 20 / subsample 2 validation windows as
every other measurement in this project, so numbers are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/repo/tools")
import realpde_sps_scoring as S  # noqa: E402
from realpde_h5_feature_adapter_train import (  # noqa: E402
    BAD_TRAIN_FILES,
    H5WindowDataset,
    list_h5,
    split_paths,
)
from realpde_h5_residual_corrector_eval import load_model  # noqa: E402


@torch.no_grad()
def gather(ckpt: Path, device):
    paths = list_h5(Path("/data/p0ab_real_h5_20260830"), BAD_TRAIN_FILES)
    train_paths, val_paths = split_paths(paths, 0.2, 41)
    ds = H5WindowDataset(val_paths, in_steps=20, out_steps=20, stride=20, sub_sample=2,
                         max_windows_per_trajectory=None, include_pressure=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    model = load_model(ckpt, Path("/third_party"), device)
    model.eval()
    preds, targs = [], []
    for x, y in loader:
        out = model(x.to(device), alpha=1.0)
        out = out.clone()
        out[..., 2] = 0.0
        preds.append(out.float().cpu().numpy().astype(np.float32))
        targs.append(y.numpy().astype(np.float32))
    return np.concatenate(preds), np.concatenate(targs), len(ds), len(train_paths), len(val_paths)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--tag", default="corrector")
    args = ap.parse_args()
    device = torch.device("cuda:0")
    rows = []
    for ck in args.checkpoints:
        pred, tgt, n, ntr, nva = gather(Path(ck), device)
        c = S.measured_channels(tgt)
        rel = float(np.mean(S.rel_l2_per_sample(pred, tgt, c)))
        tke = float(np.mean(S.tke_rel_l2_per_sample(pred, tgt, c)))
        mvpe = float(np.mean(S.mvpe_rel_l2_per_sample(pred, tgt)))
        lo, hi = S.constant_bounds(pred, 0.005, 0.01)
        sps_fixed, cov_f = S.aggregate_sps(pred, tgt, c, lower=lo, upper=hi)
        lo, hi = S.oracle_bounds(pred, tgt)
        sps_or, _ = S.aggregate_sps(pred, tgt, c, lower=lo, upper=hi)
        rows.append({
            "checkpoint": ck,
            "n_windows": n, "train_traj": ntr, "val_traj": nva,
            "rel_l2": round(rel, 6), "rel_l2_score": round(S.score_error(rel), 4),
            "tke": round(tke, 6), "tke_score": round(S.score_error(tke), 4),
            "mvpe": round(mvpe, 6), "mvpe_score": round(S.score_error(mvpe), 4),
            "fixed_sps": round(100.0 * sps_fixed, 4), "fixed_coverage": round(cov_f, 4),
            "oracle_sps": round(100.0 * sps_or, 4),
        })
        print(json.dumps(rows[-1], indent=2), flush=True)
    Path(f"/runs/eval_corrector_{args.tag}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
