#!/usr/bin/env python3
"""Scan the interval family h = floor + mult_ch*sigma + rel*|p| on the clean split.

Uses the cached frozen val tensors plus a trained head checkpoint, so it runs in
seconds without the champion.  Prints the top configurations by official SPS.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/repo/tools")
import realpde_sps_scoring as S  # noqa: E402
from train_head_fast import Head3D, HeadConfig  # noqa: E402

CACHE = Path("/runs/frozen_cache")
SIG = S.SIGMA_GLOBAL


def fast_sps(pred, target, c, h, rewards, chunk=64):
    t = np.asarray(target[..., :c], dtype=np.float32)
    lo = np.asarray(pred[..., :c] - h, dtype=np.float32)
    hi = np.asarray(pred[..., :c] + h, dtype=np.float32)
    scored = t != 0.0
    n = int(np.count_nonzero(scored))
    total = {k: 0.0 for k in rewards}
    inside_n = 0
    for s in range(0, pred.shape[0], chunk):
        sl = slice(s, s + chunk)
        nil = (hi[sl] - lo[sl]) / SIG
        ins = (t[sl] >= lo[sl]) & (t[sl] <= hi[sl])
        inside_n += int(np.count_nonzero(ins & scored[sl]))
        pen = np.exp(-nil, out=nil) * ins
        pen *= scored[sl]
        for k, r in rewards.items():
            total[k] += float(np.sum(r[sl][:, None, None, None, None] * pen))
    sps = (0.5 * total["dm"] + 0.3 * total["tke"] + 0.2 * total["mvpe"]) / n
    return float(sps) * 100.0, inside_n / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=str, required=True)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--history-context", action="store_true")
    ap.add_argument("--tag", type=str, default="scan")
    args = ap.parse_args()

    xva = np.load(str(CACHE / "frozen_va") + "_x.npy", mmap_mode="r")
    bva = np.load(str(CACHE / "frozen_va") + "_b.npy", mmap_mode="r")
    pva = np.asarray(np.load(str(CACHE / "frozen_va") + "_p.npy", mmap_mode="r"), dtype=np.float32)
    tva = np.asarray(np.load(str(CACHE / "frozen_va") + "_y.npy", mmap_mode="r"), dtype=np.float32)
    pva[..., 2:] = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    head = Head3D(HeadConfig(hidden=args.hidden, blocks=args.blocks, history_context=args.history_context)).to(device)
    ck = torch.load(args.head, map_location=device)
    head.load_state_dict(ck["head_state_dict"] if "head_state_dict" in ck else ck)
    head.eval()
    sig = np.empty((xva.shape[0], 20, 32, 64, 2), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, xva.shape[0], 64):
            sl = np.arange(s, min(s + 64, xva.shape[0]))
            x = torch.from_numpy(np.asarray(xva[sl], dtype=np.float32)).to(device)
            b = torch.from_numpy(np.asarray(bva[sl], dtype=np.float32)).to(device)
            sig[sl] = torch.exp(head(x, b)).float().cpu().numpy()

    c = S.measured_channels(tva)
    dm = S.rel_l2_per_sample(pva, tva, c)
    tke = S.tke_rel_l2_per_sample(pva, tva, c)
    mvpe = S.mvpe_rel_l2_per_sample(pva, tva)
    rewards = {"dm": S._acc_factor(dm), "tke": S._acc_factor(tke), "mvpe": S._acc_factor(mvpe)}
    ap_ = np.abs(pva[..., :2]).astype(np.float32)

    rows = []
    for f in (0.0, 0.0025, 0.005):
        for mu in (0.5, 0.75, 1.0, 1.25, 1.5):
            for mv in (0.5, 0.75, 1.0, 1.25, 1.5):
                for rel in (0.0, 0.0025, 0.005, 0.0075):
                    h = np.stack([f + mu * sig[..., 0] + rel * ap_[..., 0], f + mv * sig[..., 1] + rel * ap_[..., 1]], -1)
                    sps, cov = fast_sps(pva, tva, c, h, rewards)
                    rows.append({"sps": round(sps, 4), "coverage": round(cov, 4), "floor": f, "mult_u": mu, "mult_v": mv, "rel": rel})
    rows.sort(key=lambda r: -r["sps"])
    out = {"head": args.head, "top": rows[:15], "n_scanned": len(rows)}
    print(json.dumps(out, indent=2), flush=True)
    Path(f"/runs/scan_bounds_{args.tag}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
