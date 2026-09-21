#!/usr/bin/env python3
"""Cache (input window, frozen champion base prediction, target) to .npy memmaps.

The uncertainty head only ever needs (x, base_pred) plus the target; the champion
is frozen.  Materialising those tensors once makes head training ~10x cheaper
(no CNO forward/backward per step) so we can actually sweep head size, loss and
training length instead of running a single 2000-step configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from realpde_h5_feature_adapter_train import (  # noqa: E402
    BAD_TRAIN_FILES,
    H5WindowDataset,
    list_h5,
    paths_from_split_manifest,
    split_paths,
)
from residual_multi import load_full_residual_model  # noqa: E402


def windows(paths, stride):
    return H5WindowDataset(
        paths,
        in_steps=20,
        out_steps=20,
        stride=stride,
        sub_sample=2,
        max_windows_per_trajectory=None,
        include_pressure=False,
    )


@torch.no_grad()
def dump(model, ds, device, prefix: Path, tag: str):
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    shape = (len(ds), 20, 32, 64, 2)
    xs = np.lib.format.open_memmap(prefix.with_name(prefix.name + f"_{tag}_x.npy"), mode="w+", dtype=np.float16, shape=shape)
    bs = np.lib.format.open_memmap(prefix.with_name(prefix.name + f"_{tag}_b.npy"), mode="w+", dtype=np.float16, shape=shape)
    ps = np.lib.format.open_memmap(prefix.with_name(prefix.name + f"_{tag}_p.npy"), mode="w+", dtype=np.float16, shape=shape)
    ys = np.lib.format.open_memmap(prefix.with_name(prefix.name + f"_{tag}_y.npy"), mode="w+", dtype=np.float16, shape=shape)
    i = 0
    for j, (x, y) in enumerate(loader):
        xd = x.to(device)
        base = model.base_predict(xd)
        delta = model.predict_delta(xd, base)
        pred = model.combine(base, delta, 1.0)
        n = int(x.shape[0])
        xs[i : i + n] = x[..., :2].numpy().astype(np.float16)
        bs[i : i + n] = base[..., :2].float().cpu().numpy().astype(np.float16)
        ps[i : i + n] = pred[..., :2].float().cpu().numpy().astype(np.float16)
        ys[i : i + n] = y[..., :2].numpy().astype(np.float16)
        i += n
        if (j + 1) % 20 == 0:
            print(f"[cache] {tag} {i}/{len(ds)}", flush=True)
    xs.flush(); bs.flush(); ps.flush(); ys.flush()
    print(f"[cache] {tag} wrote {i} windows -> {prefix}", flush=True)
    return i


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-root", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, default=None)
    ap.add_argument("--allow-train-dev-overlap", action="store_true")
    ap.add_argument("--realpdebench-root", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--champion", type=Path, required=True)
    ap.add_argument("--all-data", action="store_true", help="use every trajectory as train (production head)")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.split_manifest is not None:
        tr, va = paths_from_split_manifest(
            args.real_root,
            args.split_manifest,
            allow_train_dev_overlap=args.allow_train_dev_overlap,
        )
        paths = tr
    else:
        paths = list_h5(args.real_root, BAD_TRAIN_FILES)
        tr, va = split_paths(paths, args.val_fraction, args.seed)
    if args.all_data:
        tr = list(paths)
    model, metadata = load_full_residual_model(
        args.champion,
        args.realpdebench_root,
        device,
    )
    n_tr = dump(model, windows(tr, args.stride), device, out / "frozen", "tr")
    n_va = dump(model, windows(va, 20), device, out / "frozen", "va")
    (out / "meta.json").write_text(
        json.dumps(
            {
                "champion": str(args.champion),
                "checkpoint_iteration": metadata.get("iteration"),
                "checkpoint_best_score": metadata.get("best_score"),
                "real_root": str(args.real_root),
                "split_manifest": str(args.split_manifest) if args.split_manifest else None,
                "allow_train_dev_overlap": bool(args.allow_train_dev_overlap),
                "realpdebench_root": str(args.realpdebench_root),
                "all_data": bool(args.all_data),
                "stride_train": args.stride,
                "stride_val": 20,
                "val_fraction": args.val_fraction,
                "seed": args.seed,
                "n_train": n_tr,
                "n_val": n_va,
                "dtype": "float16",
                "channels": 2,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "DONE").touch()
    print("[cache] DONE", flush=True)


if __name__ == "__main__":
    main()
