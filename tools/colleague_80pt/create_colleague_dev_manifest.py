#!/usr/bin/env python3
"""Materialize the colleague's historical all81/Dev16 trajectory split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from realpde_h5_feature_adapter_train import BAD_TRAIN_FILES, list_h5, split_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {args.out}")
    all_paths = list_h5(args.real_root, BAD_TRAIN_FILES)
    _, dev_paths = split_paths(all_paths, args.val_fraction, args.seed)
    payload = {
        "protocol": "colleague_dev16_seed41_all81",
        "source_order": "sorted H5 filename after excluding BAD_TRAIN_FILES",
        "excluded_files": sorted(BAD_TRAIN_FILES),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "train": [path.name for path in all_paths],
        "dev": [path.name for path in dev_paths],
        "train_dev_overlap": True,
        "note": "Historical competition-oriented all81 train / Dev16 screening split; Dev16 targets are in train.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"train": len(all_paths), "dev": len(dev_paths), "out": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
