#!/usr/bin/env python3
"""Build a clean held-out-AoA benchmark from the frozen 50/16 PIV split.

For target AoA=10 degrees:
- train = frozen Train50 excluding AoA 10
- inner_dev = frozen Dev16 excluding AoA 10
- heldout = every AoA-10 trajectory from frozen Train50 + Dev16

The locked-final split is never read or listed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def row_name(row) -> str:
    return str(row["file"] if isinstance(row, dict) else row)


def read_scalar(path: Path, key: str) -> float:
    with h5py.File(path, "r") as handle:
        candidates = (key, f"measured_data/{key}")
        for candidate in candidates:
            if candidate in handle:
                value = np.asarray(handle[candidate]).reshape(-1)
                if value.size < 1:
                    raise ValueError(f"empty metadata {key}: {path}")
                if not np.allclose(value, value[0], rtol=0.0, atol=1e-6):
                    raise ValueError(f"non-constant metadata {key}: {path}")
                return float(value[0])
    raise KeyError(f"{key} missing in {path}")


def build_split(
    real_root: Path,
    frozen_manifest: Path,
    *,
    heldout_aoa: float,
) -> dict[str, object]:
    payload = json.loads(frozen_manifest.read_text(encoding="utf-8"))
    frozen_train = [row_name(x) for x in payload.get("train", [])]
    frozen_dev = [row_name(x) for x in payload.get("dev", [])]
    if len(frozen_train) != 50 or len(frozen_dev) != 16:
        raise RuntimeError(
            f"expected frozen 50/16 manifest, got train={len(frozen_train)} dev={len(frozen_dev)}"
        )
    if set(frozen_train) & set(frozen_dev):
        raise RuntimeError("frozen train/dev overlap")

    def annotate(names: list[str]) -> list[dict[str, object]]:
        rows = []
        for name in names:
            path = real_root / name
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append({
                "file": name,
                "aoa": read_scalar(path, "aoa"),
                "re": read_scalar(path, "re"),
            })
        return rows

    train_rows = annotate(frozen_train)
    dev_rows = annotate(frozen_dev)

    def is_holdout(row: dict[str, object]) -> bool:
        return bool(np.isclose(float(row["aoa"]), heldout_aoa, rtol=0.0, atol=1e-6))

    train = [row for row in train_rows if not is_holdout(row)]
    inner_dev = [row for row in dev_rows if not is_holdout(row)]
    heldout = [row for row in [*train_rows, *dev_rows] if is_holdout(row)]

    all_names = {
        "train": {str(row["file"]) for row in train},
        "dev": {str(row["file"]) for row in inner_dev},
        "heldout": {str(row["file"]) for row in heldout},
    }
    if all_names["train"] & all_names["dev"]:
        raise RuntimeError("derived train/dev overlap")
    if all_names["heldout"] & (all_names["train"] | all_names["dev"]):
        raise RuntimeError("heldout AoA leaked into train/dev")
    if any(np.isclose(float(row["aoa"]), heldout_aoa) for row in [*train, *inner_dev]):
        raise RuntimeError("heldout AoA present in train or inner_dev")
    if not heldout:
        raise RuntimeError("no heldout trajectories found")

    return {
        "protocol": "heldout_aoa_from_frozen_50_16",
        "heldout_aoa": float(heldout_aoa),
        "source_manifest": str(frozen_manifest),
        "source_manifest_sha256": sha256(frozen_manifest),
        "train": train,
        "dev": inner_dev,
        "heldout": heldout,
        "counts": {
            "frozen_train": len(frozen_train),
            "frozen_dev": len(frozen_dev),
            "train": len(train),
            "inner_dev": len(inner_dev),
            "heldout": len(heldout),
        },
        "locked_final_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--heldout-aoa", type=float, default=10.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build_split(
        args.real_root,
        args.frozen_manifest,
        heldout_aoa=args.heldout_aoa,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
