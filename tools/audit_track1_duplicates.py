#!/usr/bin/env python3
"""Audit cross-split duplicate Past20 inputs for frozen Track 1 data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import tarfile
from pathlib import Path

import h5py
import numpy as np

try:
    from tools.profile_track1_dataset import INPUT_NAMES, SPLITS, load_manifest, split_files
except ModuleNotFoundError:  # direct ``python tools/audit_track1_duplicates.py`` entrypoint
    from profile_track1_dataset import INPUT_NAMES, SPLITS, load_manifest, split_files


def window_starts(frames: int, in_steps: int, out_steps: int, stride: int):
    return range(0, frames - in_steps - out_steps + 1, stride)


def past_signature(raw: bytes, in_steps: int, out_steps: int, stride: int) -> tuple[str, int, tuple[int, ...]]:
    digest = hashlib.sha256()
    count = 0
    shape = None
    with h5py.File(io.BytesIO(raw), "r") as h:
        u, v = h["u"], h["v"]
        for start in window_starts(int(u.shape[0]), in_steps, out_steps, stride):
            # Only the Past20 input slice is read. No Future20 slice is accessed.
            u_window = np.asarray(u[start:start + in_steps], dtype=np.float32)
            v_window = np.asarray(v[start:start + in_steps], dtype=np.float32)
            if u_window.shape != v_window.shape:
                raise ValueError("u/v Past20 window shapes differ")
            shape = u_window.shape
            digest.update(u_window.tobytes(order="C"))
            digest.update(v_window.tobytes(order="C"))
            count += 1
    if shape is None:
        raise ValueError("trajectory has no valid windows")
    return digest.hexdigest(), count, shape


def past_difference(raw_a: bytes, raw_b: bytes, in_steps: int, out_steps: int, stride: int) -> tuple[float, float]:
    max_abs, total_abs, total_count = 0.0, 0.0, 0
    with h5py.File(io.BytesIO(raw_a), "r") as ha, h5py.File(io.BytesIO(raw_b), "r") as hb:
        ua, va, ub, vb = ha["u"], ha["v"], hb["u"], hb["v"]
        starts_a = list(window_starts(int(ua.shape[0]), in_steps, out_steps, stride))
        starts_b = list(window_starts(int(ub.shape[0]), in_steps, out_steps, stride))
        if len(starts_a) != len(starts_b):
            return float("inf"), float("inf")
        for sa, sb in zip(starts_a, starts_b):
            for da, db in ((ua[sa:sa + in_steps], ub[sb:sb + in_steps]), (va[sa:sa + in_steps], vb[sb:sb + in_steps])):
                diff = np.abs(np.asarray(da, dtype=np.float32) - np.asarray(db, dtype=np.float32))
                max_abs = max(max_abs, float(diff.max()))
                total_abs += float(diff.sum())
                total_count += diff.size
    return max_abs, total_abs / total_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--near-threshold", type=float, default=0.1)
    parser.add_argument("--in-steps", type=int, default=20)
    parser.add_argument("--out-steps", type=int, default=20)
    parser.add_argument("--stride", type=int, default=20)
    args = parser.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest); wanted = split_files(manifest)
    rows = list(csv.DictReader(args.descriptors.open()))
    row_by_file = {r["file"]: r for r in rows}
    if set(row_by_file) != {name for split in SPLITS for name in wanted[split]}:
        raise ValueError("descriptor CSV does not match manifest files")
    matrix = np.asarray([[float(row_by_file[name][key]) for key in INPUT_NAMES] for split in SPLITS for name in wanted[split]])
    scale = matrix.std(axis=0); scale[scale == 0] = 1.0
    z = (matrix - matrix.mean(axis=0)) / scale
    entries = [(split, name) for split in SPLITS for name in wanted[split]]
    distances = []
    for i, j in itertools.combinations(range(len(entries)), 2):
        if entries[i][0] == entries[j][0]:
            continue
        distances.append({"split_a": entries[i][0], "file_a": entries[i][1], "split_b": entries[j][0], "file_b": entries[j][1], "descriptor_distance": float(np.linalg.norm(z[i] - z[j]))})
    candidates = [r for r in distances if r["descriptor_distance"] <= args.near_threshold]
    raw_by_file = {}
    signatures = {}
    with tarfile.open(args.data_archive, "r:*") as tf:
        members = {Path(m.name).name: m for m in tf.getmembers() if m.isfile() and m.name.endswith(".h5")}
        for _, name in entries:
            extracted = tf.extractfile(members[name]); assert extracted is not None
            raw = extracted.read(); raw_by_file[name] = raw
            signatures[name] = past_signature(raw, args.in_steps, args.out_steps, args.stride)
    exact = []
    for r in distances:
        a, b = r["file_a"], r["file_b"]
        if signatures[a][:2] == signatures[b][:2] and signatures[a][0] == signatures[b][0]:
            max_abs, mean_abs = past_difference(raw_by_file[a], raw_by_file[b], args.in_steps, args.out_steps, args.stride)
            exact.append({**r, "kind": "EXACT_DUPLICATE", "past20_max_abs_diff": max_abs, "past20_mean_abs_diff": mean_abs})
    near = []
    for r in candidates:
        a, b = r["file_a"], r["file_b"]
        max_abs, mean_abs = past_difference(raw_by_file[a], raw_by_file[b], args.in_steps, args.out_steps, args.stride)
        near.append({**r, "kind": "EXACT_DUPLICATE" if max_abs == 0.0 and mean_abs == 0.0 else "NEAR_DUPLICATE_CANDIDATE", "past20_max_abs_diff": max_abs, "past20_mean_abs_diff": mean_abs})
    fields = ["kind", "split_a", "file_a", "split_b", "file_b", "descriptor_distance", "past20_max_abs_diff", "past20_mean_abs_diff"]
    with (args.out_dir / "duplicate_pairs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(near)
    nearest = {f"{a}_to_{b}": sorted([r["descriptor_distance"] for r in distances if r["split_a"] == a and r["split_b"] == b or r["split_a"] == b and r["split_b"] == a])[:3] for a, b in itertools.combinations(SPLITS, 2)}
    summary = {"conclusion": "DUPLICATE_AUDIT_CLEAN" if len(exact) == 1 else "MULTIPLE_CROSS_SPLIT_DUPLICATES", "exact_duplicates": exact, "near_threshold": args.near_threshold, "near_candidates": near, "nearest_descriptor_distances": nearest, "past20_only": True, "future20_read": False, "manifest_modified": False}
    (args.out_dir / "duplicate_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
