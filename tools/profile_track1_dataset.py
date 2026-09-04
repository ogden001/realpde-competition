#!/usr/bin/env python3
"""Build a target-blind, trajectory-level Track 1 split audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

SPLITS = ("train", "dev", "final")
INPUT_NAMES = [
    "u_mean", "u_std", "v_mean", "v_std", "speed_mean", "speed_std",
    "speed_p95", "delta_mean", "delta_std", "fluctuation_rms",
    "grad_mag_mean", "vorticity_abs_mean", "strain_mag_mean",
    "high_energy_area_ratio", "spectrum_low_ratio", "spectrum_mid_ratio",
    "spectrum_high_ratio",
]
TARGET_NAMES = ["future_u_mean", "future_v_mean", "future_speed_mean", "future_fluctuation_rms", "future_tke"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(path: Path) -> dict:
    obj = json.loads(path.read_text())
    required = set(SPLITS)
    if not required <= obj.keys():
        raise ValueError(f"manifest missing split(s): {required - obj.keys()}")
    names = [x["file"] if isinstance(x, dict) else x for split in SPLITS for x in obj[split]]
    if len(set(names)) != len(names):
        raise ValueError("manifest split entries must be unique")
    return obj


def split_files(manifest: dict) -> dict[str, list[str]]:
    return {split: [Path(x["file"] if isinstance(x, dict) else x).name for x in manifest[split]] for split in SPLITS}


def member_map(tf: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    return {Path(m.name).name: m for m in tf.getmembers() if m.isfile() and m.name.endswith(".h5")}


def field(handle: h5py.File, name: str) -> np.ndarray:
    key = name if name in handle else f"measured_data/{name}"
    return np.asarray(handle[key], dtype=np.float32)


def window_descriptor(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    speed = np.sqrt(u * u + v * v)
    du_t, dv_t = np.diff(u, axis=0), np.diff(v, axis=0)
    delta = np.sqrt(du_t * du_t + dv_t * dv_t)
    mean_u, mean_v = u.mean(), v.mean()
    fu, fv = u - mean_u, v - mean_v
    fluct = np.sqrt(np.mean(fu * fu + fv * fv) / 2.0)
    du_dy, du_dx = np.gradient(u, dy, dx, axis=(-2, -1))
    dv_dy, dv_dx = np.gradient(v, dy, dx, axis=(-2, -1))
    grad_mag = np.sqrt(du_dx * du_dx + du_dy * du_dy + dv_dx * dv_dx + dv_dy * dv_dy)
    vort = dv_dx - du_dy
    strain = np.sqrt((du_dx - dv_dy) ** 2 + (du_dy + dv_dx) ** 2)
    local_energy = np.mean((u - u.mean(axis=0, keepdims=True)) ** 2 + (v - v.mean(axis=0, keepdims=True)) ** 2, axis=0)
    high_ratio = float(np.mean(local_energy > 1.5 * np.median(local_energy)))
    temporal = np.mean(np.abs(np.fft.rfft(u - u.mean(axis=0), axis=0)) ** 2 + np.abs(np.fft.rfft(v - v.mean(axis=0), axis=0)) ** 2, axis=(1, 2))[1:]
    n = len(temporal)
    cuts = [max(1, n // 3), max(2, 2 * n // 3)]
    total = float(temporal.sum()) or 1.0
    spectrum = np.array([temporal[:cuts[0]].sum() / total, temporal[cuts[0]:cuts[1]].sum() / total, temporal[cuts[1]:].sum() / total])
    return np.array([
        mean_u, u.std(), mean_v, v.std(), speed.mean(), speed.std(), np.percentile(speed, 95),
        delta.mean() if delta.size else 0.0, delta.std() if delta.size else 0.0, fluct,
        grad_mag.mean(), np.abs(vort).mean(), strain.mean(), high_ratio, *spectrum,
    ], dtype=np.float64)


def trajectory_row(name: str, split: str, raw: bytes, in_steps: int, out_steps: int, stride: int) -> tuple[dict, list[float]]:
    """Return metadata and Past20 descriptors; never compute Future20 fields."""
    with h5py.File(io.BytesIO(raw), "r") as h:
        u_ds = h["u"] if "u" in h else h["measured_data/u"]
        v_ds = h["v"] if "v" in h else h["measured_data/v"]
        if u_ds.shape != v_ds.shape or len(u_ds.shape) != 3:
            raise ValueError(f"{name}: expected u/v [T,H,W], got {u_ds.shape}/{v_ds.shape}")
        xgrid, ygrid = field(h, "x"), field(h, "y")
        aoa, reynolds, frames = float(h["aoa"][()]), float(h["re"][()]), int(u_ds.shape[0])
        dx = float(np.median(np.abs(np.diff(xgrid, axis=-1)))) if xgrid.shape[-1] > 1 else 1.0
        dy = float(np.median(np.abs(np.diff(ygrid, axis=-2)))) if ygrid.shape[-2] > 1 else 1.0
        starts = range(0, frames - in_steps - out_steps + 1, stride)
        input_rows = [window_descriptor(u_ds[start:start + in_steps], v_ds[start:start + in_steps], dx, dy) for start in starts]
    if not input_rows:
        raise ValueError(f"{name}: no windows for protocol")
    vector = np.mean(np.asarray(input_rows), axis=0)
    row = {key: float(value) for key, value in zip(INPUT_NAMES, vector)}
    row.update({"split": split, "trajectory_id": Path(name).stem, "file": name, "aoa": aoa, "re": reynolds, "frames": frames, "windows": len(input_rows)})
    return row, vector.tolist()


def percentile_dict(rows: list[dict], names: list[str]) -> dict:
    keys = ["min", "p10", "p25", "median", "p75", "p90", "p95", "max"]
    return {name: {key: float(value) for key, value in zip(keys, np.percentile([r[name] for r in rows], [0, 10, 25, 50, 75, 90, 95, 100]))} for name in names}


def write_distribution_csv(path: Path, rows_by_split: dict[str, list[dict]]) -> None:
    fields = ["split", "variable", "count", "min", "p10", "median", "p90", "p95", "max"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for split in SPLITS:
            rows = rows_by_split[split]
            for variable in ("re", "aoa", "frames", "windows", *INPUT_NAMES):
                stats = percentile_dict(rows, [variable])[variable]
                writer.writerow({"split": split, "variable": variable, "count": len(rows), **{key: stats[key] for key in ("min", "p10", "median", "p90", "p95", "max")}})


def pca_2d(vectors: np.ndarray) -> np.ndarray:
    center, scale = vectors.mean(axis=0), vectors.std(axis=0)
    scale[scale == 0] = 1.0
    z = (vectors - center) / scale
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    return z @ vt[:2].T


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-archive", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--in-steps", type=int, default=20)
    p.add_argument("--out-steps", type=int, default=20)
    p.add_argument("--stride", type=int, default=20)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    wanted = split_files(manifest)
    all_rows, vectors = [], {}
    with tarfile.open(args.data_archive, "r:*") as tf:
        members = member_map(tf)
        for split in SPLITS:
            for name in wanted[split]:
                if name not in members:
                    raise FileNotFoundError(f"{name} not found in {args.data_archive}")
                extracted = tf.extractfile(members[name])
                assert extracted is not None
                row, vector = trajectory_row(name, split, extracted.read(), args.in_steps, args.out_steps, args.stride)
                all_rows.append(row)
                vectors[name] = np.asarray(vector, dtype=float)
    rows_by_split = {split: [r for r in all_rows if r["split"] == split] for split in SPLITS}
    train = rows_by_split["train"]
    train_vectors = np.asarray([vectors[r["file"]] for r in train])
    train_mean, train_scale = train_vectors.mean(axis=0), train_vectors.std(axis=0)
    train_scale[train_scale == 0] = 1.0
    ztrain = (train_vectors - train_mean) / train_scale
    train_p5 = {n: float(np.percentile([r[n] for r in train], 5)) for n in INPUT_NAMES}
    train_p95 = {n: float(np.percentile([r[n] for r in train], 95)) for n in INPUT_NAMES}
    for row in all_rows:
        z = (vectors[row["file"]] - train_mean) / train_scale
        distance = np.sqrt(np.sum((ztrain - z) ** 2, axis=1))
        exceed = [n for n in INPUT_NAMES if row[n] < train_p5[n] or row[n] > train_p95[n]]
        row.update({"nearest_train_distance": float(np.min(distance)), "train_p5_p95_exceed_count": len(exceed), "p95_exceed_descriptors": ";".join(exceed)})
        if row["split"] == "train":
            label = "TRAIN_REFERENCE"
        elif len(exceed) >= 3 or row["nearest_train_distance"] > 4.0:
            label = "OOD_LIKE"
        elif exceed or row["nearest_train_distance"] > 2.0:
            label = "BOUNDARY"
        else:
            label = "IN_DISTRIBUTION"
        row["distribution_label"] = label
    pca = pca_2d(np.asarray([vectors[r["file"]] for r in all_rows]))
    for row, coords in zip(all_rows, pca):
        row["pca1"], row["pca2"] = float(coords[0]), float(coords[1])
    with (args.out_dir / "trajectory_descriptors.csv").open("w", newline="") as f:
        keys = list(all_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_rows)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"train": "#4c78a8", "dev": "#f58518", "final": "#54a24b"}
    for split in SPLITS:
        idx = [i for i, row in enumerate(all_rows) if row["split"] == split]
        ax.scatter(pca[idx, 0], pca[idx, 1], label=f"{split.title()} (n={len(idx)})", c=colors[split], alpha=0.85, edgecolors="white", linewidths=0.5)
    ax.set(xlabel="PC1 (all-82 standardized inputs)", ylabel="PC2 (all-82 standardized inputs)", title="Track 1 split audit: input descriptor PCA")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.out_dir / "pca_split.png", dpi=160)
    plt.close(fig)
    write_distribution_csv(args.out_dir / "split_distribution_summary.csv", rows_by_split)
    summary = {
        "manifest_sha256": sha256(args.manifest),
        "archive_sha256": sha256(args.data_archive),
        "splits": {split: len(rows_by_split[split]) for split in SPLITS},
        "window_counts": {split: int(sum(r["windows"] for r in rows_by_split[split])) for split in SPLITS},
        "basic_distribution": {split: percentile_dict(rows_by_split[split], ["re", "aoa", "frames", "windows"]) for split in SPLITS},
        "input_descriptor_names": INPUT_NAMES,
        "target_descriptor_names": [],
        "window_protocol": {"input_steps": args.in_steps, "future_steps": args.out_steps, "stride": args.stride, "valid_start": "0, stride, ... where start+input_steps+future_steps <= frames", "aggregation": "mean of non-overlapping trajectory input windows"},
        "final_target_blind": {"enabled": True, "future_descriptors_generated": False, "model_or_metric_fields_generated": False, "read_policy": "only Past20 u/v slices plus x/y and aoa/re metadata; Future20 is never sliced"},
        "coverage": {"standardization": "Train-only mean/std", "exceedance_interval": "Train p5/p95 per input descriptor", "label_rule": "OOD_LIKE if >=3 exceedances or nearest standardized distance >4; BOUNDARY if >=1 exceedance or distance >2; otherwise IN_DISTRIBUTION", "by_split": {split: {label: sum(r["distribution_label"] == label for r in rows_by_split[split]) for label in ("TRAIN_REFERENCE", "IN_DISTRIBUTION", "BOUNDARY", "OOD_LIKE")} for split in SPLITS}},
        "pca": {"standardization": "all 82 trajectory input vectors", "components": 2, "plot": "pca_split.png"},
        "audit_conclusion": "REVIEW_REQUIRED",
        "audit_note": "Distribution-only artifact: no model execution, Rel-L2/TKE/MVPE calculation, or manifest modification.",
    }
    (args.out_dir / "profile_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"out_dir": str(args.out_dir.resolve()), "splits": summary["splits"], "window_counts": summary["window_counts"]}, indent=2))


if __name__ == "__main__":
    main()
