#!/usr/bin/env python3
"""Build a trajectory-level Track 1 dataset profile from the frozen manifest.

The script reads HDF5 members directly from the released tar archive, so it
does not materialize the full archive.  All distribution/OOD calculations are
fit on Train descriptors only.  Future-window fields are explicitly
analysis-only and never participate in labels or input-side distances.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import h5py
import numpy as np


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
    required = {"train", "dev"}
    if not required <= obj.keys():
        raise ValueError(f"manifest missing split(s): {required - obj.keys()}")
    return obj


def member_map(tf: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    result = {}
    for m in tf.getmembers():
        if m.isfile() and m.name.endswith(".h5"):
            result[Path(m.name).name] = m
    return result


def field(handle: h5py.File, name: str) -> np.ndarray:
    key = name if name in handle else f"measured_data/{name}"
    return np.asarray(handle[key], dtype=np.float32)


def window_descriptor(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    speed = np.sqrt(u * u + v * v)
    du_t = np.diff(u, axis=0)
    dv_t = np.diff(v, axis=0)
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
    temporal = np.mean(np.abs(np.fft.rfft(u - u.mean(axis=0), axis=0)) ** 2 + np.abs(np.fft.rfft(v - v.mean(axis=0), axis=0)) ** 2, axis=(1, 2))
    temporal = temporal[1:]
    n = len(temporal)
    cuts = [max(1, n // 3), max(2, 2 * n // 3)]
    total = float(temporal.sum()) or 1.0
    spectrum = np.array([temporal[:cuts[0]].sum() / total, temporal[cuts[0]:cuts[1]].sum() / total, temporal[cuts[1]:].sum() / total])
    x = np.array([
        mean_u, u.std(), mean_v, v.std(), speed.mean(), speed.std(), np.percentile(speed, 95),
        delta.mean() if delta.size else 0.0, delta.std() if delta.size else 0.0, fluct,
        grad_mag.mean(), np.abs(vort).mean(), strain.mean(), high_ratio, *spectrum,
    ], dtype=np.float64)
    tke = float(np.mean(fu * fu + fv * fv) / 2.0)
    target = np.array([mean_u, mean_v, speed.mean(), fluct, tke], dtype=np.float64)
    return x, target


def trajectory_row(name: str, split: str, raw: bytes, in_steps: int, out_steps: int, stride: int) -> tuple[dict, list[float]]:
    with h5py.File(io.BytesIO(raw), "r") as h:
        u, v = field(h, "u"), field(h, "v")
        aoa, reynolds = float(h["aoa"][()]), float(h["re"][()])
        xgrid, ygrid = field(h, "x"), field(h, "y")
    if u.shape != v.shape or u.ndim != 3:
        raise ValueError(f"{name}: expected u/v [T,H,W], got {u.shape}/{v.shape}")
    dx = float(np.median(np.abs(np.diff(xgrid, axis=-1)))) if xgrid.shape[-1] > 1 else 1.0
    dy = float(np.median(np.abs(np.diff(ygrid, axis=-2)))) if ygrid.shape[-2] > 1 else 1.0
    starts = range(0, u.shape[0] - in_steps - out_steps + 1, stride)
    input_rows, target_rows = [], []
    for start in starts:
        a, b = window_descriptor(u[start:start + in_steps], v[start:start + in_steps], dx, dy)
        _, target = window_descriptor(u[start + in_steps:start + in_steps + out_steps], v[start + in_steps:start + in_steps + out_steps], dx, dy)
        input_rows.append(a); target_rows.append(target)
    if not input_rows:
        raise ValueError(f"{name}: no windows for protocol")
    a = np.asarray(input_rows); b = np.asarray(target_rows)
    values = {k: float(v) for k, v in zip(INPUT_NAMES, np.mean(a, axis=0))}
    values.update({k: float(v) for k, v in zip(TARGET_NAMES, np.mean(b, axis=0))})
    values.update({"split": split, "trajectory_id": Path(name).stem, "file": name, "aoa": aoa, "re": reynolds, "frames": int(u.shape[0]), "windows": int(len(a))})
    return values, np.mean(a, axis=0).tolist()


def percentile_dict(rows: list[dict], names: list[str]) -> dict:
    out = {}
    for name in names:
        a = np.asarray([r[name] for r in rows], dtype=float)
        out[name] = {k: float(v) for k, v in zip(["min", "p10", "p25", "median", "p75", "p90", "p95", "max"], np.percentile(a, [0, 10, 25, 50, 75, 90, 95, 100]))}
    return out


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
    wanted = {split: [Path(x["file"]).name if isinstance(x, dict) else Path(x).name for x in manifest[split]] for split in ("train", "dev")}
    all_rows, vectors = [], {}
    with tarfile.open(args.data_archive, "r:*") as tf:
        members = member_map(tf)
        for split in ("train", "dev"):
            for name in wanted[split]:
                if name not in members:
                    raise FileNotFoundError(f"{name} not found in {args.data_archive}")
                extracted = tf.extractfile(members[name])
                assert extracted is not None
                row, vector = trajectory_row(name, split, extracted.read(), args.in_steps, args.out_steps, args.stride)
                all_rows.append(row); vectors[name] = np.asarray(vector, dtype=float)
    train = [r for r in all_rows if r["split"] == "train"]
    dev = [r for r in all_rows if r["split"] == "dev"]
    center = np.asarray([vectors[r["file"]] for r in train])
    scale = np.std(center, axis=0, ddof=0); scale[scale == 0] = 1.0
    ztrain = (center - center.mean(axis=0)) / scale
    train_p95 = {n: float(np.percentile([r[n] for r in train], 95)) for n in INPUT_NAMES}
    train_min = {n: float(np.min([r[n] for r in train])) for n in INPUT_NAMES}
    train_max = {n: float(np.max([r[n] for r in train])) for n in INPUT_NAMES}
    for r in all_rows:
        z = (vectors[r["file"]] - center.mean(axis=0)) / scale
        dist = np.sqrt(np.sum((ztrain - z) ** 2, axis=1))
        r["nearest_train_distance"] = float(np.min(dist))
        out95 = [n for n in INPUT_NAMES if r[n] > train_p95[n] or r[n] < np.percentile([x[n] for x in train], 5)]
        r["train_p95_exceed_count"] = len(out95)
        if r["split"] == "train": r["distribution_label"] = "TRAIN_REFERENCE"
        elif len(out95) >= 3 or r["nearest_train_distance"] > 4.0: r["distribution_label"] = "OOD_LIKE"
        elif len(out95) >= 1 or r["nearest_train_distance"] > 2.0: r["distribution_label"] = "BOUNDARY"
        else: r["distribution_label"] = "IN_DISTRIBUTION"
        r["p95_exceed_descriptors"] = ";".join(out95)
    with (args.out_dir / "trajectory_descriptors.csv").open("w", newline="") as f:
        keys = list(all_rows[0].keys()); w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(all_rows)
    profile = {
        "manifest_sha256": sha256(args.manifest), "archive_sha256": sha256(args.data_archive),
        "splits": {"train": len(train), "dev": len(dev), "locked_final_accessed": False},
        "window_protocol": {"input_steps": args.in_steps, "future_steps": args.out_steps, "stride": args.stride, "valid_start": "0, stride, ... where start+input_steps+future_steps <= frames", "aggregation": "mean of non-overlapping trajectory windows"},
        "input_descriptor_names": INPUT_NAMES, "target_descriptor_names_analysis_only": TARGET_NAMES,
        "input_distribution": {"train": percentile_dict(train, INPUT_NAMES), "dev": percentile_dict(dev, INPUT_NAMES)},
        "coverage": {"label_rule": "Train-only: OOD_LIKE if >=3 input descriptors exceed Train p95 / fall below Train p5, or nearest standardized distance >4; BOUNDARY if >=1 descriptor exceeds those limits or distance >2; otherwise IN_DISTRIBUTION.", "train_only_standardization": True, "dev_labels": {r["trajectory_id"]: r["distribution_label"] for r in dev}},
        "analysis_only_target_note": "Future20 target descriptors are in trajectory_descriptors.csv for retrospective analysis only; they were excluded from labels, distances, and any model input.",
        "counts": {"train_windows": sum(r["windows"] for r in train), "dev_windows": sum(r["windows"] for r in dev)},
    }
    (args.out_dir / "profile_summary.json").write_text(json.dumps(profile, indent=2) + "\n")
    print(json.dumps({"out_dir": str(args.out_dir.resolve()), "train": len(train), "dev": len(dev), "train_windows": profile["counts"]["train_windows"], "dev_windows": profile["counts"]["dev_windows"]}, indent=2))


if __name__ == "__main__":
    main()
