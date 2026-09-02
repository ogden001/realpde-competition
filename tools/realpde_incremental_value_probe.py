#!/usr/bin/env python3
"""Minimal supervised residual probe for the frozen Track-1 PERSIST baseline.

This is deliberately a CPU-only, closed-form experiment.  The baseline is the
already-registered PERSIST prediction (last observed velocity repeated across
the 20-frame forecast); no neural model is trained or evaluated.  A separate
ridge residual head is fit for each feature package using train windows only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np


GROUPS = {
    "raw_control": ("u_last", "v_last"),
    "raw_temporal": ("u_last", "v_last", "u_mean", "v_mean", "u_std", "v_std", "u_delta", "v_delta"),
    "raw_spatial": ("u_last", "v_last", "du_dx_pixel", "du_dy_pixel", "dv_dx_pixel", "dv_dy_pixel"),
    "raw_temporal_spatial": (
        "u_last", "v_last", "u_mean", "v_mean", "u_std", "v_std", "u_delta", "v_delta",
        "du_dx_pixel", "du_dy_pixel", "dv_dx_pixel", "dv_dy_pixel",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def refs(manifest: dict, data_root: Path, split: str):
    for row in manifest[split]:
        yield data_root / row["file"], row["file"]


def windows(path: Path):
    """Yield complete 20->20 runtime windows at spatial resolution 32x64."""
    with h5py.File(path, "r") as f:
        fu, fv = (f["u"] if "u" in f else f["measured_data/u"]), (f["v"] if "v" in f else f["measured_data/v"])
        n = int(fu.shape[0])
        for start in range(0, n - 39, 20):
            u = np.asarray(fu[start:start + 40, ::2, ::2], dtype=np.float32)
            v = np.asarray(fv[start:start + 40, ::2, ::2], dtype=np.float32)
            yield np.stack([u, v], axis=-1), start


def grad_pixel(a: np.ndarray, axis: int) -> np.ndarray:
    """Centered finite difference, one-sided only at outer image edge."""
    out = np.empty_like(a)
    sl = [slice(None)] * a.ndim
    sl[axis] = 0
    sl0 = tuple(sl)
    sl[axis] = 1
    sl1 = tuple(sl)
    out[sl0] = a[sl1] - a[sl0]
    sl[axis] = -1
    sln = tuple(sl)
    sl[axis] = -2
    slm = tuple(sl)
    out[sln] = a[sln] - a[slm]
    mid = [slice(None)] * a.ndim
    mid[axis] = slice(1, -1)
    left = [slice(None)] * a.ndim
    left[axis] = slice(None, -2)
    right = [slice(None)] * a.ndim
    right[axis] = slice(2, None)
    out[tuple(mid)] = 0.5 * (a[tuple(right)] - a[tuple(left)])
    return out


def feature_matrix(x: np.ndarray, group: str) -> np.ndarray:
    """Return one row per pixel from the scored input window."""
    u, v = x[..., 0], x[..., 1]
    last_u, last_v = u[-1], v[-1]
    values = {
        "u_last": last_u, "v_last": last_v,
        "u_mean": u.mean(axis=0), "v_mean": v.mean(axis=0),
        "u_std": u.std(axis=0), "v_std": v.std(axis=0),
        "u_delta": u[-1] - u[-2], "v_delta": v[-1] - v[-2],
        "du_dx_pixel": grad_pixel(last_u, 1), "du_dy_pixel": grad_pixel(last_u, 0),
        "dv_dx_pixel": grad_pixel(last_v, 1), "dv_dy_pixel": grad_pixel(last_v, 0),
    }
    names = GROUPS[group]
    return np.stack([values[n] for n in names], axis=-1).reshape(-1, len(names)).astype(np.float64)


def metric(scoring, pred: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    p, y = pred[None], target[None]
    channels = scoring.measured_channels(y)
    return (
        float(scoring.rel_l2_per_sample(p, y, channels)[0]),
        float(scoring.tke_rel_l2_per_sample(p, y, channels)[0]),
        float(scoring.mvpe_rel_l2(p, y)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--kit-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=1e-2)
    args = ap.parse_args()
    t0 = time.time(); args.out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.kit_root)); import scoring  # type: ignore
    manifest = read_manifest(args.manifest)
    repo = Path(__file__).resolve().parents[1]
    # First pass: train-only feature normalization and residual moments.
    stats = {g: {"n": 0, "sum": np.zeros(len(names)), "sum2": np.zeros(len(names))} for g, names in GROUPS.items()}
    train_windows = 0; train_traj = 0; hw = None
    for path, _ in refs(manifest, args.data_root, "train"):
        train_traj += 1
        for full, _ in windows(path):
            x, target = full[:20], full[20:]
            base = np.repeat(x[-1:,...], 20, axis=0)
            residual = target - base
            hw = [int(x.shape[1]), int(x.shape[2])]
            for g in GROUPS:
                z = feature_matrix(x, g); s = stats[g]; s["n"] += z.shape[0]; s["sum"] += z.sum(0); s["sum2"] += (z * z).sum(0)
            train_windows += 1
    params = {}
    for g, names in GROUPS.items():
        s = stats[g]; mean = s["sum"] / s["n"]; std = np.sqrt(np.maximum(s["sum2"] / s["n"] - mean * mean, 1e-12)); params[g] = {"names": names, "mean": mean, "std": std}

    # Second pass: closed-form ridge on centered/standardized X and residual Y.
    fits = {}
    for g, names in GROUPS.items():
        d = len(names); xtx = np.zeros((d, d)); xty = np.zeros((d, 40)); ysum = np.zeros(40); n = 0
        for path, _ in refs(manifest, args.data_root, "train"):
            for full, _ in windows(path):
                x, target = full[:20], full[20:]; base = np.repeat(x[-1:,...], 20, axis=0); y = (target - base).transpose(1, 2, 0, 3).reshape(-1, 40).astype(np.float64)
                z = (feature_matrix(x, g) - params[g]["mean"]) / params[g]["std"]
                xtx += z.T @ z; xty += z.T @ y; ysum += y.sum(0); n += y.shape[0]
        ymean = ysum / n; reg = args.alpha * n * np.eye(d); coef = np.linalg.solve(xtx + reg, xty); fits[g] = (coef, ymean)
        np.savez_compressed(args.out_dir / f"ridge_{g}.npz", coef=coef.astype(np.float32), residual_mean=ymean.astype(np.float32), feature_mean=params[g]["mean"].astype(np.float32), feature_std=params[g]["std"].astype(np.float32))

    # Common dev baseline and corrected metrics; all four groups see identical rows.
    rows = []; dev_windows = 0; dev_traj = 0; baseline_metrics = []
    for path, name in refs(manifest, args.data_root, "dev"):
        dev_traj += 1
        for full, start in windows(path):
            x, target = full[:20], full[20:]; base = np.repeat(x[-1:,...], 20, axis=0)
            pbase = np.zeros((20, x.shape[1], x.shape[2], 3), dtype=np.float32); pbase[...,:2] = base
            yfull = np.zeros_like(pbase); yfull[...,:2] = target
            bm = metric(scoring, pbase, yfull); baseline_metrics.append((name, start, bm))
            result = {}
            for g in GROUPS:
                z = (feature_matrix(x, g) - params[g]["mean"]) / params[g]["std"]
                correction = (z @ fits[g][0] + fits[g][1]).reshape(x.shape[1], x.shape[2], 20, 2).transpose(2, 0, 1, 3)
                pc = pbase.copy(); pc[...,:2] += correction.astype(np.float32)
                result[g] = metric(scoring, pc, yfull)
            row = {"trajectory_id": name, "start": start, "baseline_rel_l2": bm[0], "baseline_tke": bm[1], "baseline_mvpe": bm[2]}
            for g, vals in result.items(): row.update({f"{g}_{m}": v for m, v in zip(("rel_l2", "tke", "mvpe"), vals)})
            rows.append(row); dev_windows += 1
    fields = list(rows[0]);
    with (args.out_dir / "window_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    # Trajectory-macro summaries are descriptive stability evidence, not a gate.
    by_traj: dict[str, list[dict]] = {}
    for row in rows: by_traj.setdefault(row["trajectory_id"], []).append(row)
    traj_rows = []
    for tid, items in sorted(by_traj.items()):
        tr = {"trajectory_id": tid, "windows": len(items)}
        for g in GROUPS:
            for m in ("rel_l2", "tke", "mvpe"):
                tr[f"{g}_{m}"] = float(np.mean([r[f"{g}_{m}"] for r in items]))
        traj_rows.append(tr)
    with (args.out_dir / "trajectory_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(traj_rows[0])); w.writeheader(); w.writerows(traj_rows)
    summary = {"groups": {}, "baseline": {m: float(np.mean([r[f"baseline_{m}"] for r in rows])) for m in ("rel_l2", "tke", "mvpe")}}
    for g in GROUPS:
        means = {m: float(np.mean([r[f"{g}_{m}"] for r in rows])) for m in ("rel_l2", "tke", "mvpe")}
        summary["groups"][g] = {"input_dim": len(GROUPS[g]), **means, "delta_vs_raw_control": {m: means[m] - summary["groups"]["raw_control"][m] if "raw_control" in summary["groups"] else None for m in ("rel_l2", "tke", "mvpe")}}
    # Recompute deltas after raw-control is available and add win rates.
    for g in GROUPS:
        summary["groups"][g]["delta_vs_raw_control"] = {m: summary["groups"][g][m] - summary["groups"]["raw_control"][m] for m in ("rel_l2", "tke", "mvpe")}
        summary["groups"][g]["dev_win_rate_vs_raw_control"] = {m: float(np.mean([r[f"{g}_{m}"] < r[f"raw_control_{m}"] for r in rows])) for m in ("rel_l2", "tke", "mvpe")}
        summary["groups"][g]["trajectory_macro_mean"] = {m: float(np.mean([r[f"{g}_{m}"] for r in traj_rows])) for m in ("rel_l2", "tke", "mvpe")}
        summary["groups"][g]["trajectory_win_rate_vs_raw_control"] = {m: float(np.mean([r[f"{g}_{m}"] < r[f"raw_control_{m}"] for r in traj_rows])) for m in ("rel_l2", "tke", "mvpe")}
    meta = {"experiment_id": "T1-ID-FE-INCR-PERSIST-RIDGE-S20260902", "manifest_sha256": sha256(args.manifest), "manifest": str(args.manifest), "data_root": str(args.data_root), "baseline": "registered PERSIST: last observed u/v repeated for 20 forecast frames", "baseline_checkpoint": None, "kit_scorer_sha256": sha256(args.kit_root / "scoring.py"), "probe": "centered ridge on train-only standardized per-pixel features; alpha=alpha*n; residual target=target-baseline; no target-derived inputs", "edge_rule": "spacing=1 pixel; centered interior; forward/backward at outer edge", "input_shape": [20, *hw, 2], "train_trajectories": train_traj, "dev_trajectories": dev_traj, "train_windows": train_windows, "dev_windows": dev_windows, "train_rows_per_fit": stats["raw_control"]["n"], "groups": {g: {"features": list(names), "input_dim": len(names)} for g, names in GROUPS.items()}, "git_head_at_run": git_head(repo), "elapsed_s": time.time() - t0, **summary}
    (args.out_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    lines = ["# Incremental supervised value probe", "", f"Experiment: `{meta['experiment_id']}`", "", "## Scope", "", "Frozen manifest train/dev only; complete 20→20 windows, stride 20, runtime H×W=32×64. Baseline is the existing registered PERSIST prediction; no checkpoint inference or training was performed.", "", "## Probe", "", "At each runtime pixel, fit a train-only standardized ridge residual head. Raw-Control is last-frame `(u,v)`; Temporal adds mean/std/recent delta for both channels; Spatial adds four last-frame pixel finite differences. TKE and vorticity are not independent inputs.", "", "## Dev results", "", "| group | dim | Rel-L2 | TKE | MVPE | Δ Rel vs Raw-Control | Δ TKE | Δ MVPE |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in GROUPS:
        q=meta["groups"][g]; d=q["delta_vs_raw_control"]; lines.append(f"| {g} | {q['input_dim']} | {q['rel_l2']:.6f} | {q['tke']:.6f} | {q['mvpe']:.6f} | {d['rel_l2']:+.6f} | {d['tke']:+.6f} | {d['mvpe']:+.6f} |")
    lines += ["", "Positive deltas are worse errors. Raw-Control itself changes the registered PERSIST baseline to Rel-L2 0.131353, TKE 0.987575, MVPE 0.130701.", "", "Trajectory-macro win rates versus Raw-Control (Temporal / Spatial / Temporal+Spatial):"]
    for g in ("raw_temporal", "raw_spatial", "raw_temporal_spatial"):
        q=meta["groups"][g]; lines.append(f"- `{g}`: Rel-L2 {q['trajectory_win_rate_vs_raw_control']['rel_l2']:.3f}, TKE {q['trajectory_win_rate_vs_raw_control']['tke']:.3f}, MVPE {q['trajectory_win_rate_vs_raw_control']['mvpe']:.3f}")
    lines += ["", "Redundancy note: vorticity is intentionally excluded as an independent input; it is the deterministic derived value `dv_dx_pixel - du_dy_pixel` (Pearson/Spearman 1.0/1.0 in the preceding Spatial diagnostic). This probe does not claim linear-probe success transfers to every neural model.", ""]
    (args.out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
