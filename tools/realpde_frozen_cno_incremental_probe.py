#!/usr/bin/env python3
"""Frozen official-CNO residual probe for the four fixed FE packages.

The CNO weights are loaded once and never optimized.  Train-only streaming
normalization and closed-form ridge correction are applied to its predictions.
All target reads are restricted to manifest train/dev complete 20->20 windows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import h5py
import numpy as np
import torch

GROUPS = {
    "raw_control": ("u_last", "v_last"),
    "raw_temporal": ("u_last", "v_last", "u_mean", "v_mean", "u_std", "v_std", "u_delta", "v_delta"),
    "raw_spatial": ("u_last", "v_last", "du_dx_pixel", "du_dy_pixel", "dv_dx_pixel", "dv_dy_pixel"),
    "raw_temporal_spatial": ("u_last", "v_last", "u_mean", "v_mean", "u_std", "v_std", "u_delta", "v_delta", "du_dx_pixel", "du_dy_pixel", "dv_dx_pixel", "dv_dy_pixel"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def grad_pixel(a: np.ndarray, axis: int) -> np.ndarray:
    out = np.empty_like(a)
    s = [slice(None)] * a.ndim; s[axis] = 0; s0 = tuple(s); s[axis] = 1; s1 = tuple(s)
    out[s0] = a[s1] - a[s0]
    s[axis] = -1; sn = tuple(s); s[axis] = -2; sm = tuple(s); out[sn] = a[sn] - a[sm]
    mid = [slice(None)] * a.ndim; mid[axis] = slice(1, -1)
    left = [slice(None)] * a.ndim; left[axis] = slice(None, -2)
    right = [slice(None)] * a.ndim; right[axis] = slice(2, None)
    out[tuple(mid)] = 0.5 * (a[tuple(right)] - a[tuple(left)])
    return out


def feature_matrix(x: np.ndarray, group: str) -> np.ndarray:
    u, v = x[..., 0], x[..., 1]; lu, lv = u[-1], v[-1]
    q = {"u_last": lu, "v_last": lv, "u_mean": u.mean(0), "v_mean": v.mean(0),
         "u_std": u.std(0), "v_std": v.std(0), "u_delta": u[-1] - u[-2], "v_delta": v[-1] - v[-2],
         "du_dx_pixel": grad_pixel(lu, 1), "du_dy_pixel": grad_pixel(lu, 0),
         "dv_dx_pixel": grad_pixel(lv, 1), "dv_dy_pixel": grad_pixel(lv, 0)}
    names = GROUPS[group]
    return np.stack([q[n] for n in names], -1).reshape(-1, len(names)).astype(np.float64)


def iter_windows(path: Path, max_windows: int | None = None):
    with h5py.File(path, "r") as f:
        fu = f["u"] if "u" in f else f["measured_data/u"]
        fv = f["v"] if "v" in f else f["measured_data/v"]
        n = int(fu.shape[0])
        for number, start in enumerate(range(0, n - 39, 20)):
            if max_windows is not None and number >= max_windows: break
            u = np.asarray(fu[start:start + 40, ::2, ::2], dtype=np.float32)
            v = np.asarray(fv[start:start + 40, ::2, ::2], dtype=np.float32)
            yield np.stack([u, v], -1), start


def load_model(kit_root: Path, checkpoint: Path, device: torch.device):
    # Newer setuptools no longer exposes pkg_resources, while the vendored
    # starting-kit CNO uses only parse_version from that legacy module.
    try:
        import pkg_resources  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        from packaging.version import parse as _parse
        sys.modules["pkg_resources"] = types.SimpleNamespace(parse_version=_parse)
    sys.path.insert(0, str(kit_root))
    from rpde_baselines.model.cno import CNO3d
    model = CNO3d(in_dim=3, out_dim=3, out_dim_mult=1, in_size=64, N_layers=3).to(device)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model_state_dict", state), strict=True)
    model.eval()
    return model


@torch.no_grad()
def cno_batch(model, xs: np.ndarray, device: torch.device) -> np.ndarray:
    z = np.zeros((*xs.shape[:1], xs.shape[1], xs.shape[2], xs.shape[3], 3), dtype=np.float32)
    z[..., :2] = xs
    x = torch.from_numpy(z).to(device)
    y = model(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
    return y.cpu().numpy().astype(np.float32)


def metric(scoring, pred: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    p, y = pred[None], target[None]; c = scoring.measured_channels(y)
    return (float(scoring.rel_l2_per_sample(p, y, c)[0]), float(scoring.tke_rel_l2_per_sample(p, y, c)[0]), float(scoring.mvpe_rel_l2(p, y)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True); ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--kit-root", type=Path, required=True); ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True); ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=1e-2); ap.add_argument("--code-commit", default="unknown")
    ap.add_argument("--max-windows-per-trajectory", type=int, default=None)
    ap.add_argument("--experiment-id", default="T1-ID-FE-INCR-FROZEN-CNO-RIDGE-S20260902")
    ap.add_argument("--baseline-family", default="official CNO; frozen weights")
    args = ap.parse_args(); t0 = time.time(); args.out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.kit_root)); import scoring  # type: ignore
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")); device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args.kit_root, args.checkpoint, device)
    # Manifest-restricted path lists; final is never touched.
    split_paths = {s: [args.data_root / r["file"] for r in manifest[s]] for s in ("train", "dev")}
    train_windows = sum(sum(1 for _ in iter_windows(p, args.max_windows_per_trajectory)) for p in split_paths["train"])
    dev_windows = sum(sum(1 for _ in iter_windows(p, args.max_windows_per_trajectory)) for p in split_paths["dev"])
    hw = None
    stats = {g: {"n": 0, "sum": np.zeros(len(ns)), "sum2": np.zeros(len(ns))} for g, ns in GROUPS.items()}

    def batches(split):
        bx, by, bn, bs = [], [], [], []
        for path in split_paths[split]:
            for full, start in iter_windows(path, args.max_windows_per_trajectory):
                bx.append(full[:20]); by.append(full[20:]); bn.append(path.name); bs.append(start)
                if len(bx) == args.batch_size:
                    yield np.asarray(bx), np.asarray(by), bn, bs; bx, by, bn, bs = [], [], [], []
        if bx: yield np.asarray(bx), np.asarray(by), bn, bs

    # Pass 1: frozen-CNO inference plus train-only feature moments.
    for xb, _, _, _ in batches("train"):
        pred = cno_batch(model, xb, device); hw = [int(xb.shape[2]), int(xb.shape[3])]
        for x in xb:
            for g in GROUPS:
                z = feature_matrix(x, g); s = stats[g]; s["n"] += len(z); s["sum"] += z.sum(0); s["sum2"] += (z * z).sum(0)
    params = {}
    for g, ns in GROUPS.items():
        s = stats[g]; mean = s["sum"] / s["n"]; std = np.sqrt(np.maximum(s["sum2"] / s["n"] - mean * mean, 1e-12)); params[g] = (mean, std)

    # Pass 2: closed-form ridge fit on identical frozen predictions/residuals.
    fits = {}
    for g, ns in GROUPS.items():
        d = len(ns); xtx = np.zeros((d, d)); xty = np.zeros((d, 40)); ysum = np.zeros(40); nrow = 0
        for xb, yb, _, _ in batches("train"):
            pred = cno_batch(model, xb, device); yres = (yb - pred[..., :2]).transpose(0, 2, 3, 1, 4).reshape(-1, 40).astype(np.float64)
            z = np.concatenate([(feature_matrix(x, g) - params[g][0]) / params[g][1] for x in xb], axis=0); xtx += z.T @ z; xty += z.T @ yres; ysum += yres.sum(0); nrow += len(z)
        ym = ysum / nrow; coef = np.linalg.solve(xtx + args.alpha * nrow * np.eye(d), xty); fits[g] = (coef, ym)
        np.savez_compressed(args.out_dir / f"ridge_{g}.npz", coef=coef.astype(np.float32), residual_mean=ym.astype(np.float32), feature_mean=params[g][0].astype(np.float32), feature_std=params[g][1].astype(np.float32))

    rows = []; traj: dict[str, list[dict]] = {}; baseline_vals = []
    for xb, yb, names, starts in batches("dev"):
        pred = cno_batch(model, xb, device)
        for i, (x, y, name, start) in enumerate(zip(xb, yb, names, starts)):
            p = pred[i].copy(); yt = np.zeros_like(p); yt[..., :2] = y
            bm = metric(scoring, p, yt); baseline_vals.append(bm)
            row = {"trajectory_id": name, "start": start, "baseline_rel_l2": bm[0], "baseline_tke": bm[1], "baseline_mvpe": bm[2]}
            for g in GROUPS:
                z = (feature_matrix(x, g) - params[g][0]) / params[g][1]; corr = (z @ fits[g][0] + fits[g][1]).reshape(hw[0], hw[1], 20, 2).transpose(2, 0, 1, 3)
                pc = p.copy(); pc[..., :2] += corr.astype(np.float32); vals = metric(scoring, pc, yt)
                row.update({f"{g}_{m}": v for m, v in zip(("rel_l2", "tke", "mvpe"), vals)})
            rows.append(row); traj.setdefault(name, []).append(row)
    with (args.out_dir / "window_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    traj_rows = []
    for name, items in sorted(traj.items()):
        r = {"trajectory_id": name, "windows": len(items)}
        for g in GROUPS:
            for m in ("rel_l2", "tke", "mvpe"): r[f"{g}_{m}"] = float(np.mean([x[f"{g}_{m}"] for x in items]))
        traj_rows.append(r)
    with (args.out_dir / "trajectory_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(traj_rows[0])); w.writeheader(); w.writerows(traj_rows)
    summary = {"experiment_id": args.experiment_id, "manifest_sha256": sha256(args.manifest), "manifest": str(args.manifest), "data_root": str(args.data_root), "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint), "baseline_family": args.baseline_family, "device": str(device), "kit_scorer_sha256": sha256(args.kit_root / "scoring.py"), "input_protocol": "20->20, stride=20, spatial subsample=2, output=20x32x64x3", "input_shape": [20, *hw, 3], "train_trajectories": len(split_paths["train"]), "dev_trajectories": len(split_paths["dev"]), "train_windows": train_windows, "dev_windows": dev_windows, "train_rows_per_fit": stats["raw_control"]["n"], "ridge": "centered train-only standardized features; alpha*n regularization; residual=target-frozen-CNO", "code_commit": args.code_commit, "elapsed_s": time.time() - t0, "baseline": {m: float(np.mean([x[j] for x in baseline_vals])) for j, m in enumerate(("rel_l2", "tke", "mvpe"))}, "groups": {}}
    for g in GROUPS:
        q = {"input_dim": len(GROUPS[g]), **{m: float(np.mean([r[f"{g}_{m}"] for r in rows])) for m in ("rel_l2", "tke", "mvpe")}}
        q["delta_vs_raw_control"] = {m: q[m] - summary["groups"]["raw_control"][m] if "raw_control" in summary["groups"] else 0.0 for m in ("rel_l2", "tke", "mvpe")}
        q["trajectory_macro_mean"] = {m: float(np.mean([r[f"{g}_{m}"] for r in traj_rows])) for m in ("rel_l2", "tke", "mvpe")}
        q["trajectory_win_rate_vs_raw_control"] = {m: float(np.mean([r[f"{g}_{m}"] < r[f"raw_control_{m}"] for r in traj_rows])) for m in ("rel_l2", "tke", "mvpe")}
        summary["groups"][g] = q
    # Correct deltas after Raw-Control is populated.
    for g in GROUPS: summary["groups"][g]["delta_vs_raw_control"] = {m: summary["groups"][g][m] - summary["groups"]["raw_control"][m] for m in ("rel_l2", "tke", "mvpe")}
    summary["joint_vs_temporal"] = {"delta": {m: summary["groups"]["raw_temporal_spatial"][m] - summary["groups"]["raw_temporal"][m] for m in ("rel_l2", "tke", "mvpe")}, "trajectory_win_rate": {m: float(np.mean([r[f"raw_temporal_spatial_{m}"] < r[f"raw_temporal_{m}"] for r in traj_rows])) for m in ("rel_l2", "tke", "mvpe")}}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Frozen-CNO incremental residual probe", "", f"Experiment: `{summary['experiment_id']}`", "", "Frozen official CNO weights; no neural training. Complete 20→20 windows, stride 20, runtime H×W=32×64.", "", "| group | dim | Rel-L2 | TKE | MVPE | Δ vs Raw-Control (Rel/TKE/MVPE) |", "|---|---:|---:|---:|---:|---:|"]
    for g in GROUPS:
        q=summary["groups"][g]; d=q["delta_vs_raw_control"]; lines.append(f"| {g} | {q['input_dim']} | {q['rel_l2']:.6f} | {q['tke']:.6f} | {q['mvpe']:.6f} | {d['rel_l2']:+.6f} / {d['tke']:+.6f} / {d['mvpe']:+.6f} |")
    lines += ["", f"Frozen-CNO raw dev: Rel-L2 {summary['baseline']['rel_l2']:.6f}, TKE {summary['baseline']['tke']:.6f}, MVPE {summary['baseline']['mvpe']:.6f}.", "", "Joint minus Temporal (negative is better): " + " / ".join(f"{summary['joint_vs_temporal']['delta'][m]:+.6f}" for m in ("rel_l2", "tke", "mvpe")) + ".", "", "Vorticity and TKE-proxy were not independent inputs. Vorticity remains the deterministic derived value `dv_dx_pixel - du_dy_pixel`. This linear probe does not establish universal neural fusion value.", ""]
    (args.out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__": main()
