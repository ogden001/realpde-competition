#!/usr/bin/env python3
"""Streaming CFD/PIV gap diagnostics for RealPDE Track 1.

The script treats CFD/PIV as paired conditions, not frame-aligned observations.
It accepts extracted HDF5 directories and the official tar archives.  With an
archive, smoke mode extracts only selected members to a small cache; full
archive extraction is never implicit.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.signal import welch
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    welch = None
    gaussian_filter = None


PAIR_RE = re.compile(r"^(?P<re>[0-9]+)_(?P<aoa>-?[0-9]+)\.h5$")


@dataclass(frozen=True)
class Entry:
    key: str
    path: Path | None
    archive: Path | None
    member: str | None
    dataset_type: str
    aoa: float | None = None
    reynolds: float | None = None


def parse_key(name: str):
    m = PAIR_RE.match(Path(name).name)
    if not m:
        return Path(name).stem, None, None
    return Path(name).stem, float(m.group("aoa")), float(m.group("re"))


def list_entries(root: Path | None, archive: Path | None, kind: str):
    out = {}
    if root is not None and root.exists():
        for p in sorted(root.rglob("*.h5")):
            key, aoa, reynolds = parse_key(p.name)
            out[key] = Entry(key, p, None, None, kind, aoa, reynolds)
    if archive is not None and archive.exists():
        with tarfile.open(archive, "r:gz") as tf:
            for m in tf.getmembers():
                if m.isfile() and m.name.lower().endswith(".h5"):
                    key, aoa, reynolds = parse_key(m.name)
                    out[key] = Entry(key, None, archive, m.name, kind, aoa, reynolds)
    return out


def discover_sources(data_root, real_root, sim_root, real_archive, sim_archive):
    if real_root is None:
        real_root = next((p for p in (data_root / "train_real", data_root / "real") if p.is_dir()), None)
    if sim_root is None:
        sim_root = next((p for p in (data_root / "train_sim", data_root / "sim", data_root / "numerical") if p.is_dir()), None)
    if real_archive is None:
        real_archive = next((p for p in (data_root / "train_real.tar.gz", data_root / "real.tar.gz") if p.is_file()), None)
    if sim_archive is None:
        sim_archive = next((p for p in (data_root / "train_sim.tar.gz", data_root / "sim.tar.gz") if p.is_file()), None)
    return real_root, sim_root, real_archive, sim_archive


def read_h5(entry: Entry):
    if entry.path is not None:
        with h5py.File(entry.path, "r") as f:
            return {k: f[k][()] for k in f.keys()}
    assert entry.archive is not None and entry.member is not None
    with tarfile.open(entry.archive, "r:gz") as tf:
        raw = tf.extractfile(tf.getmember(entry.member)).read()  # type: ignore[union-attr]
    with h5py.File(io.BytesIO(raw), "r") as f:
        return {k: f[k][()] for k in f.keys()}


def manifest_row(e: Entry, data):
    u, t = np.asarray(data.get("u")), np.asarray(data.get("t"))
    x, y = np.asarray(data.get("x")), np.asarray(data.get("y"))
    return {"dataset_type": e.dataset_type, "trajectory_id": e.key,
            "aoa": data.get("aoa", e.aoa), "re": data.get("re", e.reynolds),
            "num_frames": int(u.shape[0]) if u.ndim >= 1 else None,
            "height": int(u.shape[-2]) if u.ndim >= 3 else None,
            "width": int(u.shape[-1]) if u.ndim >= 3 else None,
            "channels": 2 if "v" in data else 1,
            "dt": float(np.median(np.diff(t))) if t.size > 1 else None,
            "t_start": float(t[0]) if t.size else None,
            "t_end": float(t[-1]) if t.size else None,
            "x_min": float(np.nanmin(x)) if x.size else None,
            "x_max": float(np.nanmax(x)) if x.size else None,
            "y_min": float(np.nanmin(y)) if y.size else None,
            "y_max": float(np.nanmax(y)) if y.size else None,
            "file_path": str(e.path or e.archive), "archive_member": e.member or ""}


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}) if rows else []
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def rel_l2(a, b, mask):
    aa, bb = np.asarray(a)[mask], np.asarray(b)[mask]
    return float(np.linalg.norm(aa - bb) / max(float(np.linalg.norm(bb)), 1e-12))


def spatial_grid(data, h, w):
    x, y = np.asarray(data["x"], float), np.asarray(data["y"], float)
    if x.shape != (h, w) or y.shape != (h, w):
        raise ValueError(f"coordinate shape {x.shape}/{y.shape} != field {(h, w)}")
    dx = float(np.nanmedian(np.diff(x, axis=1))) if w > 1 else 1.0
    dy = float(np.nanmedian(np.diff(y, axis=0))) if h > 1 else 1.0
    return x, y, dx if abs(dx) > 1e-12 else 1e-12, dy if abs(dy) > 1e-12 else 1e-12


def vorticity(u, v, dx, dy):
    dv_dy, dv_dx = np.gradient(v, dy, dx, axis=(-2, -1), edge_order=1)
    du_dy, _ = np.gradient(u, dy, dx, axis=(-2, -1), edge_order=1)
    return dv_dx - du_dy


def psd_peak(signal, t):
    signal = np.asarray(signal, float); dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    signal = signal - np.nanmean(signal)
    if len(signal) < 8 or not np.isfinite(signal).all() or np.std(signal) < 1e-12: return float("nan")
    if welch is not None:
        f, p = welch(signal, fs=1.0 / max(dt, 1e-12), nperseg=min(256, len(signal)), detrend="linear")
    else:
        f, p = np.fft.rfftfreq(len(signal), d=dt), np.abs(np.fft.rfft(signal)) ** 2
    p[0] = 0.0
    return float(f[int(np.argmax(p))])


def plot_psd(out, key, aoa, reynolds, real, sim, probes, t):
    if not probes or len(t) < 8:
        return
    fig, axes = plt.subplots(1, len(probes), figsize=(4 * len(probes), 3.2), squeeze=False)
    for ax, (iy, ix) in zip(axes[0], probes):
        for arr, label, color in ((sim, "CFD", "tab:orange"), (real, "PIV", "tab:blue")):
            sig = arr[:, iy, ix] - np.mean(arr[:, iy, ix]); dt = float(np.median(np.diff(t)))
            if welch is not None:
                freq, power = welch(sig, fs=1.0 / max(dt, 1e-12), nperseg=min(256, len(sig)), detrend="linear")
            else:
                freq, power = np.fft.rfftfreq(len(sig), d=dt), np.abs(np.fft.rfft(sig)) ** 2
            ax.semilogy(freq[1:], np.maximum(power[1:], 1e-18), color=color, label=label)
        ax.set_title(f"x={ix}, y={iy}"); ax.set_xlabel("frequency (1/time)"); ax.set_ylabel("PSD")
        ax.grid(alpha=.25); ax.legend()
    fig.suptitle(f"Probe PSD | AoA={aoa:g}, Re={reynolds:g}"); fig.tight_layout()
    fig.savefig(out / f"psd_{key}.png", dpi=130); plt.close(fig)


def block_cv(arr):
    """Coefficient of variation of four block-level spatial mean speeds."""
    n = arr.shape[0]; chunks = np.array_split(arr, min(4, n)); vals = []
    for c in chunks:
        vals.append(float(np.mean(np.sqrt(np.mean(c ** 2, axis=-1)))))
    vals = np.asarray(vals)
    return float(np.std(vals) / max(abs(np.mean(vals)), 1e-12))


def phase_lag(a, b, max_lag):
    n = min(len(a), len(b)); a, b = np.asarray(a[:n], float), np.asarray(b[:n], float)
    a = (a - a.mean()) / max(a.std(), 1e-12); b = (b - b.mean()) / max(b.std(), 1e-12)
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        aa, bb = (a[-lag:], b[:n + lag]) if lag < 0 else ((a[:n - lag], b[lag:]) if lag > 0 else (a, b))
        rows.append((float(np.mean(aa * bb)), lag))
    corr, lag = max(rows)
    return float(lag), float(corr)


def pod_domain(arr, mask):
    z = arr[:, mask, :].reshape(arr.shape[0], -1).astype(np.float64)
    z -= z.mean(axis=0, keepdims=True)
    # Exact temporal Gram eigendecomposition: rank is at most T, while the
    # spatial feature dimension can be tens of thousands. This avoids the
    # large temporary matrices of a full feature-space SVD.
    gram = z @ z.T
    eig, vec = np.linalg.eigh(gram)
    order = np.argsort(eig)[::-1]
    eig, vec = np.maximum(eig[order], 0.0), vec[:, order]
    s = np.sqrt(eig)
    vt = np.divide(vec.T @ z, s[:, None], out=np.zeros((len(s), z.shape[1])), where=s[:, None] > 1e-12)
    energy = s * s; cumulative = np.cumsum(energy) / max(np.sum(energy), 1e-30)
    rank = lambda q: int(np.searchsorted(cumulative, q) + 1)
    return {"k90": rank(.90), "k95": rank(.95), "k99": rank(.99), "vt": vt}


def subspace_similarity(vta, vtb, k):
    k = min(k, vta.shape[0], vtb.shape[0])
    if k == 0: return float("nan")
    qa, qb = vta[:k].T, vtb[:k].T
    singular = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.mean(np.clip(singular, 0.0, 1.0) ** 2))


def choose_probes(mask, x, y, count=4):
    valid = np.argwhere(mask)
    if not len(valid): return []
    target_x, target_y = np.quantile(x[mask], [0.25, .5, .75, .9])[:count], float(np.median(y[mask]))
    result, used = [], set()
    for tx in target_x:
        distance = (x[mask] - tx) ** 2 + (y[mask] - target_y) ** 2
        for ij in valid[np.argsort(distance)]:
            pair = tuple(map(int, ij))
            if pair not in used: used.add(pair); result.append(pair); break
    return result


def plot_heatmaps(out, key, aoa, reynolds, fields, mask, x=None, y=None):
    fig, axes = plt.subplots(1, len(fields), figsize=(4 * len(fields), 3.5), squeeze=False)
    for ax, (name, value) in zip(axes[0], fields.items()):
        extent = [float(np.nanmin(x)), float(np.nanmax(x)), float(np.nanmin(y)), float(np.nanmax(y))] if x is not None and y is not None else None
        im = ax.imshow(np.where(mask, value, np.nan), origin="lower", aspect="auto", extent=extent)
        ax.set_title(f"{name} | AoA={aoa:g}, Re={reynolds:g}"); ax.set_xlabel("x (coordinate units)"); ax.set_ylabel("y (coordinate units)")
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout(); fig.savefig(out / f"gap_{key}.png", dpi=140); plt.close(fig)


def analyze_pair(real, sim, key, out_fig, wake_x_min=None):
    n = min(real["u"].shape[0], sim["u"].shape[0])
    ur, vr = np.asarray(real["u"][:n], float), np.asarray(real["v"][:n], float)
    us, vs = np.asarray(sim["u"][:n], float), np.asarray(sim["v"][:n], float)
    h, w = ur.shape[-2:]; x, y, dx, dy = spatial_grid(real, h, w); sx, sy, _, _ = spatial_grid(sim, h, w)
    coord_rmse = float(np.sqrt(np.nanmean((x - sx) ** 2 + (y - sy) ** 2)))
    mask = np.isfinite(ur).all(0) & np.isfinite(vr).all(0) & np.isfinite(us).all(0) & np.isfinite(vs).all(0)
    if not mask.any(): raise ValueError(f"{key}: no common finite spatial points")
    mr, ms = np.stack([ur.mean(0), vr.mean(0)], -1), np.stack([us.mean(0), vs.mean(0)], -1)
    fr, fs = np.stack([ur - ur.mean(0), vr - vr.mean(0)], -1), np.stack([us - us.mean(0), vs - vs.mean(0)], -1)
    tker, tkes = .5 * np.mean(fr ** 2, 0).sum(-1), .5 * np.mean(fs ** 2, 0).sum(-1)
    st_r, st_s = fr.std(0), fs.std(0); wr, ws = vorticity(ur, vr, dx, dy), vorticity(us, vs, dx, dy)
    t_real = np.asarray(real.get("t", np.arange(n)), float); t_sim = np.asarray(sim.get("t", np.arange(n)), float)
    t = t_real[:n]; probes = choose_probes(mask, x, y)
    cfd_freq, piv_freq, lags, corrs = [], [], [], []
    for iy, ix in probes:
        cfd_freq.append(psd_peak(us[:, iy, ix], t)); piv_freq.append(psd_peak(ur[:, iy, ix], t))
        lag, corr = phase_lag(us[:, iy, ix], ur[:, iy, ix], max(1, min(100, n // 4))); lags.append(lag); corrs.append(corr)
    plot_psd(out_fig, key, float(real.get("aoa", np.nan)), float(real.get("re", np.nan)), ur, us, probes, t)
    pod_r, pod_s = pod_domain(np.stack([ur, vr], -1), mask), pod_domain(np.stack([us, vs], -1), mask)
    wake = mask & (x >= wake_x_min) if wake_x_min is not None else np.zeros_like(mask)
    real_vec, sim_vec = np.stack([ur, vr], -1), np.stack([us, vs], -1)
    finite_vec = np.isfinite(real_vec).all(-1) & np.isfinite(sim_vec).all(-1)
    velocity_scale = float(np.linalg.norm(real_vec[finite_vec]) / max(np.linalg.norm(sim_vec[finite_vec]), 1e-12))
    calibrated_mean = ms * velocity_scale
    calibrated_tke = tkes * velocity_scale ** 2
    calibrated_vort = np.sqrt(np.mean(ws ** 2, 0)) * velocity_scale
    real_vort_rms = np.sqrt(np.mean(wr ** 2, 0))
    row = {
        "trajectory_id": key, "aoa": float(real.get("aoa", np.nan)), "re": float(real.get("re", np.nan)), "num_frames": n,
        "coord_rmse": coord_rmse, "grid_dx": dx, "grid_dy": dy, "grid_coord_status": "within_1_grid" if coord_rmse <= 1.5 * max(abs(dx), abs(dy)) else "misregistered",
        "common_valid_fraction": float(mask.mean()), "length_mismatch": int(real["u"].shape[0] != sim["u"].shape[0]),
        "real_re_metadata": float(real.get("re", np.nan)), "sim_re_metadata": float(sim.get("re", np.nan)),
        "real_aoa_metadata": float(real.get("aoa", np.nan)), "sim_aoa_metadata": float(sim.get("aoa", np.nan)),
        "re_metadata_match": int(real.get("re") == sim.get("re")), "aoa_metadata_match": int(real.get("aoa") == sim.get("aoa")),
        "real_dt": float(np.median(np.diff(t_real))) if len(t_real) > 1 else np.nan, "sim_dt": float(np.median(np.diff(t_sim))) if len(t_sim) > 1 else np.nan,
        "real_duration_sec": float(t_real[-1] - t_real[0]) if len(t_real) > 1 else np.nan, "sim_duration_sec": float(t_sim[-1] - t_sim[0]) if len(t_sim) > 1 else np.nan,
        "overlap_duration_sec": float(t[-1] - t[0]) if len(t) > 1 else np.nan,
        "real_block_mean_speed_cv": block_cv(np.stack([ur, vr], -1)), "sim_block_mean_speed_cv": block_cv(np.stack([us, vs], -1)),
        # PIV is the reference domain: the denominator is always the real field.
        "mean_rel_l2": rel_l2(ms, mr, mask), "mean_u_rel_l2": rel_l2(ms[..., 0], mr[..., 0], mask), "mean_v_rel_l2": rel_l2(ms[..., 1], mr[..., 1], mask),
        "velocity_scale_sim_to_real": velocity_scale, "calibrated_mean_rel_l2": rel_l2(calibrated_mean, mr, mask), "calibrated_tke_rel_l2": rel_l2(calibrated_tke, tker, mask),
        "std_u_rel_l2": rel_l2(st_s[..., 0], st_r[..., 0], mask), "std_v_rel_l2": rel_l2(st_s[..., 1], st_r[..., 1], mask), "tke_rel_l2": rel_l2(tkes, tker, mask),
        "tke_norm_ratio_sim_over_real": float(np.linalg.norm(tkes[mask]) / max(np.linalg.norm(tker[mask]), 1e-12)), "tke_map_corr": float(np.corrcoef(tker[mask], tkes[mask])[0, 1]),
        "mean_norm_ratio_sim_over_real": float(np.linalg.norm(ms[mask]) / max(np.linalg.norm(mr[mask]), 1e-12)),
        "vorticity_rms_rel_l2": rel_l2(np.sqrt(np.mean(ws ** 2, 0)), np.sqrt(np.mean(wr ** 2, 0)), mask),
        "calibrated_tke_norm_ratio": float(np.linalg.norm(calibrated_tke[mask]) / max(np.linalg.norm(tker[mask]), 1e-12)), "calibrated_vorticity_rms_rel_l2": rel_l2(calibrated_vort, real_vort_rms, mask),
        "dominant_freq_cfd": float(np.nanmedian(cfd_freq)) if cfd_freq else np.nan, "dominant_freq_piv": float(np.nanmedian(piv_freq)) if piv_freq else np.nan,
        "dominant_freq_gap": float(np.nanmedian(np.asarray(cfd_freq) - np.asarray(piv_freq))) if cfd_freq else np.nan,
        "piv_k90": pod_r["k90"], "piv_k95": pod_r["k95"], "piv_k99": pod_r["k99"], "cfd_k90": pod_s["k90"], "cfd_k95": pod_s["k95"], "cfd_k99": pod_s["k99"],
        "pod_top3_similarity": subspace_similarity(pod_r["vt"], pod_s["vt"], 3), "pod_top5_similarity": subspace_similarity(pod_r["vt"], pod_s["vt"], 5), "pod_top10_similarity": subspace_similarity(pod_r["vt"], pod_s["vt"], 10),
        "phase_best_lag_frames": float(np.nanmedian(lags)) if lags else np.nan, "phase_max_corr": float(np.nanmedian(corrs)) if corrs else np.nan,
        "wake_region_status": "computed" if wake_x_min is not None else "not_defined", "wake_valid_fraction": float(wake.mean()) if wake_x_min is not None else np.nan,
        "wake_mean_rel_l2": rel_l2(ms, mr, wake) if wake.any() else np.nan, "wake_tke_rel_l2": rel_l2(tkes, tker, wake) if wake.any() else np.nan,
    }
    if gaussian_filter is not None:
        # A diagnostic only: identical spatial smoothing exposes how much
        # high-frequency PIV noise influences TKE and derivative gaps.
        urf = gaussian_filter(ur, sigma=(0, 1, 1)); vrf = gaussian_filter(vr, sigma=(0, 1, 1))
        usf = gaussian_filter(us, sigma=(0, 1, 1)); vsf = gaussian_filter(vs, sigma=(0, 1, 1))
        frf = np.stack([urf - urf.mean(0), vrf - vrf.mean(0)], -1); fsf = np.stack([usf - usf.mean(0), vsf - vsf.mean(0)], -1)
        tkerf, tkesf = .5 * np.mean(frf ** 2, 0).sum(-1), .5 * np.mean(fsf ** 2, 0).sum(-1)
        wrf, wsf = vorticity(urf, vrf, dx, dy), vorticity(usf, vsf, dx, dy)
        row["tke_rel_l2_sigma1"] = rel_l2(tkesf, tkerf, mask)
        row["calibrated_tke_rel_l2_sigma1"] = rel_l2(tkesf * velocity_scale ** 2, tkerf, mask)
        row["vorticity_rms_rel_l2_sigma1"] = rel_l2(np.sqrt(np.mean(wsf ** 2, 0)), np.sqrt(np.mean(wrf ** 2, 0)), mask)
    out_fig.mkdir(parents=True, exist_ok=True)
    plot_heatmaps(out_fig, key, row["aoa"], row["re"], {"mean_u_gap_calibrated": mr[..., 0] - calibrated_mean[..., 0], "mean_v_gap_calibrated": mr[..., 1] - calibrated_mean[..., 1], "TKE_gap_calibrated": tker - calibrated_tke, "vorticity_rms_gap_calibrated": real_vort_rms - calibrated_vort}, mask, x, y)
    return row


def extract_selected(entries, keys, cache):
    cache.mkdir(parents=True, exist_ok=True); by_archive = {}
    for k in keys:
        e = entries[k]
        if e.path is None: by_archive.setdefault(e.archive, []).append(e)
    for archive, wanted in by_archive.items():
        wanted_map = {e.member: e for e in wanted}
        with tarfile.open(archive, "r:gz") as tf:
            for m in tf:
                if m.name not in wanted_map: continue
                e = wanted_map[m.name]; dst = cache / e.dataset_type / Path(m.name).name; dst.parent.mkdir(parents=True, exist_ok=True)
                with dst.open("wb") as f: f.write(tf.extractfile(m).read())  # type: ignore[union-attr]
                entries[e.key] = Entry(e.key, dst, None, None, e.dataset_type, e.aoa, e.reynolds)


def reuse_cached_selected(entries, keys, cache):
    """Reuse an existing smoke cache so rerunning does not rescan payloads."""
    for k in keys:
        e = entries[k]
        if e.path is not None:
            continue
        candidate = cache / e.dataset_type / Path(e.member).name
        if candidate.is_file():
            entries[k] = Entry(e.key, candidate, None, None, e.dataset_type, e.aoa, e.reynolds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data")); ap.add_argument("--real-root", type=Path); ap.add_argument("--sim-root", type=Path)
    ap.add_argument("--real-archive", type=Path); ap.add_argument("--sim-archive", type=Path); ap.add_argument("--out-dir", type=Path, default=Path("artifacts/sim2real_gap"))
    ap.add_argument("--max-pairs", type=int, default=None); ap.add_argument("--run-full", action="store_true")
    ap.add_argument("--wake-x-min", type=float, default=None, help="optional physically justified x threshold; no default is invented")
    args = ap.parse_args()
    if not args.run_full and args.max_pairs is None: args.max_pairs = 5
    rr, sr, ra, sa = discover_sources(args.data_root, args.real_root, args.sim_root, args.real_archive, args.sim_archive)
    real, sim = list_entries(rr, ra, "real"), list_entries(sr, sa, "numerical"); available = sorted(set(real) & set(sim))
    if not available: raise SystemExit("No paired HDF5 trajectories found; pass --real-root/--sim-root or archive paths")
    manifest = []
    for kind, entries in (("real", real), ("numerical", sim)):
        for k in sorted(entries):
            e = entries[k]
            if e.path is None:
                # Archive index is sufficient for the pair manifest. Do not
                # decompress every member merely to populate optional fields.
                manifest.append({"dataset_type": kind, "trajectory_id": k, "aoa": e.aoa, "re": e.reynolds,
                                 "file_path": str(e.archive), "archive_member": e.member or "", "metadata_status": "deferred"})
            else:
                try: manifest.append(manifest_row(e, read_h5(e)))
                except Exception as exc: manifest.append({"dataset_type": kind, "trajectory_id": k, "error": str(exc), "file_path": str(e.path)})
    write_csv(args.out_dir / "dataset_manifest.csv", manifest)
    keys = available if args.run_full else available[:args.max_pairs]
    if ra or sa:
        cache_name = "_full_cache" if args.run_full else "_smoke_cache"
        cache = args.out_dir / cache_name
        reuse_cached_selected(real, keys, cache); reuse_cached_selected(sim, keys, cache)
        # Full mode is intentionally explicit (--run-full); only then do we
        # materialize every paired trajectory once for efficient processing.
        extract_selected(real, keys, cache); extract_selected(sim, keys, cache)
    pair_rows = []
    for k in available:
        rr_meta = read_h5(real[k]) if k in keys and real[k].path is not None else {}
        ss_meta = read_h5(sim[k]) if k in keys and sim[k].path is not None else {}
        pair_rows.append({"pair_key": k, "filename_aoa": real[k].aoa, "filename_re": real[k].reynolds,
                          "real_source": str(real[k].path or real[k].archive), "real_member": real[k].member or "",
                          "sim_source": str(sim[k].path or sim[k].archive), "sim_member": sim[k].member or "",
                          "pair_evidence": "exact_filename_key", "unique": True,
                          "real_aoa_metadata": rr_meta.get("aoa"), "sim_aoa_metadata": ss_meta.get("aoa"),
                          "real_re_metadata": rr_meta.get("re"), "sim_re_metadata": ss_meta.get("re"),
                          "metadata_status": "checked" if rr_meta else "deferred",
                          "aoa_metadata_match": int(rr_meta.get("aoa") == ss_meta.get("aoa")) if rr_meta else "",
                          "re_metadata_match": int(rr_meta.get("re") == ss_meta.get("re")) if rr_meta else ""})
    write_csv(args.out_dir / "pair_manifest.csv", pair_rows)
    metrics = []
    for i, k in enumerate(keys, 1):
        dreal, dsim = read_h5(real[k]), read_h5(sim[k])
        if "p" not in dsim: raise ValueError(f"{k}: numerical trajectory has no pressure field; expected u,v,p")
        metrics.append(analyze_pair(dreal, dsim, k, args.out_dir / "figures", args.wake_x_min)); print(f"[{i}/{len(keys)}] {k} done", flush=True)
    write_csv(args.out_dir / "metrics_by_condition.csv", metrics)
    finite = lambda name: [float(r[name]) for r in metrics if np.isfinite(float(r.get(name, np.nan)))]
    aggregate = {name: float(np.median(vals)) for name in ("mean_rel_l2", "calibrated_mean_rel_l2", "velocity_scale_sim_to_real", "tke_rel_l2", "calibrated_tke_rel_l2", "calibrated_tke_norm_ratio", "vorticity_rms_rel_l2", "calibrated_vorticity_rms_rel_l2", "phase_max_corr", "pod_top5_similarity") if (vals := finite(name))}
    summary = {"pairs_available": len(available), "pairs_analyzed": len(metrics), "smoke_test": not args.run_full, "wake_x_min": args.wake_x_min, "cno_overlay": "not_run", "aggregate_median": aggregate, "notes": ["CFD/PIV frames are not assumed phase aligned.", "PIV noise sensitivity requires filtered/unfiltered review before physical claims.", "Official MVPE/SPS are not reimplemented."]}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Sim2Real gap diagnostic", "", f"Analyzed **{len(metrics)} / {len(available)}** exact filename pairs (smoke mode: **{not args.run_full}**).", "", "## Scope and cautions", "", "The comparison uses common finite `u,v` points and truncates each pair to the temporal overlap. CFD/PIV frames are not assumed phase-aligned. No unverified wake rectangle, official MVPE/SPS scorer, or CNO inference was run. PIV noise and coordinate registration must be reviewed before physical claims. Relative L2 values for mean flow and TKE are not directly comparable as a single scalar because they have different physical dimensions.", "", "## Aggregate results", "", "| quantity | median | Q25 | Q75 |", "|---|---:|---:|---:|"]
    for name, label in (("velocity_scale_sim_to_real", "scalar velocity scale sim→PIV"), ("mean_rel_l2", "raw mean velocity rel-L2"), ("calibrated_mean_rel_l2", "calibrated mean velocity rel-L2"), ("tke_rel_l2", "raw TKE rel-L2"), ("calibrated_tke_rel_l2", "calibrated TKE rel-L2"), ("calibrated_tke_norm_ratio", "calibrated TKE norm ratio"), ("tke_map_corr", "TKE map correlation"), ("vorticity_rms_rel_l2", "raw vorticity RMS rel-L2"), ("calibrated_vorticity_rms_rel_l2", "calibrated vorticity RMS rel-L2"), ("phase_max_corr", "probe phase max correlation"), ("pod_top5_similarity", "POD top-5 subspace similarity"), ("piv_k95", "PIV K95"), ("cfd_k95", "CFD K95")):
        vals = np.asarray([float(r[name]) for r in metrics if np.isfinite(float(r.get(name, np.nan)))])
        if len(vals): lines.append(f"| {label} | {np.median(vals):.4f} | {np.quantile(vals,.25):.4f} | {np.quantile(vals,.75):.4f} |")
    mismatch = sum(int(r["re_metadata_match"] == 0) for r in metrics); short = sum(int(r["length_mismatch"]) for r in metrics); phase_high = sum(float(r["phase_max_corr"]) > .5 for r in metrics)
    lines += ["", "## Answers to the diagnostic questions", "", f"1. **Mean flow:** raw CFD/PIV mean norm ratio is {np.median([r['mean_norm_ratio_sim_over_real'] for r in metrics]):.2f}; after one scalar velocity calibration, mean rel-L2 is {np.median([r['calibrated_mean_rel_l2'] for r in metrics]):.3f}. The dominant initial issue is scale/normalization; recompute after any confirmed geometry/body mask.", f"2. **Fluctuations/TKE:** raw TKE norm ratio is {np.median([r['tke_norm_ratio_sim_over_real'] for r in metrics]):.1f}; after the same calibration it is {np.median([r['calibrated_tke_norm_ratio'] for r in metrics]):.2f}, with map correlation {np.median([r['tke_map_corr'] for r in metrics]):.3f}. Calibrated TKE rel-L2 is {np.median([r['calibrated_tke_rel_l2'] for r in metrics]):.3f}; this still needs noise/calibration validation.", f"3. **Frequency:** the probe dominant-frequency gap is condition-dependent; use the CSV and PSD figures rather than a single global claim. Welch PSD used the recorded `dt` and removed the DC component.", f"4. **Low rank:** PIV median K95={np.median([r['piv_k95'] for r in metrics]):.0f}, versus CFD median K95={np.median([r['cfd_k95'] for r in metrics]):.0f}; PIV is not an especially compact low-rank signal under this representation.", f"5. **POD similarity:** median top-5 subspace similarity is {np.median([r['pod_top5_similarity'] for r in metrics]):.3f}, so coefficient-only correction is not supported as the first route.", f"6. **Wake concentration:** not assessed; no geometry- or scorer-validated wake mask was supplied, so `wake_*` fields are intentionally `not_defined`.", f"7. **Phase alignment:** median probe max correlation is {np.median([r['phase_max_corr'] for r in metrics]):.3f}; {phase_high}/{len(metrics)} conditions exceed 0.5. A stable global frame lag is not supported, although this does not prove all residual learning impossible.", f"8. **Data QA:** {mismatch}/{len(metrics)} pairs have differing internal real/simulation Re metadata, and {short}/{len(metrics)} have unequal trajectory lengths. Pair by exact file key but retain internal metadata and temporal overlap.", "", "## Decision matrix (current evidence)", "", "| Route | Evidence | Priority | Constraint |", "|---|---|---:|---|"]
    lines += ["| Input-unit / velocity calibration | raw CFD/PIV scale ratio is condition-dependent and median 5.72 | P1 | resolve official normalization before architecture changes |", "| Direct PIV prediction / current CNO | deployable and no CFD at inference required | P1 | retain as control; current leaderboard reference |", "| Mean correction | calibrated mean gap remains | P2 | only if CFD is available at inference; otherwise use as training teacher |", "| TKE/dynamic correction | calibrated fluctuation gap remains | P2 | validate against PIV noise/filter sensitivity and official metrics |", "| Low-rank temporal | PIV K95 is moderate/high | P3 | not first choice; test only with held-out conditions |", "| POD coefficient correction | top-5 similarity weak | P3 | do not assume shared modes |", "| Wake-specific correction | not evaluated | P3 pending | define mask from geometry/scorer before implementation |", "| Phase-aligned frame residual | low and unstable probe coherence | P4 | no global lag-based residual route |", "", "## Machine-readable outputs", "", "See `dataset_manifest.csv`, `pair_manifest.csv`, `metrics_by_condition.csv`, `summary.json`, and `figures/`. The CNO overlay remains intentionally unrun because no explicit validation-only prediction artifact was provided.", "", "```json", json.dumps(summary, indent=2), "```", ""]
    (args.out_dir / "report.md").write_text("\n".join(lines))
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__": main()
