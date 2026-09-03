#!/usr/bin/env python3
"""V2 Sim2Real gap audit for RealPDE Track 1.

This is deliberately a diagnostics-only script.  It does not train a model or
touch the competition data.  It reads extracted official HDF5 files, removes
the known bad training trajectory, evaluates at the competition resolution
(32x64), and writes one machine-readable table plus a concise Markdown report.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.ndimage import binary_dilation
    from scipy.signal import welch
except Exception:  # pragma: no cover
    binary_dilation = None
    welch = None

PAIR_RE = re.compile(r"^(?P<re>[0-9]+)_(?P<aoa>-?[0-9]+)\.h5$")


def parse_name(path: Path):
    m = PAIR_RE.match(path.name)
    if not m:
        return path.stem, np.nan, np.nan
    return path.stem, float(m.group("re")), float(m.group("aoa"))


def read_h5(path: Path):
    with h5py.File(path, "r") as f:
        return {k: f[k][()] for k in f.keys()}


def metadata(data, fallback_re, fallback_aoa):
    def scalar(name, fallback):
        val = data.get(name, fallback)
        a = np.asarray(val).reshape(-1)
        return float(a[0]) if a.size else float(fallback)
    return scalar("re", fallback_re), scalar("aoa", fallback_aoa)


def downsample(data, ds):
    u = np.asarray(data["u"], dtype=np.float64)[:, ::ds, ::ds]
    v = np.asarray(data["v"], dtype=np.float64)[:, ::ds, ::ds]
    t = np.asarray(data["t"], dtype=np.float64).reshape(-1)
    x = np.asarray(data["x"], dtype=np.float64)[::ds, ::ds]
    y = np.asarray(data["y"], dtype=np.float64)[::ds, ::ds]
    n = min(len(t), len(u))
    return {"u": u[:n], "v": v[:n], "t": t[:n], "x": x, "y": y}


def finite_pair_mask(real, sim, boundary_pixels=1):
    # There is no explicit geometry mask in the official top-level HDF5 files.
    # Infer only near-zero, time-invariant regions and disclose this limitation.
    def near_zero(d):
        speed = np.hypot(d["u"], d["v"])
        m = np.nanmean(speed, axis=0)
        s = np.nanstd(speed, axis=0)
        p95 = float(np.nanpercentile(m[np.isfinite(m)], 95)) if np.isfinite(m).any() else 1.0
        threshold = max(1e-8, 0.005 * p95)
        return np.isfinite(m) & np.isfinite(s) & (m <= threshold) & (s <= threshold)
    common = (np.isfinite(real["x"]) & np.isfinite(real["y"]) &
              np.isfinite(sim["x"]) & np.isfinite(sim["y"]))
    solid = near_zero(real) | near_zero(sim)
    if binary_dilation is not None and boundary_pixels > 0:
        solid = binary_dilation(solid, iterations=int(boundary_pixels))
    return common & ~solid, common, solid


def spatial_derivatives(u, v, x, y):
    dx = float(np.nanmedian(np.diff(x, axis=1))) if x.shape[1] > 1 else 1.0
    dy = float(np.nanmedian(np.diff(y, axis=0))) if y.shape[0] > 1 else 1.0
    dx = dx if abs(dx) > 1e-12 else 1.0
    dy = dy if abs(dy) > 1e-12 else 1.0
    dvdy, dvdx = np.gradient(v, dy, dx, axis=(-2, -1), edge_order=1)
    dudy, _ = np.gradient(u, dy, dx, axis=(-2, -1), edge_order=1)
    return dvdx - dudy


def rel_l2(a, b, mask):
    aa, bb = np.asarray(a)[mask], np.asarray(b)[mask]
    return float(np.linalg.norm(aa - bb) / max(np.linalg.norm(bb), 1e-12))


def corr(a, b, mask=None):
    aa, bb = np.asarray(a), np.asarray(b)
    if mask is not None:
        aa, bb = aa[mask], bb[mask]
    good = np.isfinite(aa) & np.isfinite(bb)
    if good.sum() < 3 or np.std(aa[good]) < 1e-12 or np.std(bb[good]) < 1e-12:
        return np.nan
    return float(np.corrcoef(aa[good], bb[good])[0, 1])


def pod_k(arr, mask):
    z = np.stack([arr["u"][:, mask], arr["v"][:, mask]], axis=-1).reshape(arr["u"].shape[0], -1)
    z = z - np.nanmean(z, axis=0, keepdims=True)
    z = np.nan_to_num(z)
    gram = z @ z.T
    eig = np.maximum(np.linalg.eigvalsh(gram)[::-1], 0.0)
    cum = np.cumsum(eig) / max(float(eig.sum()), 1e-30)
    return tuple(int(np.searchsorted(cum, q) + 1) for q in (0.90, 0.95, 0.99))


def subspace_similarity(a, b, mask, k=5):
    def basis(arr):
        z = np.stack([arr["u"][:, mask], arr["v"][:, mask]], axis=-1).reshape(arr["u"].shape[0], -1)
        z -= np.nanmean(z, axis=0, keepdims=True); z = np.nan_to_num(z)
        # Temporal Gram eigendecomposition avoids a costly feature-space SVD.
        g = z @ z.T
        eig, vec = np.linalg.eigh(g)
        order = np.argsort(eig)[::-1]; eig, vec = np.maximum(eig[order], 0.0), vec[:, order]
        s = np.sqrt(eig)
        return np.divide(vec.T @ z, s[:, None], out=np.zeros((len(s), z.shape[1])), where=s[:, None] > 1e-12)
    va, vb = basis(a), basis(b)
    kk = min(k, va.shape[0], vb.shape[0])
    if kk == 0: return np.nan
    s = np.linalg.svd(va[:kk] @ vb[:kk].T, compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1) ** 2))


def interp_signal(arr, t_src, t_dst, iy, ix):
    z = np.asarray(arr[:, iy, ix], float)
    good = np.isfinite(z) & np.isfinite(t_src)
    return np.interp(t_dst, t_src[good], z[good]) if good.sum() >= 2 else np.full_like(t_dst, np.nan)


def phase_metrics(real, sim, mask):
    overlap_lo, overlap_hi = max(real["t"][0], sim["t"][0]), min(real["t"][-1], sim["t"][-1])
    if overlap_hi <= overlap_lo + 0.2: return np.nan, np.nan, np.nan, np.nan, 0.0
    n = min(512, max(32, int((overlap_hi - overlap_lo) / max(np.median(np.diff(real["t"])), 1e-3))))
    tt = np.linspace(overlap_lo, overlap_hi, n)
    idx = np.argwhere(mask)
    if len(idx) == 0: return np.nan, np.nan, np.nan, np.nan, float(overlap_hi - overlap_lo)
    iy, ix = map(int, idx[len(idx) // 2])
    ar = interp_signal(np.hypot(real["u"], real["v"]), real["t"], tt, iy, ix)
    bs = interp_signal(np.hypot(sim["u"], sim["v"]), sim["t"], tt, iy, ix)
    ar -= np.mean(ar); bs -= np.mean(bs)
    if np.std(ar) < 1e-12 or np.std(bs) < 1e-12: return np.nan, np.nan, np.nan, np.nan, float(overlap_hi - overlap_lo)
    cc = np.correlate(ar / np.std(ar), bs / np.std(bs), mode="full") / len(tt)
    j = int(np.argmax(cc)); lag = (j - (len(tt) - 1)) * float(np.median(np.diff(tt)))
    def peak(z, tt):
        if len(z) < 8 or np.std(z) < 1e-12: return np.nan
        if welch is not None:
            f, p = welch(z - np.mean(z), fs=1.0 / max(float(np.median(np.diff(tt))), 1e-12), nperseg=min(256, len(z)), detrend="linear")
        else:
            f = np.fft.rfftfreq(len(z), d=float(np.median(np.diff(tt))))
            p = np.abs(np.fft.rfft(z - np.mean(z))) ** 2
        if len(p) > 1: p[0] = 0.0
        return float(f[int(np.argmax(p))])
    return float(lag), float(cc[j]), peak(ar, tt), peak(bs, tt), float(overlap_hi - overlap_lo)


def analyze_pair(real, sim, key, fre, fa, boundary_pixels, make_plot=False, figdir=None):
    rre, raoa = metadata(real, fre, fa); sre, saoa = metadata(sim, fre, fa)
    real = downsample(real, 2); sim = downsample(sim, 2)
    mask, common, solid = finite_pair_mask(real, sim, boundary_pixels)
    if mask.sum() < 10: raise RuntimeError(f"too few fluid pixels for {key}: {mask.sum()}")
    mr = np.stack([real["u"].mean(0), real["v"].mean(0)], axis=-1)
    ms = np.stack([sim["u"].mean(0), sim["v"].mean(0)], axis=-1)
    vr = 0.5 * (np.var(real["u"], axis=0) + np.var(real["v"], axis=0))
    vs = 0.5 * (np.var(sim["u"], axis=0) + np.var(sim["v"], axis=0))
    nr = np.hypot(real["u"], real["v"]); ns = np.hypot(sim["u"], sim["v"])
    scale = float(np.linalg.norm(np.stack([real["u"][:, mask], real["v"][:, mask]], axis=-1)) /
                  max(np.linalg.norm(np.stack([sim["u"][:, mask], sim["v"][:, mask]], axis=-1)), 1e-12))
    # Explicit full-trajectory statistics (no min-length truncation).
    sr = spatial_derivatives(real["u"], real["v"], real["x"], real["y"])
    ss = spatial_derivatives(sim["u"], sim["v"], sim["x"], sim["y"])
    wmr, wms = sr.mean(0), ss.mean(0)
    lag, phase, piv_freq, cfd_freq, overlap = phase_metrics(real, sim, mask)
    k90, k95, k99 = pod_k(real, mask)
    row = {
        "record_type": "pair", "trajectory_id": key, "filename_re": fre, "filename_aoa": fa,
        "real_re": rre, "real_aoa": raoa, "sim_re": sre, "sim_aoa": saoa,
        "real_frames": len(real["t"]), "sim_frames": len(sim["t"]),
        "real_t_end": float(real["t"][-1]), "sim_t_end": float(sim["t"][-1]),
        "grid_h": real["u"].shape[1], "grid_w": real["u"].shape[2],
        "common_fraction": float(common.mean()), "solid_boundary_fraction": float(solid.mean()),
        "fluid_fraction": float(mask.mean()), "velocity_scale_piv_over_cfd": scale,
        "mean_raw_rel_l2": rel_l2(ms, mr, np.repeat(mask[..., None], 2, axis=-1)),
        "mean_calibrated_rel_l2": rel_l2(scale * ms, mr, np.repeat(mask[..., None], 2, axis=-1)),
        "tke_norm_ratio_raw": float(np.linalg.norm(vs[mask]) / max(np.linalg.norm(vr[mask]), 1e-12)),
        "tke_norm_ratio_calibrated": float(np.linalg.norm((scale ** 2) * vs[mask]) / max(np.linalg.norm(vr[mask]), 1e-12)),
        "tke_map_corr_raw": corr(vs, vr, mask), "tke_map_corr_calibrated": corr(scale ** 2 * vs, vr, mask),
        "vorticity_rel_l2_raw": rel_l2(wms, wmr, mask),
        "vorticity_rel_l2_calibrated": rel_l2(scale * wms, wmr, mask),
        "vorticity_corr": corr(wms, wmr, mask),
        "pod_k90_piv": k90, "pod_k95_piv": k95, "pod_k99_piv": k99,
        "pod_k90_cfd": pod_k(sim, mask), "pod_similarity_top5": subspace_similarity(real, sim, mask),
        "phase_lag_sec": lag, "phase_corr": phase, "overlap_duration_sec": overlap,
        "psd_peak_piv_hz": piv_freq, "psd_peak_cfd_hz": cfd_freq,
    }
    if make_plot and figdir is not None:
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
        for a, z, title in ((ax[0], np.hypot(mr[..., 0], mr[..., 1]), "PIV mean |u|"),
                            (ax[1], np.hypot(ms[..., 0], ms[..., 1]), "CFD mean |u|"),
                            (ax[2], np.where(mask, np.hypot(ms[..., 0], ms[..., 1]) * scale - np.hypot(mr[..., 0], mr[..., 1]), np.nan), "scaled CFD − PIV")):
            im = a.imshow(np.where(mask, z, np.nan), origin="lower", aspect="auto"); a.set_title(title); fig.colorbar(im, ax=a, shrink=.8)
        fig.suptitle(f"V2 gap | {key} | 32×64"); fig.tight_layout(); fig.savefig(figdir / f"gap_{key}.png", dpi=140); plt.close(fig)
    return row


def fit_line(x, y):
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3 or np.ptp(x[good]) < 1e-12: return None
    a, b = np.polyfit(x[good], y[good], 1); pred = a * x[good] + b
    err = y[good] - pred
    r = float(np.corrcoef(x[good], y[good])[0, 1]) if np.std(x[good]) and np.std(y[good]) else np.nan
    r2 = float(1 - np.sum(err ** 2) / max(np.sum((y[good] - y[good].mean()) ** 2), 1e-30))
    return {"a": float(a), "b": float(b), "pearson_r": r, "r2": r2,
            "mae": float(np.mean(np.abs(err))), "mean_abs_relative_error": float(np.mean(np.abs(err) / np.maximum(np.abs(y[good]), 1e-12)))}


def add_regression(rows, predictor, keyname):
    x = np.array([r[predictor] for r in rows], float); y = np.array([r["velocity_scale_piv_over_cfd"] for r in rows], float)
    fit = fit_line(x, y)
    if fit is not None:
        for r in rows:
            r[f"loo_pred_{keyname}"] = np.nan; r[f"loo_relerr_{keyname}"] = np.nan
        for group in sorted(set(x[np.isfinite(x)])):
            train = np.isfinite(x) & (x != group)
            if train.sum() < 3 or np.ptp(x[train]) < 1e-12: continue
            a, b = np.polyfit(x[train], y[train], 1)
            for r in rows:
                if r[predictor] == group:
                    p = float(a * group + b); r[f"loo_pred_{keyname}"] = p; r[f"loo_relerr_{keyname}"] = float((p - r["velocity_scale_piv_over_cfd"]) / max(abs(r["velocity_scale_piv_over_cfd"]), 1e-12))
    return fit


def write_csv(path, rows):
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-root", type=Path, required=True)
    ap.add_argument("--sim-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-pairs", type=int, default=0, help="0 means all pairs")
    ap.add_argument("--bad-key", default="7575_0")
    ap.add_argument("--boundary-pixels", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True); figdir = args.out_dir / "figures"; figdir.mkdir(exist_ok=True)
    real_paths = {parse_name(p)[0]: p for p in args.real_root.glob("*.h5")}
    sim_paths = {parse_name(p)[0]: p for p in args.sim_root.glob("*.h5")}
    keys = sorted(set(real_paths) & set(sim_paths), key=lambda k: (parse_name(real_paths[k])[1], parse_name(real_paths[k])[2]))
    excluded = [k for k in keys if k == args.bad_key]; keys = [k for k in keys if k != args.bad_key]
    if args.max_pairs: keys = keys[:args.max_pairs]
    rows, errors = [], []
    plot_keys = {k for k in ("3750_10", "10125_10", "26700_10") if k in keys}
    for key in keys:
        _, fre, fa = parse_name(real_paths[key])
        try:
            rr = analyze_pair(read_h5(real_paths[key]), read_h5(sim_paths[key]), key, fre, fa, args.boundary_pixels, key in plot_keys, figdir)
            rows.append(rr)
        except Exception as e:
            errors.append({"trajectory_id": key, "error": repr(e)})
    fits = {
        "filename_re": add_regression(rows, "filename_re", "filename_re"),
        "real_re": add_regression(rows, "real_re", "real_re"),
        "filename_aoa": add_regression(rows, "filename_aoa", "aoa"),
    }
    write_csv(args.out_dir / "metrics_v2.csv", rows)
    def fmt(v): return "n/a" if v is None else ("%.4g" % v)
    summary = ["# RealPDE Track 1：Sim2Real Gap V2 复核报告", "", "## 结论摘要", "",
               f"- 配对分析：{len(rows)} 条；按 V2 要求排除 `{args.bad_key}.h5`（共 {len(excluded)} 条）。失败条目：{len(errors)}。",
               "- 所有统计在 32×64（官方 `::2` 下采样）上计算；Mean/TKE/POD 使用各自完整轨迹，PSD/相位使用各自时间向量在重叠时间段插值。",
               "- 流体区域没有官方显式 geometry mask；本轮使用跨域共同有限区域，并把两域近零且时间不变区域作为固体候选，再做 1 像素膨胀排除。该 mask 是可复现启发式，不应当当作官方 mask。",
               "- CNO 75.58 overlay：**未执行**。本地/远程可访问范围内没有能确认对应分数的 checkpoint 与 validation prediction artifact；使用其它 baseline 代替会造成错误归因。", ""]
    summary += ["## 归一化与单位审计", "",
                "- **Confirmed**：官方归档样本为顶层 `u/v/t/x/y`（CFD 另有 `p`）；当前竞赛 loader 使用实数域与数值域各自 `::2`，目标分辨率为 32×64。",
                "- **Confirmed**：现有训练代码的 GaussianNormalizer 是按训练数据集统计 mean/std 的 dataset-level normalizer；TKE 微调入口显式设为 `normalizer=none`。",
                "- **Unknown**：当前仓库没有可核验的官方 starting-kit 单位转换说明，因此不能把 `u/v` 直接标成 m/s、无量纲速度或 `u/U∞`。",
                "- **Inferred**：原始 CFD/PIV 速度范数比随 Re 显著变化，提示存在尺度/单位或测量标定差异；以下 scale 定义为 `||PIV||/||CFD||`，不是 leaderboard 分数。", ""]
    summary += ["## Scale 与 Re 关系（线性诊断）", "", "| 自变量 | Pearson r | R² | MAE | 平均绝对相对误差 |", "|---|---:|---:|---:|---:|"]
    for name, f in fits.items():
        summary.append(f"| {name} | {fmt(f.get('pearson_r') if f else None)} | {fmt(f.get('r2') if f else None)} | {fmt(f.get('mae') if f else None)} | {fmt(f.get('mean_abs_relative_error') if f else None)} |")
    summary += ["", "LOO 规则：按文件名 Re 分组，每次完整留出一个 Re（该 Re 的所有 AoA），再以 `a·Re+b` 预测；逐条预测与相对误差已写入 `metrics_v2.csv` 的 `loo_pred_*` / `loo_relerr_*` 列。", ""]
    if rows:
        def med(k): return float(np.nanmedian([r[k] for r in rows]))
        summary += ["## 清洗后 gap 指标（中位数）", "", f"- `||PIV||/||CFD||`：{med('velocity_scale_piv_over_cfd'):.4f}", f"- Mean rel-L2（原始 / scalar 校准）：{med('mean_raw_rel_l2'):.4f} / {med('mean_calibrated_rel_l2'):.4f}", f"- TKE 范数比（原始 / scalar² 校准）：{med('tke_norm_ratio_raw'):.4f} / {med('tke_norm_ratio_calibrated'):.4f}", f"- TKE 空间相关（原始 / 校准）：{med('tke_map_corr_raw'):.4f} / {med('tke_map_corr_calibrated'):.4f}", f"- 涡量 rel-L2（原始 / scalar 校准）：{med('vorticity_rel_l2_raw'):.4f} / {med('vorticity_rel_l2_calibrated'):.4f}", f"- PIV POD K95：{med('pod_k95_piv'):.1f}；top-5 子空间相似度：{med('pod_similarity_top5'):.4f}", f"- 共同时间重叠时长：{med('overlap_duration_sec'):.3f} s；PIV/CFD PSD 主峰（插值探针）：{med('psd_peak_piv_hz'):.4g} / {med('psd_peak_cfd_hz'):.4g}", ""]
        summary += ["## TKE 分组汇总（按文件名 Re / AoA）", "", "| Re | AoA | 配对数 | scalar 校准 TKE 范数比中位数 | TKE 空间相关中位数 |", "|---:|---:|---:|---:|---:|"]
        groups = {}
        for r in rows:
            g = (r['filename_re'], r['filename_aoa']); groups.setdefault(g, []).append(r)
        for (gre, gaoa), gr in sorted(groups.items()):
            summary.append(f"| {gre:g} | {gaoa:g} | {len(gr)} | {np.nanmedian([z['tke_norm_ratio_calibrated'] for z in gr]):.4f} | {np.nanmedian([z['tke_map_corr_calibrated'] for z in gr]):.4f} |")
        summary.append("")
    summary += ["## 四路决策", "", "1. **Normalization**：P1。先核验官方单位/归一化定义，并把可确认的 transform 固化到 loader；当前证据足以支持尺度审计，但不足以宣称物理单位。",
                 "2. **Dynamic/TKE**：P2。保留 scalar 校准作为诊断基线；TKE 空间结构与涡量仍有残差，不能仅靠全局 scale 解释。",
                 "3. **Wake/MVPE**：P2。V2 只做无主观 wake 矩形的全域 mask；尚无官方 MVPE 复算或 75.58 overlay，不应据此做局部优化。",
                 "4. **Low-rank**：P3。PIV K95 与 CFD/PIV 子空间结果只能作为表征诊断，证据不足以提出低秩模型改动。", "",
                 "## Recommended Next Step", "", "P1：补齐官方 starting-kit 的单位、normalizer 与 75.58 validation prediction artifact；在同一 held-out split 上重跑 overlay。P2：对官方 mask（若可获得）做一次 mask sensitivity；P3：在证据补齐前不改模型、不做 test-time CFD residual。", "",
                 "## 复现", "", f"`{Path(__file__).as_posix()} --real-root {args.real_root} --sim-root {args.sim_root} --out-dir {args.out_dir}`", ""]
    (args.out_dir / "report.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"wrote {args.out_dir / 'report.md'} and {args.out_dir / 'metrics_v2.csv'}; pairs={len(rows)} errors={len(errors)}")


if __name__ == "__main__":
    main()
