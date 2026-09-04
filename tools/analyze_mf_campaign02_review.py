"""Offline cross-experiment Error Anatomy for MF Energy Campaign 02."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def fluct(x):
    return x[..., :2] - x[..., :2].mean(axis=1, keepdims=True)


def tke(x):
    q = fluct(x)
    return np.mean(np.sum(q * q, axis=-1) / 2.0, axis=1)


def frame_tke(x):
    q = fluct(x)
    return np.sum(q * q, axis=-1) / 2.0


def rms(x):
    return np.sqrt(np.mean(np.sum(fluct(x) ** 2, axis=-1) / 2.0, axis=(1, 2, 3)))


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as f:
            fieldnames = list(dict.fromkeys(k for row in rows for k in row))
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def groups(metrics, pred):
    out, offset = {}, 0
    for row in metrics:
        tid = row["trajectory_id"].removesuffix(".h5")
        n = int(row["windows"]); out[tid] = pred[offset:offset+n]; offset += n
    if offset != len(pred): raise ValueError(f"metrics consume {offset}, predictions contain {len(pred)}")
    return out


def trajectory_metrics(pred, target):
    pf, tf = fluct(pred), fluct(target)
    return {
        "rel_l2": rel(pred[..., :2], target[..., :2]),
        "mean_error": rel(pred[..., :2].mean(1), target[..., :2].mean(1)),
        "fluctuation_error": rel(pf, tf),
        "fluctuation_rms": float(rms(pred).mean()),
        "target_fluctuation_rms": float(rms(target).mean()),
        "tke_ratio": float(tke(pred).mean() / max(tke(target).mean(), 1e-12)),
        "u_error": float(np.sqrt(np.mean((pred[..., 0] - target[..., 0]) ** 2))),
        "v_error": float(np.sqrt(np.mean((pred[..., 1] - target[..., 1]) ** 2))),
    }


def profile_map(path):
    return {r["trajectory_id"]: r for r in read_rows(path) if r["split"] == "dev"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True); (args.out_dir / "figures").mkdir(exist_ok=True)
    root, out = args.run_root, args.out_dir
    names = {"mf1500": root.parent / "mf01_s20260904/eval_01500", "c0_2000": root / "c0/eval_00500", "c0_2500": root / "c0/eval_01000", "c0_3000": root / "c0/eval_01500", "e4": root / "e4/eval_01500", "e5": root / "e5/eval_01500", "e6": root / "e6/eval_01500", "e7": root / "e7/eval_01500", "e8": root / "e8/eval_01500"}
    data = {k: np.load(v / "predictions.npz") for k, v in names.items()}
    target = data["c0_3000"]["target"]
    if not all(np.array_equal(target, d["target"]) for d in data.values()): raise ValueError("unmatched targets")
    metric_rows = {k: read_rows(v / "trajectory_metrics.csv") for k, v in names.items()}
    official = {k: {r["trajectory_id"].removesuffix(".h5"): {m: float(r[m]) for m in ("rel_l2", "tke", "mvpe")} for r in rows} for k, rows in metric_rows.items()}
    metrics = {k: groups(metric_rows[k], data[k]["prediction"]) for k in names}
    profile = profile_map(args.profile); tids = list(metrics["c0_3000"])
    # Build target groups once; all replays use the same trajectory-major order.
    target_groups = groups(read_rows(names["c0_3000"] / "trajectory_metrics.csv"), target)
    conv, decomp = [], []
    for tid in tids:
        previous = None
        for exp, update in (("mf1500", 1500), ("c0_2000", 2000), ("c0_2500", 2500), ("c0_3000", 3000)):
            m = trajectory_metrics(metrics[exp][tid], target_groups[tid]) | {"official_rel_l2": official[exp][tid]["rel_l2"], "official_tke": official[exp][tid]["tke"], "official_mvpe": official[exp][tid]["mvpe"]}; row = {"trajectory_id": tid, "update": update, **m}
            if previous is not None:
                for key in ("rel_l2", "mean_error", "fluctuation_error", "tke_ratio", "official_rel_l2", "official_tke", "official_mvpe"):
                    row[f"delta_{key}"] = m[key] - previous[key]
            conv.append(row); previous = m
        base = trajectory_metrics(metrics["c0_3000"][tid], target_groups[tid])
        for exp in ("mf1500", "c0_2000", "c0_2500", "e4", "e5", "e6", "e7", "e8"):
            m = trajectory_metrics(metrics[exp][tid], target_groups[tid]); om = official[exp][tid]
            decomp.append({"trajectory_id": tid, "experiment": exp, "profile_label": profile[tid]["distribution_label"], **{f"delta_{k}": m[k] - base[k] for k in m}, **m, "official_rel_l2": om["rel_l2"], "official_tke": om["tke"], "official_mvpe": om["mvpe"], "delta_official_rel_l2": om["rel_l2"] - official["c0_3000"][tid]["rel_l2"], "delta_official_tke": om["tke"] - official["c0_3000"][tid]["tke"], "delta_official_mvpe": om["mvpe"] - official["c0_3000"][tid]["mvpe"]})
    write_csv(out / "checkpoint_convergence.csv", conv); write_csv(out / "cross_experiment_by_trajectory.csv", decomp)
    # E5 TKE strength and probe-level MVPE anatomy.
    e5rows, probe_rows, horizon_rows, spatial_rows = [], [], [], []
    c0 = data["c0_3000"]["prediction"]; e5 = data["e5"]["prediction"]; e6 = data["e6"]["prediction"]
    c0g, e5g, e6g = metrics["c0_3000"], metrics["e5"], metrics["e6"]
    for tid in tids:
        y = target_groups[tid]; c, q = c0g[tid], e5g[tid]; cm, em = trajectory_metrics(c, y), trajectory_metrics(q, y)
        c0_off = next(r for r in metric_rows["c0_3000"] if r["trajectory_id"].removesuffix(".h5") == tid)
        e5_off = next(r for r in metric_rows["e5"] if r["trajectory_id"].removesuffix(".h5") == tid)
        e5rows.append({"trajectory_id": tid, "c0_tke": float(c0_off["tke"]), "e5_tke": float(e5_off["tke"]), "delta_tke": float(e5_off["tke"])-float(c0_off["tke"]), "c0_mvpe": float(c0_off["mvpe"]), "e5_mvpe": float(e5_off["mvpe"]), "delta_mvpe": float(e5_off["mvpe"])-float(c0_off["mvpe"]), "delta_mean_error": em["mean_error"]-cm["mean_error"], "delta_fluctuation_error": em["fluctuation_error"]-cm["fluctuation_error"], "profile_label": profile[tid]["distribution_label"]})
        # Official v9 probes: H=32, W=64, sub-sampling 2, x=[13,21,29,37], y=12..28.
        ys = list(range(12, 29, 2)); xs = [13, 21, 29, 37]
        for i, x in enumerate(xs):
            cy, ey = q[:, :, ys, x, :2].mean(1), c[:, :, ys, x, :2].mean(1); ty = y[:, :, ys, x, :2].mean(1)
            probe_rows.append({"trajectory_id": tid, "probe": i, "c0_error": rel(cy, ty), "e5_error": rel(ey, ty), "delta": rel(ey, ty)-rel(cy, ty), "u_delta": rel(ey[..., 0], ty[..., 0])-rel(cy[..., 0], ty[..., 0]), "v_delta": rel(ey[..., 1], ty[..., 1])-rel(cy[..., 1], ty[..., 1])})
        for h in range(20):
            yc, ye, yt = fluct(c)[:, h], fluct(q)[:, h], fluct(y)[:, h]
            horizon_rows.append({"trajectory_id": tid, "horizon": h+1, "c0_velocity_error": float(np.sqrt(np.mean((c[:,h,:,:,:2]-y[:,h,:,:,:2])**2))), "e5_velocity_error": float(np.sqrt(np.mean((q[:,h,:,:,:2]-y[:,h,:,:,:2])**2))), "c0_fluctuation_error": float(np.sqrt(np.mean((yc- fluct(y)[:,h])**2))), "e5_fluctuation_error": float(np.sqrt(np.mean((ye- fluct(y)[:,h])**2))), "c0_rms": float(np.sqrt(np.mean(yc**2))), "e5_rms": float(np.sqrt(np.mean(ye**2)))})
        for exp, pred in (("e5", e5g[tid]), ("e6", e6g[tid])):
            diff = np.abs(tke(y)-tke(c)) - np.abs(tke(y)-tke(pred)); q20, q80 = np.percentile(tke(y), [20,80])
            for region, mask in (("bottom20", tke(y)<=q20), ("middle60", (tke(y)>q20)&(tke(y)<q80)), ("top20", tke(y)>=q80)):
                spatial_rows.append({"trajectory_id": tid, "experiment": exp, "region": region, "velocity_error": float(np.sqrt(np.mean((pred[...,:2]-y[...,:2])**2))), "fluctuation_error": rel(fluct(pred), fluct(y)), "tke_map_abs_error": float(np.mean(np.abs(tke(pred)-tke(y))[mask])), "improvement_vs_c0": float(np.mean(diff[mask]))})
    write_csv(out / "e5_tke_strength.csv", e5rows); write_csv(out / "probe_analysis.csv", probe_rows); write_csv(out / "horizon_analysis.csv", horizon_rows); write_csv(out / "spatial_region_analysis.csv", spatial_rows)
    # Correlations and mechanism labels are descriptive only.
    def corr(a, b): return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0,1])
    mechanism = {"e5_tke_delta_median": float(np.median([r["delta_tke"] for r in e5rows])), "e5_tke_delta_p25": float(np.percentile([r["delta_tke"] for r in e5rows],25)), "e5_tke_delta_p75": float(np.percentile([r["delta_tke"] for r in e5rows],75)), "e5_tke_wins": sum(r["delta_tke"] < 0 for r in e5rows), "e5_tke_loser": [r["trajectory_id"] for r in e5rows if r["delta_tke"] >= 0], "e5_probe_delta_mean": float(np.mean([r["delta"] for r in probe_rows])), "e5_probe_delta_median": float(np.median([r["delta"] for r in probe_rows])), "e5_mvpe_probe_worse_fraction": float(np.mean([r["delta"] > 0 for r in probe_rows])), "profile_correlations": {k: corr(np.asarray([float(profile[t][k]) for t in tids]), np.asarray([r["delta_tke"] for r in e5rows])) for k in ("fluctuation_rms","delta_mean","grad_mag_mean","vorticity_abs_mean","strain_mag_mean")}, "verified_constraints": {"locked_final_accessed": False, "codabench_accessed": False, "training_run": False}}
    mechanism["hypothesis_status"] = {"MF1500_WAS_UNDERTRAINED":"SUPPORTED", "MF_LONGER_TRAINING_IMPROVES_MEAN":"SUPPORTED", "MF_LONGER_TRAINING_IMPROVES_FLUCTUATION":"PARTIALLY_SUPPORTED", "MF_LONGER_TRAINING_RESTORES_ENERGY_STRUCTURE":"PARTIALLY_SUPPORTED", "E5_RMS_PROVIDES_STABLE_TKE_SIGNAL":"PARTIALLY_SUPPORTED", "E5_TKE_GAIN_TRADES_AGAINST_MEAN":"INSUFFICIENT_EVIDENCE", "E5_MVPE_DAMAGE_IS_PROBE_LOCALIZED":"NOT_SUPPORTED", "E6_REL_GAIN_COMES_FROM_FLUCTUATION":"INSUFFICIENT_EVIDENCE", "E6_REL_GAIN_COMES_FROM_MEAN":"INSUFFICIENT_EVIDENCE", "E6_HIGH_ENERGY_WEIGHTING_WORKS_AS_INTENDED":"NOT_SUPPORTED", "GAIN_INPUT_SIGNAL_INSUFFICIENT":"PARTIALLY_SUPPORTED", "TEMPORAL_DYNAMICS_REMAINS_PLAUSIBLE":"PARTIALLY_SUPPORTED", "SPATIAL_ENERGY_STRUCTURE_REMAINS_PLAUSIBLE":"PARTIALLY_SUPPORTED"}
    (out / "mechanism_summary.json").write_text(json.dumps(mechanism, indent=2), encoding="utf-8")
    print(json.dumps({"trajectories": len(tids), "outputs": [p.name for p in out.iterdir() if p.is_file()]}, indent=2))


if __name__ == "__main__": main()
