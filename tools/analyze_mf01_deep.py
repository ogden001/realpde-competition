#!/usr/bin/env python3
"""Prediction-only MF-01 mechanism diagnostics.

Inputs are matched Control/MF-01 Future20 predictions from the same frozen
Dev replay.  No model, checkpoint, target split, or per-trajectory tuning is
performed here.  Target-dependent quantities are diagnostic-only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def tke_map(x):
    q = x[..., :2] - x[..., :2].mean(axis=1, keepdims=True)
    return np.mean(np.sum(q * q, axis=-1) / 2.0, axis=1)


def frame_tke(x):
    q = x[..., :2] - x[..., :2].mean(axis=1, keepdims=True)
    return np.sum(q * q, axis=-1) / 2.0


def rms(x):
    q = x[..., :2] - x[..., :2].mean(axis=1, keepdims=True)
    return np.sqrt(np.mean(np.sum(q * q, axis=-1) / 2.0, axis=(1, 2, 3)))


def frame_rms(x):
    q = x[..., :2] - x[..., :2].mean(axis=1, keepdims=True)
    return np.sqrt(np.mean(np.sum(q * q, axis=-1) / 2.0, axis=(2, 3)))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def spearman(x, y):
    def rank(a):
        order = np.argsort(a); out = np.empty(len(a), float); out[order] = np.arange(len(a)); return out
    return float(np.corrcoef(rank(np.asarray(x)), rank(np.asarray(y)))[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--mf", type=Path, required=True)
    p.add_argument("--control-metrics", type=Path, required=True)
    p.add_argument("--mf-metrics", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True); (args.out_dir / "figures").mkdir(exist_ok=True)
    c = np.load(args.control); m = np.load(args.mf)
    cp, ct = c["prediction"], c["target"]; mp, mt = m["prediction"], m["target"]
    if cp.shape != mp.shape or not np.array_equal(ct, mt): raise ValueError("Control/MF prediction replays are not matched")
    if cp.shape[1:] != (20, 32, 64, 3): raise ValueError(f"unexpected shape {cp.shape}")
    cm, mm = read_csv(args.control_metrics), read_csv(args.mf_metrics)
    profile = {r["trajectory_id"]: r for r in read_csv(args.profile) if r["split"] == "dev"}
    # Loader order is trajectory-major; metrics carries the authoritative order/count.
    rows = []; horizons = []; group_maps = {}
    offset = 0
    for cr, mr in zip(cm, mm):
        tid = cr["trajectory_id"].removesuffix(".h5"); n = int(cr["windows"])
        if tid not in profile or mr["trajectory_id"] != cr["trajectory_id"]: raise ValueError(f"unmatched trajectory {tid}")
        c0, m0, y = cp[offset:offset+n], mp[offset:offset+n], ct[offset:offset+n]; offset += n
        # Per-window then trajectory-mean diagnostics, matching the existing handoff convention.
        cmean, mmean, ymean = c0[..., :2].mean(1), m0[..., :2].mean(1), y[..., :2].mean(1)
        cf, mf, yf = c0[..., :2]-cmean[:,None], m0[..., :2]-mmean[:,None], y[..., :2]-ymean[:,None]
        mean_c, mean_m = np.mean([rel(a,b) for a,b in zip(cmean,ymean)]), np.mean([rel(a,b) for a,b in zip(mmean,ymean)])
        fl_c, fl_m = np.mean([rel(a,b) for a,b in zip(cf,yf)]), np.mean([rel(a,b) for a,b in zip(mf,yf)])
        crms, mrms, yrms = rms(c0), rms(m0), rms(y)
        ctke, mtke, ytke = tke_map(c0), tke_map(m0), tke_map(y)
        cframe, mframe, yframe = frame_tke(c0), frame_tke(m0), frame_tke(y)
        cfrms, mfrms, yfrms = frame_rms(c0), frame_rms(m0), frame_rms(y)
        # alpha* minimizes || alpha^2 * MF_TKE_map - target_TKE_map ||_2.
        pm, tm = mtke.mean(0), ytke.mean(0); beta = max(0., float(np.sum(pm*tm) / max(np.sum(pm*pm), 1e-20))); alpha = np.sqrt(beta)
        hrows=[]
        for h in range(20):
            hrows.append({"trajectory_id":tid,"horizon":h+1,"velocity_error_control":float(np.sqrt(np.mean((c0[:,h,:,:,:2]-y[:,h,:,:,:2])**2))),"velocity_error_mf":float(np.sqrt(np.mean((m0[:,h,:,:,:2]-y[:,h,:,:,:2])**2))),"fluctuation_error_control":float(np.sqrt(np.mean((cf[:,h]-yf[:,h])**2))),"fluctuation_error_mf":float(np.sqrt(np.mean((mf[:,h]-yf[:,h])**2))),"target_fluct_rms":float(yfrms[:,h].mean()),"control_fluct_rms":float(cfrms[:,h].mean()),"mf_fluct_rms":float(mfrms[:,h].mean()),"target_tke":float(yframe[:,h].mean()),"control_tke":float(cframe[:,h].mean()),"mf_tke":float(mframe[:,h].mean())})
        horizons.extend(hrows)
        rows.append({"trajectory":tid,"profile_label":profile[tid]["distribution_label"],"nearest_train_distance":profile[tid]["nearest_train_distance"],"fluctuation_rms_input":profile[tid]["fluctuation_rms"],"delta_mean_input":profile[tid]["delta_mean"],"grad_mag_mean_input":profile[tid]["grad_mag_mean"],"vorticity_abs_mean_input":profile[tid]["vorticity_abs_mean"],"strain_mag_mean_input":profile[tid]["strain_mag_mean"],"spectrum_low_ratio":profile[tid]["spectrum_low_ratio"],"spectrum_mid_ratio":profile[tid]["spectrum_mid_ratio"],"spectrum_high_ratio":profile[tid]["spectrum_high_ratio"],"control_rel_l2":cr["rel_l2"],"mf_rel_l2":mr["rel_l2"],"delta_rel_l2":float(mr["rel_l2"])-float(cr["rel_l2"]),"control_tke":cr["tke"],"mf_tke":mr["tke"],"delta_tke":float(mr["tke"])-float(cr["tke"]),"control_mvpe":cr["mvpe"],"mf_mvpe":mr["mvpe"],"delta_mvpe":float(mr["mvpe"])-float(cr["mvpe"]),"control_mean_error":mean_c,"mf_mean_error":mean_m,"delta_mean_error":mean_m-mean_c,"control_fluctuation_error":fl_c,"mf_fluctuation_error":fl_m,"delta_fluctuation_error":fl_m-fl_c,"target_fluct_rms":float(yrms.mean()),"control_fluct_rms":float(crms.mean()),"mf_fluct_rms":float(mrms.mean()),"control_pred_target_rms_ratio":float(crms.mean()/max(yrms.mean(),1e-12)),"mf_pred_target_rms_ratio":float(mrms.mean()/max(yrms.mean(),1e-12)),"control_pred_target_tke_ratio":float(ctke.mean()/max(ytke.mean(),1e-12)),"mf_pred_target_tke_ratio":float(mtke.mean()/max(ytke.mean(),1e-12)),"alpha_star":float(alpha)})
        group_maps[tid]=(np.mean(ytke,0),np.mean(ctke,0),np.mean(mtke,0))
    if offset != len(cp): raise ValueError(f"metrics consume {offset} of {len(cp)} windows")
    rows.sort(key=lambda r: float(r["delta_tke"])); write_csv(args.out_dir/"by_trajectory.csv",rows); write_csv(args.out_dir/"by_horizon.csv",horizons)
    groups={}
    for label in ["IN_DISTRIBUTION","BOUNDARY","OOD_LIKE"]:
        a=[r for r in rows if r["profile_label"]==label]
        if a: groups[label]={"n":len(a),"delta_tke_mean":float(np.mean([r["delta_tke"] for r in a])),"delta_tke_median":float(np.median([r["delta_tke"] for r in a])),"delta_fluctuation_error_mean":float(np.mean([r["delta_fluctuation_error"] for r in a])),"alpha_star_median":float(np.median([r["alpha_star"] for r in a]))}
    corr={k:spearman([float(r[k]) for r in rows],[float(r["delta_tke"]) for r in rows]) for k in ["fluctuation_rms_input","delta_mean_input","grad_mag_mean_input","vorticity_abs_mean_input","strain_mag_mean_input","nearest_train_distance"]}
    alpha=[float(r["alpha_star"]) for r in rows]
    spatial_summary={}
    required = ["26700_0", "24150_10", "20325_20", "20325_5", "10125_0", "8850_20"]
    selected = list(dict.fromkeys(required + [r["trajectory"] for r in rows[:3]] + [r["trajectory"] for r in rows[-3:][::-1]]))
    for tid in selected:
        target, control, mf=group_maps[tid]; improvement=np.abs(target-control)-np.abs(target-mf); q20=np.percentile(target,20); q80=np.percentile(target,80)
        spatial_summary[tid]={"target_tke_p20":float(q20),"target_tke_p80":float(q80),"high_target_region_fraction":float(np.mean(target>=q80)),"low_target_region_fraction":float(np.mean(target<=q20)),"improvement_mean_high_target":float(np.mean(improvement[target>=q80])),"improvement_mean_low_target":float(np.mean(improvement[target<=q20])),"improvement_mean_global":float(np.mean(improvement)),"mf_error_minus_control_error_high_target":float(np.mean(np.abs(target-mf)[target>=q80]-np.abs(target-control)[target>=q80])),"mf_error_minus_control_error_low_target":float(np.mean(np.abs(target-mf)[target<=q20]-np.abs(target-control)[target<=q20]))}
    summary={"n_trajectories":len(rows),"n_windows":int(len(cp)),"sorted_delta_tke_good3":[r["trajectory"] for r in rows[:3]],"sorted_delta_tke_bad3":[r["trajectory"] for r in rows[-3:][::-1]],"group_summary":groups,"spearman_delta_tke":corr,"alpha_star":{"min":float(np.min(alpha)),"p25":float(np.percentile(alpha,25)),"median":float(np.median(alpha)),"p75":float(np.percentile(alpha,75)),"max":float(np.max(alpha))},"official_tke_delta_mean":float(np.mean([r["delta_tke"] for r in rows])),"horizon_mean":{"velocity_delta":[],"fluctuation_error_delta":[],"tke_amplitude_ratio_control":[],"tke_amplitude_ratio_mf":[]},"spatial_summary":spatial_summary,"locked_final_accessed":False,"codabench_accessed":False}
    for h in range(1,21):
        a=[r for r in horizons if int(r["horizon"])==h]; yt=np.mean([r["target_tke"] for r in a]); yc=np.mean([r["control_tke"] for r in a]); ym=np.mean([r["mf_tke"] for r in a]); summary["horizon_mean"]["velocity_delta"].append(float(np.mean([r["velocity_error_mf"]-r["velocity_error_control"] for r in a]))); summary["horizon_mean"]["fluctuation_error_delta"].append(float(np.mean([r["fluctuation_error_mf"]-r["fluctuation_error_control"] for r in a]))); summary["horizon_mean"]["tke_amplitude_ratio_control"].append(float(yc/yt)); summary["horizon_mean"]["tke_amplitude_ratio_mf"].append(float(ym/yt))
    (args.out_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    for tid in summary["sorted_delta_tke_good3"]+summary["sorted_delta_tke_bad3"]:
        target, control, mf=group_maps[tid]; improvement=np.abs(target-control)-np.abs(target-mf)
        fig,ax=plt.subplots(1,4,figsize=(16,4)); ims=[(target,"Target TKE"),(np.abs(target-control),"Control |error|"),(np.abs(target-mf),"MF |error|"),(improvement,"MF−Control improvement")]
        for aa,(z,title) in zip(ax,ims): im=aa.imshow(z,cmap="coolwarm"); aa.set_title(title); aa.set_xticks([]); aa.set_yticks([]); fig.colorbar(im,ax=aa,fraction=.046)
        fig.suptitle(tid); fig.tight_layout(); fig.savefig(args.out_dir/"figures"/f"{tid}_tke_maps.png",dpi=130); plt.close(fig)
    print(json.dumps({"trajectories":len(rows),"good3":summary["sorted_delta_tke_good3"],"bad3":summary["sorted_delta_tke_bad3"],"out":str(args.out_dir)},indent=2))


if __name__ == "__main__": main()
