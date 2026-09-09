#!/usr/bin/env python3
"""Fact-only CPU summaries for structured temporal dynamics rounds."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
METRICS=("rel_l2","tke","mvpe"); BANDS={"Early":(1,5),"Mid":(6,10),"Late":(11,15),"Far":(16,20)}
def rows(path):
    with path.open(newline="") as f: return list(csv.DictReader(f))
def write(path, data):
    if not data: return
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
def f(row,key): return float(row[key])
def improvement(control, candidate):
    return "" if control == 0 else (control-candidate)/control*100
def summarize(root: Path, prefix: str, out: Path):
    arms=sorted(p.name for p in root.iterdir() if p.is_dir() and (p/"aggregate_metrics.csv").exists()); out.mkdir(parents=True,exist_ok=True)
    ag={a:rows(root/a/"aggregate_metrics.csv") for a in arms}; control={int(r["update"]):r for r in ag["C0"]}; hd=[]; ts=[]; bs=[]
    for arm in arms:
        if arm=="C0": continue
        for r in ag[arm]:
            u=int(r["update"])
            if u not in control: continue
            c=control[u]; row={"arm":arm,"update":u}
            if not (root/arm/f"eval_{u:05d}").exists() or not (root/"C0"/f"eval_{u:05d}").exists(): continue
            for m in METRICS: row[m+"_improvement_pct"]=improvement(f(c,m),f(r,m))
            ta=rows(root/arm/f"eval_{u:05d}"/"temporal_metrics.csv"); tc=rows(root/"C0"/f"eval_{u:05d}"/"temporal_metrics.csv")
            for m in ("increment_uv_mse","error_l2"): row[m+"_delta"]=sum(f(x,m) for x in ta)/len(ta)-sum(f(x,m) for x in tc)/len(tc)
            hd.append(row)
            a={x["trajectory"]:x for x in rows(root/arm/f"eval_{u:05d}"/"trajectory_metrics.csv")}; d={x["trajectory"]:x for x in rows(root/"C0"/f"eval_{u:05d}"/"trajectory_metrics.csv")}; wins={m:sum(f(a[k],m)<f(d[k],m) for k in a) for m in METRICS}
            ts.append({"arm":arm,"update":u,**{m+"_trajectory_wins":wins[m] for m in METRICS},"all_three_wins":sum(all(f(a[k],m)<f(d[k],m) for m in METRICS) for k in a),"rel_mvpe_wins_tke_degradation_le_2pct":sum(f(a[k],"rel_l2")<f(d[k],"rel_l2") and f(a[k],"mvpe")<f(d[k],"mvpe") and f(a[k],"tke")<=1.02*f(d[k],"tke") for k in a)})
            ha=rows(root/arm/f"eval_{u:05d}"/"horizon_metrics.csv"); hc=rows(root/"C0"/f"eval_{u:05d}"/"horizon_metrics.csv")
            for band,(lo,hi) in BANDS.items():
                aa=[x for x in ha if lo<=int(x["horizon"])<=hi]; cc=[x for x in hc if lo<=int(x["horizon"])<=hi]; bs.append({"arm":arm,"update":u,"band":band,**{m+"_improvement_pct":improvement(sum(f(x,m) for x in cc)/len(cc),sum(f(x,m) for x in aa)/len(aa)) for m in METRICS}})
    write(out/f"{prefix}_horizon_delta.csv",hd); write(out/f"{prefix}_trajectory_summary.csv",ts); write(out/f"{prefix}_horizon_band_summary.csv",bs)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--r1-root",type=Path); p.add_argument("--r2-root",type=Path); p.add_argument("--out-dir",type=Path,required=True); a=p.parse_args()
    if a.r1_root: summarize(a.r1_root,"r1",a.out_dir)
    if a.r2_root: summarize(a.r2_root,"r2",a.out_dir)
if __name__=="__main__": main()
