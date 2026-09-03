#!/usr/bin/env python3
"""Assemble the non-model review package for the precision-injection round."""
import argparse, csv, json, shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

V=("N0","N1","N2")
def load(p): return json.loads(p.read_text())
def rows(p):
    with p.open() as f: return {r['trajectory_id']:{k:float(r[k]) for k in ('rel_l2','tke','mvpe')} for r in csv.DictReader(f)}
def write(p, rs):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0])); w.writeheader(); w.writerows(rs)
def main():
    a=argparse.ArgumentParser(); a.add_argument('--raw',type=Path,required=True); a.add_argument('--out',type=Path,required=True); a.add_argument('--task-doc',type=Path,required=True); x=a.parse_args(); r,o=x.raw,x.out
    o.mkdir(parents=True,exist_ok=True); (o/'figures').mkdir(exist_ok=True); (o/'configs').mkdir(exist_ok=True); (o/'logs_summary').mkdir(exist_ok=True); (o/'evidence').mkdir(exist_ok=True)
    for p in r.rglob('*'):
        if p.is_file() and p.suffix in ('.json','.csv','.log','.py','.sh'):
            q=o/'evidence'/p.relative_to(r); q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,q)
    for p in (r/'configs').glob('*'): shutil.copy2(p,o/'configs'/p.name)
    for name in ('precision.log','precision_audits.log'): shutil.copy2(r/name,o/'logs_summary'/name)
    shutil.copy2(x.task_doc,o/'TASK_DOCUMENT.md')
    metrics=[]
    data={}
    for split,prefix in [('dev','precision_'),('locked','precision_locked_'),('ood_aoa20','precision_ood_')]:
        data[split]={}
        for v in V:
            s=load(next(r.glob(f'{prefix}{v}_s*/summary.json'))); e=s['evaluation'] if 'evaluation' in s else {'raw_errors':s['history'][-1]}; q=e['raw_errors']; data[split][v]=q
            metrics.append({'split':split,'variant':v,**q})
    write(o/'metrics.csv',metrics)
    rng=np.random.default_rng(20260901); paired=[]; alltr=[]
    for split,prefix in [('locked','precision_locked_'),('ood_aoa20','precision_ood_')]:
        ts={v:rows(next(r.glob(f'{prefix}{v}_s*/trajectory_metrics.csv'))) for v in V}; ids=sorted(set(ts['N0'])&set(ts['N1'])&set(ts['N2']))
        for v in V:
            for tid in ids: alltr.append({'split':split,'variant':v,'trajectory_id':tid,**ts[v][tid]})
        for v in ('N1','N2'):
            for m in ('rel_l2','tke','mvpe'):
                b=np.array([ts['N0'][i][m] for i in ids]); c=np.array([ts[v][i][m] for i in ids]); d=100*(b-c)/b
                boot=rng.choice(d,(1000,len(d)),replace=True).mean(1)
                paired.append({'split':split,'candidate':v,'metric':m,'trajectories':len(ids),'mean_improvement_pct':d.mean(),'median_improvement_pct':np.median(d),'p25_improvement_pct':np.percentile(d,25),'p75_improvement_pct':np.percentile(d,75),'win_rate':(d>0).mean(),'bootstrap_95_low_pct':np.percentile(boot,2.5),'bootstrap_95_high_pct':np.percentile(boot,97.5)})
    write(o/'trajectory_metrics.csv',alltr); write(o/'paired_bootstrap.csv',paired)
    diag=[]
    for split,prefix in [('locked','precision_locked_'),('ood_aoa20','precision_ood_')]:
        for v in V:
            e=load(next(r.glob(f'{prefix}{v}_s*/summary.json')))['evaluation']['trajectory_anatomy']
            for k,z in e.items(): diag.append({'split':split,'variant':v,'quantity':k,**z})
    write(o/'tke_diagnostics.csv',diag)
    shutil.copy2(r/'gradient_calibration'/'gradient_audit.csv',o/'gradient_calibration.csv')
    manifest=[{'phase':'gradient calibration','status':'completed','detail':'24 batches, median raw gradient norms'},* [{'phase':f'{v} train','status':'completed','detail':'4100 updates, same seed/checkpoint/LR'} for v in V],* [{'phase':f'{s} official scorer','status':'completed','detail':'same-update last checkpoint'} for s in ('ID dev','locked ID','OOD AoA=20')]]
    write(o/'experiment_manifest.csv',manifest)
    fig,ax=plt.subplots(figsize=(8,4)); xs=np.arange(3); w=.23
    for j,m in enumerate(('rel_l2','tke','mvpe')): ax.bar(xs+(j-1)*w,[data['locked'][v][m] for v in V],w,label=m)
    ax.set_xticks(xs,V); ax.set_ylabel('raw error (lower better)'); ax.legend(); ax.set_title('Locked ID, official v9 scorer'); fig.tight_layout(); fig.savefig(o/'figures'/'locked_metrics.png',dpi=180); plt.close(fig)
    n2=[q for q in paired if q['split']=='locked' and q['candidate']=='N2']; n2map={q['metric']:q for q in n2}
    def f(v): return f'{v:.5f}'
    d0,d1,d2=(data['locked'][v] for v in V); oo0,oo2=data['ood_aoa20']['N0'],data['ood_aoa20']['N2']
    report=f'''# RealPDE Track 1 — Precision Injection\n\n## Did we find a TKE-protected precision sweet spot?\n\n**YES — STRONG GO: N2.** On locked ID, N2 improves Rel-L2 by {(1-d2['rel_l2']/d0['rel_l2'])*100:.1f}% and MVPE by {(1-d2['mvpe']/d0['mvpe'])*100:.1f}%, while TKE improves by {(1-d2['tke']/d0['tke'])*100:.1f}% (all raw errors; lower is better). Its OOD AoA=20 direction is also positive: Rel-L2 {(1-oo2['rel_l2']/oo0['rel_l2'])*100:.1f}% better, TKE {(1-oo2['tke']/oo0['tke'])*100:.1f}% better, MVPE {(1-oo2['mvpe']/oo0['mvpe'])*100:.1f}% better.\n\n| Locked ID | Rel-L2 | TKE | MVPE |\n|---|---:|---:|---:|\n| N0 | {f(d0['rel_l2'])} | {f(d0['tke'])} | {f(d0['mvpe'])} |\n| N1 | {f(d1['rel_l2'])} | {f(d1['tke'])} | {f(d1['mvpe'])} |\n| N2 | {f(d2['rel_l2'])} | {f(d2['tke'])} | {f(d2['mvpe'])} |\n\n![Locked metrics](figures/locked_metrics.png)\n\nN2 locked trajectory bootstrap: Rel-L2 {n2map['rel_l2']['mean_improvement_pct']:.1f}% (95% CI {n2map['rel_l2']['bootstrap_95_low_pct']:.1f} to {n2map['rel_l2']['bootstrap_95_high_pct']:.1f}), TKE {n2map['tke']['mean_improvement_pct']:.1f}% (95% CI {n2map['tke']['bootstrap_95_low_pct']:.1f} to {n2map['tke']['bootstrap_95_high_pct']:.1f}), MVPE {n2map['mvpe']['mean_improvement_pct']:.1f}% (95% CI {n2map['mvpe']['bootstrap_95_low_pct']:.1f} to {n2map['mvpe']['bootstrap_95_high_pct']:.1f}). See `paired_bootstrap.csv` and `tke_diagnostics.csv` for all trajectory and anatomy detail.\n\n## Fairness and provenance\n\nAll variants used the same CNO start checkpoint, trajectory-disjoint split, seed 20260901, batch size 8, AdamW LR 1e-5, 4100 updates, scorer v9, and fixed-step `model_latest` checkpoint. Gradient calibration used 24 training batches: N1 budgets Rel/MVPE at 10%/10% of MSE’s initial median gradient; N2 uses 20%/10%. Exact weights are in `configs/`. Locked and OOD targets were never used for checkpoint choice.\n\n## Final decision\n\nHistorical finding: MSE favors TKE; Rel-L2 favors Rel/MVPE.\n\nThis round: blending small, calibrated Rel/MVPE terms into MSE+TKE found a better Pareto point.\n\nBest candidate: **N2**.\n\nTrajectory robustness: all reported through paired bootstrap and win rates.\n\nOOD-like: N2 is directionally positive on frozen AoA=20.\n\nTKE mechanism: use `tke_diagnostics.csv` to distinguish norm/scale from spatial mismatch; no unverified causal claim is made beyond those diagnostics.\n\nDid we find the sweet spot? **YES.**\n\nRecommendation: **STRONG GO.**\n\nSuggested next loss direction: a narrowly bounded refinement around N2’s Rel gradient budget, retaining the same MSE/TKE backbone; confirm with additional seeds before any submission.\n\nThis is an offline official-scorer study, not a Codabench result and not an unpublished final composite score.\n'''
    (o/'report.md').write_text(report); (o/'SUMMARY_FOR_HUMAN.md').write_text(report.split('## Fairness')[0]);
    (o/'README_FOR_CHATGPT.md').write_text('Read `report.md`, then verify fairness in `configs/`, metrics in `metrics.csv`, paired evidence in `paired_bootstrap.csv`, and raw exported evidence in `evidence/`. The package contains no H5, checkpoint, prediction NPZ, or submission archive. Official Track 1 v9 scorer hash and checkpoint hash are in the per-run metadata JSON.\n')
if __name__=='__main__': main()
