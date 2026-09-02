#!/usr/bin/env python3
"""Streaming pixel-space primitive-gradient diagnosis for Track 1.

Only the current 20-frame u/v input is read.  The four primitive derivatives
use first-order one-sided differences on the outer image edge and centered
differences in the interior; no smoothing, clipping, coordinates or masks are
introduced.  H5 members are read directly from the official tar archive.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, subprocess, tarfile, tempfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import h5py
import numpy as np

FEATURES=("du_dx_pixel","du_dy_pixel","dv_dx_pixel","dv_dy_pixel","vorticity_pixel")
PRIMITIVES=FEATURES[:4]

class Acc:
    def __init__(self, cap:int, seed:int):
        self.n=self.finite=self.zero=0; self.sum=self.sumsq=0.; self.lo=np.inf; self.hi=-np.inf
        self.cap=cap; self.k=0; self.rng=np.random.default_rng(seed)
        self.sample=np.empty(cap,np.float32); self.keys=np.empty(cap,np.float64)
    def add(self, x):
        x=np.asarray(x,np.float32).ravel(); self.n+=x.size; good=x[np.isfinite(x)]; self.finite+=good.size
        if not good.size: return
        xf=good.astype(np.float64); self.sum+=xf.sum(); self.sumsq+=np.square(xf).sum(); self.zero+=int(np.count_nonzero(good==0))
        self.lo=min(self.lo,float(good.min())); self.hi=max(self.hi,float(good.max()))
        take=min(4096,good.size); vals=good[np.linspace(0,good.size-1,take,dtype=np.int64)]; keys=self.rng.random(take)
        if self.k<self.cap:
            fill=min(self.cap-self.k,take); self.sample[self.k:self.k+fill]=vals[:fill]; self.keys[self.k:self.k+fill]=keys[:fill]; self.k+=fill; vals,keys=vals[fill:],keys[fill:]
        if vals.size and self.k:
            keep=keys<self.keys[:self.k].max()
            if np.any(keep):
                allv=np.concatenate((self.sample[:self.k],vals[keep])); allk=np.concatenate((self.keys[:self.k],keys[keep]))
                ix=np.argpartition(allk,min(self.cap,allk.size)-1)[:self.cap]; self.sample[:len(ix)]=allv[ix]; self.keys[:len(ix)]=allk[ix]; self.k=len(ix)
    def row(self):
        s=self.sample[:self.k]; q=lambda p:float(np.quantile(s,p)) if s.size else np.nan
        mean=self.sum/self.finite if self.finite else np.nan; sd=math.sqrt(max(0.,self.sumsq/self.finite-mean*mean)) if self.finite else np.nan
        return dict(count=self.n,finite_ratio=self.finite/self.n if self.n else np.nan,zero_ratio=self.zero/self.finite if self.finite else np.nan,mean=mean,std=sd,p05=q(.05),p50=q(.5),p95=q(.95),p99=q(.99),min=self.lo,max=self.hi,abs_mean=float(np.mean(np.abs(s))) if s.size else np.nan,abs_p95=float(np.quantile(np.abs(s),.95)) if s.size else np.nan,abs_p99=float(np.quantile(np.abs(s),.99)) if s.size else np.nan,percentile_sample_count=self.k)

@contextmanager
def h5_member(tf, member):
    src=tf.extractfile(member)
    if src is None: raise FileNotFoundError(member)
    with tempfile.SpooledTemporaryFile(max_size=8*1024*1024) as fp:
        while chunk:=src.read(1024*1024): fp.write(chunk)
        fp.seek(0)
        with h5py.File(fp,"r") as f: yield f

def derivative(x, axis):
    out=np.empty_like(x,dtype=np.float32)
    if axis==2: # x / columns
        out[:,:,0]=(x[:,:,1]-x[:,:,0]); out[:,:,-1]=(x[:,:,-1]-x[:,:,-2]); out[:,:,1:-1]=(x[:,:,2:]-x[:,:,:-2])/np.float32(2.)
    else: # y / rows
        out[:,0,:]=(x[:,1,:]-x[:,0,:]); out[:,-1,:]=(x[:,-1,:]-x[:,-2,:]); out[:,1:-1,:]=(x[:,2:,:]-x[:,:-2,:])/np.float32(2.)
    return out

def corr(a,b):
    a=np.asarray(a,np.float64); b=np.asarray(b,np.float64); ok=np.isfinite(a)&np.isfinite(b); a,b=a[ok],b[ok]
    if len(a)<2 or np.std(a)==0 or np.std(b)==0:return np.nan,np.nan,len(a)
    return float(np.corrcoef(a,b)[0,1]),float(np.corrcoef(np.argsort(np.argsort(a)),np.argsort(np.argsort(b)))[0,1]),len(a)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-archive',type=Path,required=True); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--out-dir',type=Path,required=True); ap.add_argument('--reservoir',type=int,default=100000)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True); manifest=json.loads(a.manifest.read_text()); selected={s:[x['file'] for x in manifest[s]] for s in ('train','dev')}; locked={x['file'] for x in manifest['final']}; assert not (set(selected['train'])|set(selected['dev']))&locked
    script=Path(__file__).resolve(); repo=script.parents[1]; commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    meta={'protocol':'spatial_diagnostic_batch1_v1','definition':'pixel-space finite difference, spacing=1 pixel','edge_rule':'first-order forward at index 0, first-order backward at last index; centered difference elsewhere; no smoothing/clipping/normalization','T_in':20,'stride':20,'valid':'start + 20 <= trajectory_length','dtype':'float32','ddof':0,'implementation_file':str(script),'git_commit':commit,'file_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'splits':{k:len(v) for k,v in selected.items()},'locked_final_accessed':False,'features':list(FEATURES)}
    (a.out_dir/'spatial_definition_v1.json').write_text(json.dumps(meta,indent=2)+'\n')
    acc={s:{f:Acc(a.reservoir,100+i) for i,f in enumerate(FEATURES)} for s in selected}; traj=defaultdict(lambda:defaultdict(list)); edge_acc={s:{f:{r:Acc(50000,900+i) for i,r in enumerate(('outer_edge','interior'))} for f in PRIMITIVES} for s in selected}; pairs=defaultdict(lambda:[[],[]]); windows=defaultdict(int)
    with tarfile.open(a.data_archive,'r:gz') as tf:
        members={Path(m.name).name:m for m in tf.getmembers() if m.isfile()}
        for split,files0 in selected.items():
            for fn in sorted(files0,key=lambda z:members[z].offset):
                # Per-trajectory accumulators keep trajectory-level statistics
                # without retaining all windows in memory.
                local={name:Acc(min(a.reservoir,100000),700+i) for i,name in enumerate(FEATURES)}
                with h5_member(tf,members[fn]) as f:
                    u_ds,v_ds=f['u'],f['v']; length=int(u_ds.shape[0]); h,w=int(u_ds.shape[1]),int(u_ds.shape[2]); edge=np.zeros((h,w),bool); edge[0,:]=edge[-1,:]=True; edge[:,0]=edge[:,-1]=True
                    for start in range(0,length-19,20):
                        u=np.asarray(u_ds[start:start+20],np.float32); v=np.asarray(v_ds[start:start+20],np.float32); du_dx=derivative(u,2); du_dy=derivative(u,1); dv_dx=derivative(v,2); dv_dy=derivative(v,1); vort=dv_dx-du_dy; vals={'du_dx_pixel':du_dx,'du_dy_pixel':du_dy,'dv_dx_pixel':dv_dx,'dv_dy_pixel':dv_dy,'vorticity_pixel':vort}
                        for name,x in vals.items(): acc[split][name].add(x); local[name].add(x)
                        for name in PRIMITIVES:
                            x=vals[name]; edge_acc[split][name]['outer_edge'].add(x[:,edge]); edge_acc[split][name]['interior'].add(x[:,~edge])
                        # Small deterministic paired sample for redundancy diagnostics.
                        for left,right,label in [('vorticity_pixel','vorticity_pixel','vorticity vs (dv_dx-du_dy)'),('vorticity_pixel','du_dy_pixel','vorticity vs du_dy'),('vorticity_pixel','dv_dx_pixel','vorticity vs dv_dx')]:
                            x=vals[left].ravel(); y=(dv_dx-du_dy).ravel() if label.startswith('vorticity vs (') else vals[right].ravel(); step=max(1,x.size//512); pairs[(split,label)][0].append(x[::step]); pairs[(split,label)][1].append(y[::step])
                        windows[split]+=1
                    for name,stat in local.items():
                        row=stat.row(); traj[(split,name)]['mean'].append(float(row['mean'])); traj[(split,name)]['std'].append(float(row['std'])); traj[(split,name)]['p95'].append(float(row['p95']))
                print(f'{split}: {fn}',flush=True)
    rows=[]
    for split in selected:
        for name in FEATURES: rows.append({'split':split,'feature':name,**acc[split][name].row(),'windows':windows[split]})
    with (a.out_dir/'spatial_summary_value.csv').open('w',newline='') as f: writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    trows=[]
    for (split,name),d in traj.items():
        m=np.array(d['mean']); trows.append({'split':split,'feature':name,'trajectory_count':len(m),'trajectory_macro_mean':m.mean(),'trajectory_macro_std':m.std(),'trajectory_mean_p05':np.quantile(m,.05),'trajectory_mean_p50':np.quantile(m,.5),'trajectory_mean_p95':np.quantile(m,.95),'trajectory_feature_std_macro':np.mean(d['std']),'trajectory_feature_p95_macro':np.mean(d['p95'])})
    with (a.out_dir/'spatial_summary_trajectory.csv').open('w',newline='') as f: writer=csv.DictWriter(f,fieldnames=list(trows[0])); writer.writeheader(); writer.writerows(trows)
    erows=[]
    for split in selected:
        for name in PRIMITIVES:
            x={r:edge_acc[split][name][r].row() for r in ('outer_edge','interior')}; erows.append({'split':split,'feature':name,'outer_edge_count':x['outer_edge']['count'],'interior_count':x['interior']['count'],'outer_abs_mean':x['outer_edge']['abs_mean'],'interior_abs_mean':x['interior']['abs_mean'],'outer_abs_p95':x['outer_edge']['abs_p95'],'interior_abs_p95':x['interior']['abs_p95'],'outer_abs_p99':x['outer_edge']['abs_p99'],'interior_abs_p99':x['interior']['abs_p99'],'abs_mean_ratio_edge_over_interior':x['outer_edge']['abs_mean']/max(x['interior']['abs_mean'],1e-30),'abs_p95_ratio_edge_over_interior':x['outer_edge']['abs_p95']/max(x['interior']['abs_p95'],1e-30)})
    with (a.out_dir/'spatial_edge_summary.csv').open('w',newline='') as f: writer=csv.DictWriter(f,fieldnames=list(erows[0])); writer.writeheader(); writer.writerows(erows)
    crows=[]
    for (split,label),(aa,bb) in pairs.items():
        p,s,n=corr(np.concatenate(aa),np.concatenate(bb)); crows.append({'split':split,'pair':label,'pearson':p,'spearman':s,'sample_count':n,'note':'deterministic derived identity' if 'dv_dx-du_dy' in label else 'derived-vs-primitive descriptive correlation'})
    with (a.out_dir/'spatial_correlation.csv').open('w',newline='') as f: writer=csv.DictWriter(f,fieldnames=list(crows[0])); writer.writeheader(); writer.writerows(crows)
    by={(r['split'],r['feature']):r for r in rows}; eb={(r['split'],r['feature']):r for r in erows}; tb={(r['split'],r['feature']):r for r in trows}
    def change(name):
        x,y=by['train',name],by['dev',name]; d=max(abs(float(y['p95'])-float(x['p95']))/(abs(float(x['p95']))+1e-12),abs(float(y['mean'])-float(x['mean']))/(abs(float(x['mean']))+float(x['std'])+1e-12)); return '明显变化' if d>.35 else ('轻度变化' if d>.12 else '接近')
    report=['# Spatial Batch 1 基础特征数据诊断','',f"范围：冻结 manifest 的 train 50 / dev 16，窗口 {windows['train']} / {windows['dev']}；H×W={h}×{w}；仅输入 u/v，未读取 locked final。",'', '## Definition','', '`du_dx_pixel`/`dv_dx_pixel`：列方向，边缘 forward/backward、内部 centered；`du_dy_pixel`/`dv_dy_pixel`：行方向，同样 edge rule；spacing=1 pixel，float32，无 smoothing/clipping/normalization。`vorticity_pixel = dv_dx_pixel - du_dy_pixel`。','', '## 结果','', '| Feature | train mean / p95 | dev mean / p95 | train/dev |','|---|---:|---:|---|']
    for name in FEATURES: report.append(f"| `{name}` | {float(by['train',name]['mean']):.4g} / {float(by['train',name]['p95']):.4g} | {float(by['dev',name]['mean']):.4g} / {float(by['dev',name]['p95']):.4g} | {change(name)} |")
    report += ['', '## Edge sensitivity','', '详见 `spatial_edge_summary.csv`。outer edge 定义为首/末行或首/末列，interior 为其余像素；仅比较 abs_mean、abs_p95、abs_p99，不做显著性硬 Gate。','', '## 五个问题', '', '1. 四个 primitive gradient 的整体稳定性、轨迹级差异见 value/trajectory CSV；finite、零比例和 raw range 均有记录。', '2. edge 与 interior 的绝对分布通过 edge CSV 单列比较；edge 放大则列为 WATCH，未出现放大则不把 image edge 当作主要风险。', '3. train/dev 只作描述性尺度比较，不据此选择阈值或归一化。', '4. `vorticity_pixel` 是 `dv_dx_pixel - du_dy_pixel` 的确定性 derived summary，不是新增原始信息；相关性 CSV 显式记录该身份关系。', '', '## Shortlist', '', '- **KEEP / WATCH / LOW_VALUE**：每个 feature 的最终标记见下表。']
    for name in FEATURES:
        e1,e2=eb['train',name] if name in PRIMITIVES else None,None
        if name=='vorticity_pixel': label='WATCH' if change(name)!='接近' else 'KEEP'
        else:
            e=eb['train',name]; ratio=float(e['abs_p95_ratio_edge_over_interior']); label='WATCH' if ratio>1.25 else 'KEEP'
        report.append(f'- `{name}`: **{label}**。')
    report += ['', '统计说明：value-level moments/counts 使用全部值；分位数使用固定有界 priority reservoir 并在 CSV 标出样本数。trajectory-level 先逐轨迹统计后等权汇总。']
    (a.out_dir/'report.md').write_text('\n'.join(report)+'\n')
if __name__=='__main__': main()
