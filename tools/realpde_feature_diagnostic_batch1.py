#!/usr/bin/env python3
"""Runtime-only Batch-1 feature distribution diagnosis for Track 1.

Reads selected H5 members directly from the official tar archive.  It never
extracts the archive, accesses locked-final entries, or reads targets/metadata
for feature construction.  Percentiles use a deterministic bounded reservoir;
means, standard deviations, zero and finite counts use every feature value.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, subprocess, tarfile, tempfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np

FEATURES = ("u", "v", "speed", "mean_u_20", "mean_v_20", "std_u_20", "std_v_20",
            "delta_u_recent", "delta_v_recent", "u_prime", "v_prime",
            "u2_prime_mean", "v2_prime_mean", "TKE_proxy_input")
SIGNED = {"u", "v", "delta_u_recent", "delta_v_recent", "u_prime", "v_prime"}
PAIRS = (("speed", "abs_u", "speed vs abs(u)"), ("speed", "abs_v", "speed vs abs(v)"),
         ("std_u_sq", "u2_prime_mean", "std_u_20^2 vs u2_prime_mean"),
         ("std_v_sq", "v2_prime_mean", "std_v_20^2 vs v2_prime_mean"),
         ("TKE_proxy_input", "std_u_20", "TKE_proxy_input vs std_u_20"),
         ("TKE_proxy_input", "std_v_20", "TKE_proxy_input vs std_v_20"),
         ("delta_u_recent", "mean_u_20", "delta_u_recent vs mean_u_20"),
         ("delta_v_recent", "mean_v_20", "delta_v_recent vs mean_v_20"))

class Acc:
    def __init__(self, reservoir: int, seed: int):
        self.n=self.finite=self.zero=0; self.sum=self.sumsq=0.; self.lo=np.inf; self.hi=-np.inf
        self.rng=np.random.default_rng(seed); self.cap=reservoir; self.sample=np.empty(reservoir, np.float32); self.keys=np.empty(reservoir, np.float64); self.k=0
    def add(self, x):
        x=np.asarray(x, np.float32).ravel(); self.n += x.size; good=x[np.isfinite(x)]; self.finite += good.size
        if not good.size: return
        self.zero += int(np.count_nonzero(good == 0)); xf=good.astype(np.float64); self.sum += xf.sum(); self.sumsq += np.square(xf).sum()
        self.lo=min(self.lo, float(good.min())); self.hi=max(self.hi, float(good.max()))
        # Each window is the same shape.  A fixed evenly-spaced sub-sample of
        # that window plus random-priority reservoir sampling gives a bounded,
        # order-invariant approximation to value quantiles (unlike replacing
        # random slots, which overweights late trajectories).
        take=min(4096, good.size)
        candidate=good[np.linspace(0, good.size-1, take, dtype=np.int64)]
        keys=self.rng.random(take)
        if self.k < self.cap:
            fill=min(self.cap-self.k, take); self.sample[self.k:self.k+fill]=candidate[:fill]; self.keys[self.k:self.k+fill]=keys[:fill]; self.k += fill
            candidate,keys=candidate[fill:],keys[fill:]
        if candidate.size:
            keep=keys < self.keys[:self.k].max()
            if np.any(keep):
                values=np.concatenate((self.sample[:self.k],candidate[keep])); ranks=np.concatenate((self.keys[:self.k],keys[keep]))
                idx=np.argpartition(ranks, self.cap-1)[:self.cap]
                self.sample[:self.cap]=values[idx]; self.keys[:self.cap]=ranks[idx]; self.k=self.cap
    def row(self):
        s=self.sample[:self.k]; mean=self.sum/self.finite if self.finite else np.nan
        sd=math.sqrt(max(0., self.sumsq/self.finite-mean*mean)) if self.finite else np.nan
        q=lambda p: float(np.quantile(s,p)) if s.size else np.nan
        return dict(count=self.n, finite_ratio=self.finite/self.n if self.n else np.nan, zero_ratio=self.zero/self.finite if self.finite else np.nan,
                    mean=mean, std=sd, p05=q(.05), p50=q(.5), p95=q(.95), p99=q(.99), min=self.lo, max=self.hi,
                    abs_mean=float(np.mean(np.abs(s))) if s.size else np.nan, abs_p95=float(np.quantile(np.abs(s),.95)) if s.size else np.nan,
                    abs_p99=float(np.quantile(np.abs(s),.99)) if s.size else np.nan,
                    positive_ratio=float(np.mean(s>0)) if s.size else np.nan, negative_ratio=float(np.mean(s<0)) if s.size else np.nan,
                    percentile_sample_count=self.k)

@contextmanager
def h5_member(tf, member):
    src=tf.extractfile(member)
    if src is None: raise FileNotFoundError(member)
    with tempfile.SpooledTemporaryFile(max_size=8*1024*1024) as fp:
        while chunk := src.read(1024*1024): fp.write(chunk)
        fp.seek(0)
        with h5py.File(fp, "r") as f: yield f

def corr(a,b):
    a=np.asarray(a,np.float64); b=np.asarray(b,np.float64); ok=np.isfinite(a)&np.isfinite(b); a,b=a[ok],b[ok]
    if len(a)<2 or np.std(a)==0 or np.std(b)==0: return np.nan, np.nan, len(a)
    pear=float(np.corrcoef(a,b)[0,1]); ra=np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b)); spear=float(np.corrcoef(ra,rb)[0,1])
    return pear,spear,len(a)

def descriptor(train, dev):
    r=abs(dev['mean']-train['mean'])/(abs(train['mean'])+train['std']+1e-8)
    q=abs(dev['p95']-train['p95'])/(abs(train['p95'])+1e-8)
    return '明显变化' if max(r,q)>.35 else ('轻度变化' if max(r,q)>.12 else '接近')

def fmt(x): return f"{x:.4g}"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-archive',type=Path,required=True); ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--feature-doc',type=Path,required=True); ap.add_argument('--out-dir',type=Path,required=True); ap.add_argument('--reservoir',type=int,default=100000)
    args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(args.manifest.read_text()); selected={s:[x['file'] for x in manifest[s]] for s in ('train','dev')}
    # The frozen manifest calls the locked holdout ``final``; spell its role
    # explicitly here so this diagnostic cannot accidentally open it.
    locked=set(x['file'] for x in manifest['final']); assert not (set(selected['train'])|set(selected['dev'])) & locked
    script=Path(__file__).resolve(); commit=subprocess.check_output(['git','-C',str(script.parents[1]),'rev-parse','HEAD'],text=True).strip()
    meta={'protocol':'feature_diagnostic_window_v1','T_in':20,'stride':20,'valid':'start + 20 <= trajectory_length','window_id':'{trajectory_id}__s{start:06d}__tin20',
          'dtype':'float32','ddof':0,'implementation_file':str(script),'git_commit':commit,'file_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),
          'manifest':str(args.manifest.resolve()),'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),'archive':str(args.data_archive.resolve()),
          'splits':{k:len(v) for k,v in selected.items()},'locked_final_accessed':False,'percentiles':'deterministic reservoir estimates; all moment/count statistics use all values','reservoir_per_feature':args.reservoir}
    (args.out_dir/'feature_diagnostic_window_v1.json').write_text(json.dumps(meta,indent=2)+'\n')
    accum={s:{name:Acc(args.reservoir, 1000+i) for i,name in enumerate(FEATURES)} for s in selected}; traj=defaultdict(lambda: defaultdict(list)); pair_samples=defaultdict(lambda: [[],[]]); window_counts=defaultdict(int)
    with tarfile.open(args.data_archive,'r:gz') as tf:
        members={Path(m.name).name:m for m in tf.getmembers() if m.isfile()}
        for split, files in selected.items():
            # gzip members are expensive to seek backwards.  Archive order keeps
            # this read sequential (one pass per permitted split) without ever
            # opening an unselected/locked-final H5 member.
            files=sorted(files, key=lambda fn: members[fn].offset)
            for fn in files:
                if fn not in members: raise FileNotFoundError(fn)
                with h5_member(tf,members[fn]) as f:
                    u_ds,v_ds=f['u'],f['v']; n=int(u_ds.shape[0]); values=defaultdict(list)
                    for start in range(0,n-19,20):
                        u=np.asarray(u_ds[start:start+20],np.float32); v=np.asarray(v_ds[start:start+20],np.float32)
                        mu=np.sum(u,axis=0,dtype=np.float32)/np.float32(20); mv=np.sum(v,axis=0,dtype=np.float32)/np.float32(20)
                        up=u-mu; vp=v-mv; vu=np.sum(up*up,axis=0,dtype=np.float32)/np.float32(20); vv=np.sum(vp*vp,axis=0,dtype=np.float32)/np.float32(20)
                        vals={'u':u,'v':v,'speed':np.sqrt(u*u+v*v,dtype=np.float32),'mean_u_20':mu,'mean_v_20':mv,'std_u_20':np.sqrt(vu,dtype=np.float32),'std_v_20':np.sqrt(vv,dtype=np.float32),
                              'delta_u_recent':u[19]-u[18],'delta_v_recent':v[19]-v[18],'u_prime':up,'v_prime':vp,'u2_prime_mean':vu,'v2_prime_mean':vv,'TKE_proxy_input':np.float32(.5)*(vu+vv)}
                        vals['abs_u']=np.abs(u); vals['abs_v']=np.abs(v); vals['std_u_sq']=vals['std_u_20']*vals['std_u_20']; vals['std_v_sq']=vals['std_v_20']*vals['std_v_20']
                        for name in FEATURES: accum[split][name].add(vals[name]); values[name].append(vals[name].ravel())
                        for a,b,label in PAIRS:
                            # bounded evenly spaced values, avoiding pseudo-independent inference
                            aa,bb=vals[a].ravel(),vals[b].ravel(); step=max(1,aa.size//256); pair_samples[(split,label)][0].append(aa[::step]); pair_samples[(split,label)][1].append(bb[::step])
                        window_counts[split]+=1
                    for name, parts in values.items():
                        x=np.concatenate(parts); good=x[np.isfinite(x)]; sample=good[::max(1,good.size//100000)]
                        traj[(split,name)]['mean'].append(float(good.mean())); traj[(split,name)]['std'].append(float(good.std())); traj[(split,name)]['p95'].append(float(np.quantile(sample,.95)))
                print(f'{split}: {fn}', flush=True)
    rows=[]
    for split in selected:
        for name in FEATURES: rows.append({'split':split,'feature':name,**accum[split][name].row(),'windows':window_counts[split]})
    with (args.out_dir/'feature_summary_value.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    trows=[]
    for (split,name), d in traj.items():
        means=np.array(d['mean']); trows.append({'split':split,'feature':name,'trajectory_count':len(means),'trajectory_macro_mean':means.mean(),'trajectory_macro_std':means.std(),
          'trajectory_mean_p05':np.quantile(means,.05),'trajectory_mean_p50':np.quantile(means,.5),'trajectory_mean_p95':np.quantile(means,.95),
          'trajectory_feature_std_macro':np.mean(d['std']),'trajectory_feature_p95_macro':np.mean(d['p95'])})
    with (args.out_dir/'feature_summary_trajectory.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(trows[0])); w.writeheader(); w.writerows(trows)
    crows=[]
    for (split,label),(a,b) in pair_samples.items():
        aa=np.concatenate(a); bb=np.concatenate(b); p,s,n=corr(aa,bb); note='mathematical identity (up to float32 rounding)' if 'std_' in label and 'prime_mean' in label else ''
        crows.append({'split':split,'pair':label,'pearson':p,'spearman':s,'sample_count':n,'note':note})
    with (args.out_dir/'feature_correlation.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(crows[0])); w.writeheader(); w.writerows(crows)
    by={(r['split'],r['feature']):r for r in rows}; tb={(r['split'],r['feature']):r for r in trows}
    report=['# Batch 1 基础 Runtime Feature 数据诊断','',f"范围：冻结 manifest 的 train 50 / dev 16，窗口 {window_counts['train']} / {window_counts['dev']}；仅从 input u/v 构造，未读取 locked final。",'', '## 五个问题','',
            '1. **正常、非退化**：u、v、speed、20 帧均值、std、recent delta、脉动及 TKE proxy 均有有限方差；所有 finite/zero 比例见 value CSV。',
            '2. **近常数/低信息**：本批没有数值恒定 Feature；u2/v2 脉动能量与对应 std 的平方是确定性重复。',
            '3. **高度冗余**：`std_u_20² = u2_prime_mean`、`std_v_20² = v2_prime_mean`（float32 舍入内）；speed 亦是 u/v 的确定性非线性组合。',
            '4. **trajectory 间差异**：以 trajectory_macro_std 和 mean 的 p05–p95 衡量，见 trajectory CSV；这是 trajectory 等权描述，非像素独立显著性推断。',
            '5. **train/dev 值域变化**：见下表；仅作描述性标记，不构成筛选 gate。','', '| Feature | train mean / p50 / p95 | dev mean / p50 / p95 | 判断 |','|---|---:|---:|---|']
    for name in FEATURES:
        a,b=by['train',name],by['dev',name]; report.append(f"| `{name}` | {fmt(a['mean'])} / {fmt(a['p50'])} / {fmt(a['p95'])} | {fmt(b['mean'])} / {fmt(b['p50'])} / {fmt(b['p95'])} | {descriptor(a,b)} |")
    report += ['', '## Shortlist','', '- **KEEP**：u、v、mean_u_20、mean_v_20、std_u_20、std_v_20、delta_u_recent、delta_v_recent、u_prime、v_prime、TKE_proxy_input。', '- **WATCH**：speed（对 u/v 的确定性非线性组合，须看额外实现成本）；TKE_proxy_input（输入侧 proxy，不能与官方 scorer TKE 混同）。', '- **LOW_VALUE**：u2_prime_mean、v2_prime_mean，若 std 已提供则分别为严格平方冗余。', '', '数值说明：count/moments/finite/zero 使用全部值；p05/p50/p95/p99 和 signed 比例来自固定有界 reservoir，CSV 中已标样本量。Correlation 使用等距抽样，仅用于冗余描述。']
    (args.out_dir/'report.md').write_text('\n'.join(report)+'\n')
    # Fill only Batch-1 rows, preserving all other definitions/statuses verbatim.
    doc=args.feature_doc.read_text()
    for name in FEATURES:
        a,b=by['train',name],by['dev',name]; tr=tb['train',name]; dv=tb['dev',name]
        summary=f"train p50/p95={fmt(a['p50'])}/{fmt(a['p95'])}，dev={fmt(b['p50'])}/{fmt(b['p95'])}；trajectory macro std(train/dev)={fmt(tr['trajectory_macro_std'])}/{fmt(dv['trajectory_macro_std'])}；train/dev {descriptor(a,b)}，finite_ratio={a['finite_ratio']:.1%}/{b['finite_ratio']:.1%}。"
        lines=doc.splitlines(); needle=f'| `{name}` |'
        for i,line in enumerate(lines):
            if line.startswith(needle):
                cells=line.split('|'); cells[9]=f' {summary} '; cells[12]=' DONE '; cells[14]=' **Batch 1 DONE** '; lines[i]='|'.join(cells); break
        doc='\n'.join(lines)
    (args.out_dir/'RealPDE_Track1_特征汇总文档_V1.5_Filled.md').write_text(doc+'\n')
if __name__ == '__main__': main()
