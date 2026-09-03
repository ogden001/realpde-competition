#!/usr/bin/env python3
"""Bounded loss audit/ablation for the official-HDF5 CNO P0 control path."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from realpde_p0_data import H5WindowDataset, list_h5, split_paths

def rel(p: Tensor, y: Tensor) -> Tensor:
    return ((p-y).reshape(p.shape[0],-1).norm(dim=1)/y.reshape(y.shape[0],-1).norm(dim=1).clamp_min(1e-8)).mean()
def tke_map(x: Tensor) -> Tensor:
    z=x[...,:2]-x[...,:2].mean(dim=1,keepdim=True); return .5*z.square().mean(dim=1).sum(dim=-1)
def tke(p: Tensor,y: Tensor) -> Tensor: return rel(tke_map(p),tke_map(y))
def temporal(p: Tensor,y: Tensor) -> Tensor: return rel(p[:,1:,:,:, :2]-p[:,:-1,:,:, :2], y[:,1:,:,:, :2]-y[:,:-1,:,:, :2])
def spatial(p: Tensor,y: Tensor) -> Tensor:
    p,y=p[...,:2],y[...,:2]
    return .5*rel(p[:,:,:,1:]-p[:,:,:,:-1],y[:,:,:,1:]-y[:,:,:,:-1])+.5*rel(p[:,:,1:]-p[:,:,:-1],y[:,:,1:]-y[:,:,:-1])
def mean(p: Tensor,y: Tensor) -> Tensor: return rel(p[...,:2].mean(1),y[...,:2].mean(1))
def fluct(p: Tensor,y: Tensor) -> Tensor:
    p,y=p[...,:2],y[...,:2]; return rel(p-p.mean(1,keepdim=True),y-y.mean(1,keepdim=True))
def mvpe(p: Tensor,y: Tensor) -> Tensor:
    _,_,h,w,_=p.shape; ys=[q for q in range(16-2*4,16+2*5,2) if 0<=q<h]; terms=[]
    for x in [21,37,53,69]:
        if x<w and ys: terms.append(rel(p[:,:,ys,x,:2].mean(1),y[:,:,ys,x,:2].mean(1)))
    return torch.stack(terms).mean() if terms else p.new_tensor(0.)
def forward(model,x): return model(x.permute(0,4,1,2,3)).permute(0,2,3,4,1)
def parts(p,y):
    return {"rel":rel(p[...,:2],y[...,:2]),"mse":(p[...,:2]-y[...,:2]).square().mean(),"tke":tke(p,y),"mvpe":mvpe(p,y),"temporal":temporal(p,y),"spatial":spatial(p,y),"mean":mean(p,y),"fluct":fluct(p,y),"pressure":p[...,2].square().mean()}
def load_model(root: Path, ckpt: Path, device):
    sys.path.insert(0,str(root)); from realpdebench.model.cno import CNO3d
    model=CNO3d(in_dim=3,out_dim=3,out_dim_mult=1,in_size=64,N_layers=3).to(device)
    state=torch.load(ckpt,map_location="cpu"); state=state.get("model_state_dict",state); model.load_state_dict(state,strict=True); return model
def build(args):
    paths=list_h5(args.real_root); tr,va=split_paths(paths,args.val_fraction,args.seed)
    train=H5WindowDataset(tr,max_windows_per_trajectory=args.max_windows); val=H5WindowDataset(va,max_windows_per_trajectory=args.max_windows)
    return DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True),DataLoader(val,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True),va
def stats(a):
    a=np.asarray(a,float); return {"mean":float(a.mean()),"median":float(np.median(a)),"std":float(a.std()),"p10":float(np.percentile(a,10)),"p90":float(np.percentile(a,90))}
@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); sums={k:0. for k in ("rel","tke","mvpe")}; n=0
    for x,y,*_ in loader:
        x,y=x.to(device),y.to(device); z=parts(forward(model,x),y); b=x.shape[0]; n+=b
        for k in sums:sums[k]+=float(z[k])*b
    return {k:v/n for k,v in sums.items()}|{"n":n}
def audit(model,loader,device,out,max_batches,grad_batches,weights):
    rows=[]; g_rows=[]; cos=[]; kp=[]; ky=[]; params=[q for q in model.parameters() if q.requires_grad]
    for bi,(x,y,*_) in enumerate(loader):
        if bi>=max_batches: break
        x,y=x.to(device),y.to(device); model.eval(); pred=forward(model,x); pp=tke_map(pred).detach().cpu().numpy(); yy=tke_map(y).detach().cpu().numpy(); kp.append(pp);ky.append(yy)
        pf=pred[...,:2]-pred[...,:2].mean(1,keepdim=True); yf=y[...,:2]-y[...,:2].mean(1,keepdim=True)
        for i in range(x.shape[0]):
            a,b=pp[i].ravel(),yy[i].ravel(); alpha=float(a.dot(b)/max(a.dot(a),1e-12)); corr=float(np.corrcoef(a,b)[0,1]) if a.std() and b.std() else float("nan")
            ku=.5*pf[i,...,0].square().mean(0).detach().cpu().numpy(); kut=.5*yf[i,...,0].square().mean(0).detach().cpu().numpy(); kv=.5*pf[i,...,1].square().mean(0).detach().cpu().numpy(); kvt=.5*yf[i,...,1].square().mean(0).detach().cpu().numpy()
            rows.append({"sample":len(rows),"tke_ratio":np.linalg.norm(a)/max(np.linalg.norm(b),1e-12),"tke_sum_ratio":a.sum()/max(b.sum(),1e-12),"tke_rel":np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12),"scale_corrected_tke_rel":np.linalg.norm(alpha*a-b)/max(np.linalg.norm(b),1e-12),"scale_alpha":alpha,"correlation":corr,"u_ratio":np.linalg.norm(ku)/max(np.linalg.norm(kut),1e-12),"v_ratio":np.linalg.norm(kv)/max(np.linalg.norm(kvt),1e-12)})
        if bi<grad_batches:
            ls=parts(pred,y); vectors={}
            for name,value in ls.items():
                gs=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True); vectors[name]=[g.detach() for g in gs if g is not None]
                norm=float(torch.sqrt(sum((g.square().sum() for g in vectors[name]),torch.zeros((),device=device))))
                g_rows.append({"batch":bi,"loss":name,"raw_norm":norm,"weighted_norm":abs(weights.get(name,0))*norm,"value":float(value.detach())})
            for a,ga in vectors.items():
                for b,gb in vectors.items():
                    dot=sum((u*v).sum() for u,v in zip(ga,gb)); na=torch.sqrt(sum((u.square().sum() for u in ga))); nb=torch.sqrt(sum((v.square().sum() for v in gb))); cos.append({"batch":bi,"loss_a":a,"loss_b":b,"cosine":float(dot/(na*nb).clamp_min(1e-12))})
    for name,data in (("tke_anatomy.csv",rows),("gradient_norms.csv",g_rows),("gradient_cosines.csv",cos)):
        with (out/name).open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(data[0]) if data else ["empty"]);w.writeheader();w.writerows(data)
    P,Y=np.concatenate(kp),np.concatenate(ky); fig,ax=plt.subplots();ax.hist([r["tke_ratio"] for r in rows],bins=20);ax.set(xlabel="TKE norm ratio");fig.savefig(out/"tke_ratio_hist.png",dpi=140);plt.close(fig)
    fig,ax=plt.subplots();ax.scatter([r["tke_ratio"] for r in rows],[r["correlation"] for r in rows]);ax.set(xlabel="TKE ratio",ylabel="Pearson correlation");fig.savefig(out/"tke_ratio_correlation.png",dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,4,figsize=(14,3)); ims=[P.mean(0),Y.mean(0),abs(P.mean(0)-Y.mean(0)),abs(P.mean(0)-Y.mean(0))/np.maximum(abs(Y.mean(0)),1e-8)]
    for ax,im,title in zip(axs,ims,["pred TKE","GT TKE","absolute error","relative error"]):ax.imshow(im);ax.set_title(title);ax.axis("off")
    fig.savefig(out/"tke_maps.png",dpi=140);plt.close(fig)
    return {"samples":len(rows),"tke":{k:stats([r[k] for r in rows]) for k in rows[0] if k!="sample"}}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--mode",choices=("audit","train"),required=True);ap.add_argument("--real-root",type=Path,required=True);ap.add_argument("--checkpoint",type=Path,required=True);ap.add_argument("--realpdebench-root",type=Path,required=True);ap.add_argument("--out-dir",type=Path,required=True);ap.add_argument("--seed",type=int,default=1);ap.add_argument("--batch-size",type=int,default=12);ap.add_argument("--workers",type=int,default=2);ap.add_argument("--val-fraction",type=float,default=.2);ap.add_argument("--max-windows",type=int);ap.add_argument("--max-batches",type=int,default=8);ap.add_argument("--gradient-batches",type=int,default=3);ap.add_argument("--updates",type=int,default=300);ap.add_argument("--lr",type=float,default=1e-5);ap.add_argument("--weights",default=None);ap.add_argument("--weights-file",type=Path,default=None);args=ap.parse_args()
    if bool(args.weights) == bool(args.weights_file): raise ValueError("provide exactly one of --weights or --weights-file")
    args.out_dir.mkdir(parents=True,exist_ok=False); torch.manual_seed(args.seed);np.random.seed(args.seed);weights=json.loads(args.weights if args.weights else args.weights_file.read_text());device=torch.device("cuda" if torch.cuda.is_available() else "cpu");train,val,paths=build(args);model=load_model(args.realpdebench_root,args.checkpoint,device)
    if args.mode=="audit": result={"eval":evaluate(model,val,device),"anatomy":audit(model,val,device,args.out_dir,args.max_batches,args.gradient_batches,weights),"val_paths":[p.name for p in paths],"weights":weights}
    else:
        baseline=evaluate(model,val,device);opt=torch.optim.AdamW(model.parameters(),lr=args.lr);hist=[{"iteration":0}|baseline];it=iter(train)
        for step in range(1,args.updates+1):
            try:x,y,*_=next(it)
            except StopIteration:it=iter(train);x,y,*_=next(it)
            x,y=x.to(device),y.to(device); z=parts(forward(model,x),y);loss=sum(weights.get(k,0)*v for k,v in z.items());opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if step%max(1,args.updates//4)==0 or step==args.updates:hist.append({"iteration":step}|evaluate(model,val,device))
        torch.save({"model_state_dict":model.state_dict(),"history":hist,"weights":weights},args.out_dir/"best.pth");result={"baseline":baseline,"history":hist,"val_paths":[p.name for p in paths],"weights":weights}
    (args.out_dir/"summary.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=="__main__":main()
