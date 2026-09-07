#!/usr/bin/env python3
"""Read-only Future20 statistics collector for fixed Plain and P0-A CNOs."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from horizon_cliff_audit import (EXPECTED_SHAPE, load_cno, p0a_config_from_checkpoint, forward_p0a,
    paths_from_manifest, verify_all_target_alignment, sha256)
from realpde_p0_data import H5WindowDataset
from realpde_p0_features import P0FeatureBuilder

def tail_squared_error_contribution(se: np.ndarray) -> dict[str, float]:
    total = float(se.sum())
    return {"t19_fraction": float(se[18]/total), "t20_fraction": float(se[19]/total),
            "t19_t20_fraction": float((se[18]+se[19])/total)}

def scorer(kit: Path):
    sys.path.insert(0, str(kit)); import scoring
    return scoring

def model_metrics(sc, pred, target):
    c=2
    return {"rel_l2": float(sc.rel_l2_per_sample(pred,target,c).mean()),
            "tke": float(sc.tke_rel_l2_per_sample(pred,target,c).mean()),
            "mvpe": float(sc.mvpe_rel_l2_per_sample(pred,target).mean())}

def collect(name, pred, target, refs, sc, rows, spatial, contribution):
    se_h=[]
    for h in range(20):
        p,t=pred[:,h:h+1],target[:,h:h+1]
        err=p[...,:2]-t[...,:2]; sq=err**2
        speed_p=np.linalg.norm(p[...,:2],axis=-1); speed_t=np.linalg.norm(t[...,:2],axis=-1)
        se=float(sq.sum()); se_h.append(se)
        spatial_row={"model":name,"horizon":h+1,"rel_l2":float(sc.rel_l2_per_sample(p,t,2).mean()),
          "mvpe":float(sc.mvpe_rel_l2_per_sample(p,t).mean()),
          "tke_error":float(sc.tke_rel_l2_per_sample(p,t,2).mean()),
          "velocity_squared_error":float(sq.mean()),"pred_speed_mean":float(speed_p.mean()),"pred_speed_std":float(speed_p.std()),"target_speed_mean":float(speed_t.mean()),"target_speed_std":float(speed_t.std())}
        spatial["horizon_rows"].append(spatial_row)
        for i,ref in enumerate(refs): rows.append({"model":name,"trajectory":ref.path.stem,"window_start":ref.start,"horizon":h+1,"rel_l2":float(sc.rel_l2_per_sample(p[i:i+1],t[i:i+1],2)[0]),"mvpe":float(sc.mvpe_rel_l2_per_sample(p[i:i+1],t[i:i+1])[0]),"velocity_squared_error":float(sq[i].mean())})
        if h in (17,18,19):
            spatial[f"{name}_t{h+1}_mean_abs_velocity_error"]=np.linalg.norm(err,axis=-1).mean(axis=0)
            spatial[f"{name}_t{h+1}_mean_squared_velocity_error"]=sq.mean(axis=(0,4))
    se=np.array(se_h); tail=tail_squared_error_contribution(se)
    for h,value in enumerate(se,1): contribution.append({"model":name,"horizon":h,"squared_error_sum":float(value),"squared_error_fraction":float(value/se.sum()),"se_total":float(se.sum()),**tail})
    return {"full20":model_metrics(sc,pred,target),"first18":model_metrics(sc,pred[:,:18],target[:,:18]),"tail":tail}

@torch.no_grad()
def run(a):
    sc=scorer(a.kit_root); dev=paths_from_manifest(a.manifest,a.data_root,"dev")
    ds=H5WindowDataset(dev,in_steps=20,out_steps=20,stride=20,sub_sample=2,include_pressure=False)
    assert len(ds)==659; alignment=verify_all_target_alignment(ds); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plain=load_cno(a.kit_root,in_dim=3,checkpoint=a.plain_checkpoint,device=device)
    payload=torch.load(a.p0a_checkpoint,map_location="cpu"); builder=P0FeatureBuilder(p0a_config_from_checkpoint(payload)).to(device)
    p0a=load_cno(a.kit_root,in_dim=len(builder.feature_names),checkpoint=a.p0a_checkpoint,device=device)
    pp=[]; pa=[]; yy=[]
    for x,y,_,_ in DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=0):
        x=x.to(device); pp.append(plain(x.permute(0,4,1,2,3)).permute(0,2,3,4,1).cpu().numpy()); pa.append(forward_p0a(p0a,builder,x).cpu().numpy()); yy.append(y.numpy())
    pp,pa,yy=map(np.concatenate,(pp,pa,yy)); assert pp.shape==pa.shape==yy.shape==EXPECTED_SHAPE
    rows=[]; spatial={"horizon_rows":[]}; contribution=[]
    summary={"plain":collect("plain_piv_cno",pp,yy,ds.refs,sc,rows,spatial,contribution),"p0a":collect("p0a_cno",pa,yy,ds.refs,sc,rows,spatial,contribution)}
    a.out_dir.mkdir(parents=True,exist_ok=False)
    with (a.out_dir/"horizon_metrics_closeout.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(spatial["horizon_rows"][0]));w.writeheader();w.writerows(spatial.pop("horizon_rows"))
    with (a.out_dir/"per_window_horizon_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (a.out_dir/"tail_contribution.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(contribution[0]));w.writeheader();w.writerows(contribution)
    np.savez_compressed(a.out_dir/"spatial_residuals.npz",**spatial)
    evidence={"prediction_shape":list(pp.shape),"target_shape":list(yy.shape),"dev_windows":len(ds),"trajectories":len(dev),"horizons":20,"alignment":alignment,"plain_sha256":sha256(a.plain_checkpoint),"p0a_sha256":sha256(a.p0a_checkpoint),"manifest_sha256":sha256(a.manifest),"summary":summary}
    (a.out_dir/"compute_evidence.json").write_text(json.dumps(evidence,indent=2)+"\n")
def main():
 p=argparse.ArgumentParser();
 for n in ("data_root","manifest","kit_root","plain_checkpoint","p0a_checkpoint","out_dir"):p.add_argument("--"+n.replace("_","-"),type=Path,required=True)
 p.add_argument("--batch-size",type=int,default=8);run(p.parse_args())
if __name__=="__main__":main()
