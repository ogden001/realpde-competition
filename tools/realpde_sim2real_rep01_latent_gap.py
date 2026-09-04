#!/usr/bin/env python3
"""Small frozen-CNO CFD/PIV representation gap diagnostic (train/dev only)."""
import argparse, json, sys
from pathlib import Path
import h5py, numpy as np, torch

def main():
    p=argparse.ArgumentParser(); p.add_argument('--piv-root',type=Path,required=True); p.add_argument('--sim-root',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--kit-root',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); a=p.parse_args()
    sys.path.insert(0,str(a.kit_root)); from rpde_baselines.model.cno import CNO3d
    m=json.loads(a.manifest.read_text()); keys=[r['file'] for s in ('train','dev') for r in m[s]]; dev='cuda' if torch.cuda.is_available() else 'cpu'
    model=CNO3d(in_dim=3,out_dim=3,out_dim_mult=1,in_size=64,N_layers=3).to(dev); q=torch.load(a.checkpoint,map_location='cpu',weights_only=False); model.load_state_dict(q.get('model_state_dict',q)); model.eval(); vals=[]
    for fn in keys:
        arr=[]
        for root in (a.piv_root,a.sim_root):
            with h5py.File(root/fn,'r') as f: arr.append(np.stack([f['u'][:20,::2,::2],f['v'][:20,::2,::2],np.zeros((20,32,64),np.float32)],-1))
        x=torch.tensor(np.stack(arr),dtype=torch.float32,device=dev)
        with torch.no_grad(): z=model(x.permute(0,4,1,2,3)).permute(0,2,3,4,1).cpu().numpy()
        mu=z.mean((1,2,3)); va=z.var((1,2,3)); l2=np.linalg.norm(mu[0]-mu[1])/(np.linalg.norm(mu[1])+1e-8); cos=1-np.dot(mu[0],mu[1])/(np.linalg.norm(mu[0])*np.linalg.norm(mu[1])+1e-8); vl2=np.linalg.norm(va[0]-va[1])/(np.linalg.norm(va[1])+1e-8); vals.append((l2,cos,vl2))
    print(json.dumps({'n':len(vals),'median_l2':float(np.median([v[0] for v in vals])),'median_cosine_distance':float(np.median([v[1] for v in vals])),'median_variance_l2':float(np.median([v[2] for v in vals]))}))
if __name__=='__main__': main()
