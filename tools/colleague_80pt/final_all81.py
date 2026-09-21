import sys, json, argparse
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, '/repo/tools')
import realpde_base_quickcompare as QC
from realpde_h5_feature_adapter_train import BAD_TRAIN_FILES, H5WindowDataset, list_h5

ap = argparse.ArgumentParser()
ap.add_argument('--stride', type=int, default=1)
ap.add_argument('--updates', type=int, default=8723)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--loss', choices=['decomp', 'weighted'], default='decomp')
ap.add_argument('--out', type=str, default='/runs/cno_final_all81.pt')
a = ap.parse_args()

dev = torch.device('cuda')
paths = list_h5(Path('/data/p0ab_real_h5_20260830'), BAD_TRAIN_FILES)   # all 81 usable trajectories
mk = dict(in_steps=20, out_steps=20, stride=a.stride, sub_sample=2, max_windows_per_trajectory=None, include_pressure=False)
H, W, T = 32, 64, 20
wake = np.zeros((H, W), np.float32); wake[6:26,10:45]=1.0
ramp = 1.0 + np.linspace(0,1.0,T,dtype=np.float32)
w_np = ramp[None,:,None,None]*(1.0+wake[None,None,:,:]); w_np/=w_np.mean()

def loss_fn(pred, target):
    w = torch.from_numpy(w_np).to(pred.device).unsqueeze(-1)
    p = pred[..., :2]; t = target[..., :2]
    pm = p.mean(1, keepdim=True); tm = t.mean(1, keepdim=True)
    pf = p - pm; tf = t - tm
    main = (w*(pm-tm).square()).mean() + (w*(pf-tf).square()).mean()
    if a.loss == 'weighted':
        main = (w*(p-t).square()).mean()
    kp = (0.5*(pf**2).mean(1)).sum(-1); kt = (0.5*(tf**2).mean(1)).sum(-1)
    wk = torch.from_numpy((1.0+wake)).to(pred.device)
    tke = (wk*(kp-kt).square()).mean()
    return main + 0.05*tke + 0.01*(pred[...,2]**2).mean()

ds = H5WindowDataset(paths, **mk)
ld = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=0, drop_last=True)
print('all81 windows', len(ds), 'stride', a.stride, 'updates', a.updates, flush=True)
m = QC.make_model('cno', Path('/data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth'), Path('/third_party'), dev)
opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.updates)
it = iter(ld); step = 0
while step < a.updates:
    try: x,y = next(it)
    except StopIteration: it = iter(ld); x,y = next(it)
    x=x.to(dev); y=y.to(dev)
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(QC.forward(m,'cno',x), y); loss.backward(); opt.step(); sched.step(); step+=1
    if step % 500 == 0: print('step', step, float(loss.detach().cpu()), flush=True)
torch.save({'state_dict': m.state_dict()}, a.out)
print('saved', a.out, flush=True)
