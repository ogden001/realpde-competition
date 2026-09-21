import sys, time
from pathlib import Path
sys.path.insert(0,'/repo/tools')
import numpy as np, torch
from realpde_h5_feature_adapter_train import BAD_TRAIN_FILES, H5WindowDataset, list_h5, split_paths
from realpde_h5_residual_corrector_eval import load_model
dev = torch.device('cuda')
p = list_h5(Path('/data/p0ab_real_h5_20260830'), BAD_TRAIN_FILES)
tr, va = split_paths(p, 0.2, 41)
ds = H5WindowDataset(tr, in_steps=20, out_steps=20, stride=20, sub_sample=2, max_windows_per_trajectory=None, include_pressure=False)
t0=time.time(); xs=[ds[i][0] for i in range(8)]; t1=time.time()
x = torch.stack(xs).to(dev)
print('load8_s', round(t1-t0,2), flush=True)
m = load_model(Path('/runs/residual_split_20260916/model_best.pth'), Path('/third_party'), dev)
print('loaded', flush=True)
with torch.no_grad():
    for i in range(3):
        torch.cuda.synchronize(); t=time.time()
        b = m.base_predict(x); d = m.predict_delta(x, b); pr = m.combine(b, d, 1.0)
        torch.cuda.synchronize()
        print('iter', i, 'sec', round(time.time()-t,4), flush=True)
print('ok', flush=True)
