import sys, os, json, importlib.util
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

D = sys.argv[1]
sys.path.insert(0, '/repo/tools')
sys.path.insert(0, D)
import realpde_sps_scoring as S
from realpde_h5_feature_adapter_train import BAD_TRAIN_FILES, H5WindowDataset, list_h5, split_paths

spec = importlib.util.spec_from_file_location('sub_x', os.path.join(D, 'submission.py'))
m = importlib.util.module_from_spec(spec)
sys.modules['sub_x'] = m
spec.loader.exec_module(m)

paths = list_h5(Path('/data/p0ab_real_h5_20260830'), BAD_TRAIN_FILES)
tr, va = split_paths(paths, 0.2, 41)
mk = dict(in_steps=20, out_steps=20, stride=20, sub_sample=2,
          max_windows_per_trajectory=None, include_pressure=False)
ds = H5WindowDataset(va, **mk)
ld = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
print('EVAL_DIR', D, '| VAL windows', len(ds), '(stride=20, held-out 16 traj)', flush=True)

ps, ts = [], []
for x, y in ld:
    ps.append(m.predict(x.numpy().astype(np.float32))['prediction'])
    ts.append(y.numpy().astype(np.float32))
pred, tgt = np.concatenate(ps), np.concatenate(ts)
c = S.measured_channels(tgt)
dm = float(np.mean(S.rel_l2_per_sample(pred, tgt, c)))
tk = float(np.mean(S.tke_rel_l2_per_sample(pred, tgt, c)))
mv = float(S.mvpe_rel_l2(pred, tgt))
print('RESULT rel %.5f/%.3f  tke %.5f/%.3f  mvpe %.5f/%.3f'
      % (dm, S.score_error(dm), tk, S.score_error(tk), mv, S.score_error(mv)), flush=True)
