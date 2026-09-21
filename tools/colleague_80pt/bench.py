import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, submission
x = (np.random.RandomState(0).randn(2, 20, 32, 64, 3).astype(np.float32) * 0.1); x[...,2]=0
o = submission.predict(x)
ts=[]
for _ in range(5):
    t0=time.perf_counter(); o=submission.predict(x); ts.append((time.perf_counter()-t0)/x.shape[0])
p,lo,hi=o['prediction'],o['lower'],o['upper']; pers=np.repeat(x[:,-1:,:,:,:],20,axis=1)
print('RESULT per_window_s', round(float(np.mean(ts)),5), 'shape', p.shape, 'finite', bool(np.isfinite(p).all()), 'lo<=hi', bool((lo<=hi).all()), 'not_fallback', float(np.abs(p-pers).mean())>1e-4, flush=True)
