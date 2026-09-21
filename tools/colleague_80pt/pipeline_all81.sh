#!/bin/bash
RUNS=/home/chyfuture/realpde_runs
LOG=$RUNS/final_all81.log
PL=$RUNS/pipeline.log
for i in $(seq 1 360); do
  if grep -q "^saved" "$LOG" 2>/dev/null; then echo "FINAL_ALL81_OK $(date)" >> $PL; break; fi
  if grep -q "DONE_ALL81" "$LOG" 2>/dev/null; then echo "FINAL_ALL81_NOSAVE $(date)" >> $PL; exit 1; fi
  if [ "$i" = "360" ]; then echo "FINAL_ALL81_TIMEOUT $(date)" >> $PL; exit 1; fi
  sleep 60
done
docker exec realpde-residual-0831 python -c "
import torch
ck = torch.load('/runs/cno_final_all81.pt', map_location='cpu')
print('CKPT', type(ck).__name__, sorted(ck.keys())[:4] if isinstance(ck, dict) else '-')
" > $RUNS/ckpt_check.log 2>&1
docker exec -d realpde-residual-0831 bash -lc "cd /runs && python -u -B /runs/strict/residual_multi.py --real-root /data/p0ab_real_h5_20260830 --checkpoint /runs/cno_final_all81.pt --realpdebench-root /third_party --base-model cno --out-dir /runs/residual_all81_20260915 --updates 3000 --hidden 64 --blocks 2 --batch-size 8 --stride 20 --train-on-all --train-alpha 1.0 --bound-abs 0.0075 --bound-rel 0.0075 > /runs/residual_all81.log 2>&1; echo RESIDUAL_DONE >> /runs/residual_all81.log"
echo "RESIDUAL_LAUNCHED $(date)" >> $PL
