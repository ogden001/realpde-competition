#!/bin/bash
R=/home/chyfuture/realpde_runs
D="docker exec -d -e PYTHONPATH=/repo/tools realpde-residual-0831 bash -lc"

nvidia-smi --query-gpu=memory.used --format=csv,noheader

setsid nohup docker exec realpde-residual-0831 python -u -B /runs/cache_frozen.py --stride 5 --all-data --champion /runs/residual_h96x2_all81_20260919/model_best.pth --out /runs/frozen_h96x2 > $R/cache_h96x2.log 2>&1 < /dev/null &

$D "cd /runs && python -u -B /runs/strict/residual_multi.py --real-root /data/p0ab_real_h5_20260830 --checkpoint /runs/cno_final_all81.pt --realpdebench-root /third_party --base-model cno --out-dir /runs/residual_h96x4_all81_20260919 --updates 19200 --hidden 96 --blocks 2 --batch-size 8 --stride 20 --train-on-all --train-alpha 1.0 --bound-abs 0.0075 --bound-rel 0.0075 --max-delta 0.04 > /runs/residual_h96x4_all81.log 2>&1"

sleep 70
echo "=== PROCS ==="
docker exec realpde-residual-0831 pgrep -af cache_frozen | head -1
docker exec realpde-residual-0831 pgrep -af residual_h96x4 | head -1
echo "=== CACHE ==="
tail -1 $R/cache_h96x2.log 2>&1
echo "=== H96X4 ==="
head -2 $R/residual_h96x4_all81.log 2>&1
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
