# TRAINING

三阶段串行训练。所有参数来自实际运行脚本与 `evidence/run_config.json`，非重新设计。

## Stage 0. 起点

官方 real-finetune CNO 权重：

```text
/data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth
size   32154928 bytes
sha256 82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61
```

## Stage 1 — CNO all81 fine-tune

脚本：`tools/colleague_80pt/final_all81.py`
输出：`/runs/cno_final_all81.pt`

```text
init checkpoint : sim_real_cno.pth
dataset         : all 81 usable trajectories (BAD_TRAIN_FILES excluded)
in/out steps    : 20 / 20
stride          : 1
sub_sample      : 2
windows         : 65926
optimizer       : AdamW(lr=1e-4), no weight decay specified
scheduler       : CosineAnnealingLR(T_max=updates)
batch size      : 8
updates         : 8723  (~1.06 epoch)
grad accum      : none (effective batch = 8)
AMP             : none (pure fp32)
grad clip       : none
seed            : NOT SET in this script  (UNKNOWN / non-deterministic data order)
checkpoint      : final step only, no best-selection
eval            : none inside Stage 1
```

### Stage 1 loss（decomp 模式，实际默认）

```python
w = ramp[None,:,None,None] * (1.0 + wake[None,None,:,:])
w = w / w.mean()
pm, tm = p.mean(1, keepdim=True), t.mean(1, keepdim=True)
pf, tf = p - pm, t - tm
main = mean(w * (pm - tm)^2) + mean(w * (pf - tf)^2)
kp = (0.5 * mean_t(pf^2)).sum(-1)
kt = (0.5 * mean_t(tf^2)).sum(-1)
tke = mean((1 + wake) * (kp - kt)^2)
loss = main + 0.05 * tke + 0.01 * mean(p_pred^2)
```

其中：

```text
p, t        = pred[..., :2], target[..., :2]
wake[h,w]   = 1.0 for h in [6,26), w in [10,45), else 0.0   # on the 32x64 grid
ramp[t]     = 1 + linspace(0, 1, 20)                         # per output frame
```

### Stage 1 实测日志（evidence/final_all81.log.txt）

```text
all81 windows 65926 stride 1 updates 8723
step 500  ... step 8500
saved /runs/cno_final_all81.pt
```

## Stage 2 — Residual corrector

脚本：`tools/colleague_80pt/residual_multi.py`
输出：`/runs/residual_h96x8_all81_20260920/model_best.pth`
启动脚本（原始）：`tools/colleague_80pt/launch_j.sh`

```text
base checkpoint : /runs/cno_final_all81.pt        (frozen, eval mode)
corrector       : ResidualCorrector3D(hidden=96, blocks=2, dropout=0, include_pressure=True,
                                      max_delta=0.04, history_context=False)
init            : last Conv3d zero-init -> iteration 0 == pure CNO
dataset         : all 81 trajectories (--train-on-all), stride=20, 3341 windows
dev set         : 16 trajectories, 640 windows (stride=20, same split seed=41)
optimizer       : AdamW(corrector.parameters(), lr=2e-4, weight_decay=1e-5)
scheduler       : CosineAnnealingLR(T_max=38400)
batch size      : 8 (test_batch_size=32)
updates         : 38400
grad accum      : none
AMP             : none (pure fp32)
grad clip       : clip_grad_norm_(corrector, 1.0)
seed            : 41
eval interval   : 100 updates
eval alphas     : [0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]
eval boundaries : abs=0.0075, rel=0.0075
checkpoint      : save model_best.pth when best_final_est improves
best iteration  : 34000
```

### Stage 2 loss（公式级）

基础物理项（`physics_loss`，`realpde_h5_feature_adapter_train.py`）：

```text
L_point    = mean_b( ||pred_uv - target_uv||_2 / ||target_uv||_2 )
L_mse      = mean( (pred_uv - target_uv)^2 )
L_tke      = mean_b( ||KE(pred) - KE(target)|| / ||KE(target)|| )
L_temporal = mean_b( ||Δ_t pred - Δ_t target|| / ||Δ_t target|| )
L_grad     = 0.5*RelL2(Δ_x pred, Δ_x target) + 0.5*RelL2(Δ_y pred, Δ_y target)
L_pzero    = mean( pred_p^2 )
```

其中 `KE(x) = 0.5 * (var_t(u) + var_t(v))`，先在时间维求方差得到 `(B,H,W)` 场再取相对 L2。

总损失：

```text
L = 1.0   * L_point
  + 0.05  * L_mse
  + 0.06  * L_tke
  + 0.03  * L_temporal
  + 0.015 * L_grad
  + 0.01  * L_pzero
  + 0.25  * L_residual_mse
  + 0.02  * L_delta_penalty
```

额外两项：

```text
L_residual_mse  = mean( (delta_uv - (target_uv - base_uv))^2 )
L_delta_penalty = mean( delta_uv^2 )
```

它把 residual 训练同时约束成“拟合真实残差”和“残差尽量小”，是残差头稳定收敛的关键。

## Stage 3 — Uncertainty head

脚本：`tools/colleague_80pt/train_head_fast.py`
特征缓存：`tools/colleague_80pt/cache_frozen.py`
输出：`/runs/head_h96x8_h64logmae/head_5000.pth`

### Stage 3.1 冻结缓存

```text
champion_used : /runs/residual_h96x8_all81_20260920/model_best.pth
all_data      : true (81 trajectories)
stride_train  : 5   -> 13202 train windows
stride_val    : 20  -> 640 val windows (16 trajectories, seed=41)
dtype         : float16
channels      : 2
saved fields  : frozen_{tr,va}_{x,b,p,y}.npy
```

`b` = base CNO prediction，`p` = base + delta（即最终 prediction 的 u/v），`y` = target。

### Stage 3.2 head 训练

```text
head            : UncertaintyHead3D(hidden=64, blocks=2, dropout=0, include_pressure=True)
init            : last Conv3d zero-init
dataset         : cached frozen_{tr,va} memmaps
optimizer       : AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler       : warmup 200 steps + CosineAnnealingLR
batch size      : 16
updates         : 6000
eval every      : 500
seed            : 41
AMP             : none
checkpoint      : every 500 steps (head_{step}.pth) + eval_log.json
selected        : head_5000.pth  (best dev SPS 51.5955)
```

### Stage 3 loss（logmae）

```python
err     = torch.abs(y[..., :2] - pred[..., :2])       # 每像素绝对误差
log_std = head(x, base)                               # clamp 后的 log sigma
per     = torch.abs(log_std - torch.log(err + 1e-6))  # logmae loss
loss    = per.mean()
```

即让 `sigma` 在乘法尺度上逼近逐像素 `|error|`。logmae 优于 nll / pinball / mae / sps surrogate
（见 `EXPERIMENT_HISTORY.md` 的 TESTED BUT NOT USED 段）。

### Stage 3 实测曲线（evidence/eval_log.json）

```text
step 500  sps 50.9846
step 1000 sps 51.1910
step 2000 sps 51.4202
step 3000 sps 51.4489
step 4000 sps 51.5687
step 5000 sps 51.5955   <- selected
step 6000 sps 51.5954
```

## 阶段汇总

| Stage | Module | Frozen | Trainable | windows | updates | lr | batch | best iter |
|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | CNO all81 | — | all CNO | 65926 (stride 1) | 8723 | 1e-4 | 8 | final |
| 2 | residual corrector h96/b2 | CNO | corrector | 3341 (stride 20) | 38400 | 2e-4 | 8 | 34000 |
| 3 | uncertainty head h64/b2 | CNO + corrector | head | 13202 (stride 5) | 6000 | 1e-3 | 16 | 5000 |

## 复现命令

```bash
# Stage 1
python -u -B tools/colleague_80pt/final_all81.py \
  --stride 1 --updates 8723 --batch 8 --lr 1e-4 \
  --out /runs/cno_final_all81.pt

# Stage 2
python -u -B tools/colleague_80pt/residual_multi.py \
  --real-root /data/p0ab_real_h5_20260830 \
  --checkpoint /runs/cno_final_all81.pt \
  --realpdebench-root /third_party --base-model cno \
  --out-dir /runs/residual_h96x8_all81_20260920 \
  --updates 38400 --hidden 96 --blocks 2 --batch-size 8 --stride 20 \
  --train-on-all --train-alpha 1.0 \
  --bound-abs 0.0075 --bound-rel 0.0075 --max-delta 0.04

# Stage 3 cache
python -u -B tools/colleague_80pt/cache_frozen.py \
  --stride 5 --all-data \
  --champion /runs/residual_h96x8_all81_20260920/model_best.pth \
  --out /runs/frozen_h96x8

# Stage 3 head
python -u -B tools/colleague_80pt/train_head_fast.py \
  --cache /runs/frozen_h96x8 \
  --hidden 64 --blocks 2 --loss logmae \
  --updates 6000 --eval-every 500 \
  --out /runs/head_h96x8_h64logmae
```
