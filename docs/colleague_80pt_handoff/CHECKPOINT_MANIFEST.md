# CHECKPOINT_MANIFEST

80.078849 分方案的 checkpoint 清单。所有 sha256 均为服务器上实测值。

**这些 checkpoint 文件本身没有提交到 Git**（`.gitignore` 排除 `*.pth` / `*.pt`，且体积过大）。
本文件记录精确身份，便于在其他存储位置校验。

## 1. 训练起点（官方权重）

```text
name   : sim_real_cno.pth
path   : /data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth
size   : 32154928 bytes
sha256 : 82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61
role   : Stage 1 初始化权重（官方 real-finetune CNO baseline）
in_git : NO (external official asset)
```

## 2. Stage 1 输出：all81 CNO

```text
name    : cno_final_all81.pt
path    : /runs/cno_final_all81.pt
size    : 31944962 bytes
sha256  : ff28aaf0114d57e320e5ea9cf9945d874a7482f22e3f0aeaed5ebbe286573f8a
role    : Stage 2 / Stage 3 的冻结 backbone
produced: final_all81.py, stride=1, updates=8723, batch=8, lr=1e-4
in_git  : NO (large binary)
```

## 3. Stage 2 输出：residual corrector

```text
name    : model_best.pth
path    : /runs/residual_h96x8_all81_20260920/model_best.pth
size    : 37102398 bytes
sha256  : 909fdc7f8a6a42335e4ea4ce7471fc9a507135e9a29a4927bc7b7318be9c85b2
role    : residual corrector checkpoint (best_iter=34000)
config  : hidden=96, blocks=2, max_delta=0.04, include_pressure=true
in_git  : NO (large binary)
note    : 这个文件同时包含 base_model.* 与 corrector.*，打包时直接作为 submission 的 model.pth
```

## 4. Stage 3 输出：uncertainty head

```text
name    : head_5000.pth
path    : /runs/head_h96x8_h64logmae/head_5000.pth
size    : 2024582 bytes
sha256  : 1dc6b56cf29b62a0e017cb0b434262215f81d94df22b964a04e1898720aa0522
role    : uncertainty head checkpoint (best dev SPS 51.5955)
config  : hidden=64, blocks=2, loss=logmae, updates=6000
in_git  : NO (large binary)
note    : head_6000.pth 的 dev SPS 为 51.5954，几乎相同；选择 5000 是严格 best
```

## 5. 提交包

```text
name    : submission_h_h96x8_20260920.zip
path    : /runs/submission_h_h96x8_20260920.zip
local   : submission_h_h96x8_20260920.zip (repo root, gitignored)
size    : 35670415 bytes
sha256  : 52347f088cb4f44f922e55fc915dcd9a2286764e6c9fd0f2686dc6775b49aca7
role    : 线上 80.078849 提交
in_git  : NO (*.zip ignored)
```

## 6. 校验命令

```bash
sha256sum /runs/cno_final_all81.pt
sha256sum /runs/residual_h96x8_all81_20260920/model_best.pth
sha256sum /runs/head_h96x8_h64logmae/head_5000.pth
sha256sum /runs/submission_h_h96x8_20260920.zip
```

## 7. 存储位置说明

服务器：`chyfuture@192.168.0.148`，容器 `realpde-residual-0831`，挂载：

```text
/runs         <- /home/chyfuture/realpde_runs
/data         <- real data + official baseline checkpoints
/repo         <- read-only repo mount
/third_party  <- realpdebench source
```

如果 `/runs` 中的数据被清理，Stage 1 可以用官方 `sim_real_cno.pth` 重训；Stage 2 / 3 也可以
按 `TRAINING.md` 重训，但需要约 7.5 小时（Stage 2）+ 约 0.3 小时（Stage 3）。
