# ENVIRONMENT

## 1. 运行环境

```text
OS            : Debian GNU/Linux 13 (trixie)
Kernel        : Linux 7.0.0-31-generic x86_64 (container)
Container     : realpde-residual-0831
Base image    : pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
Python        : 3.12.13
PyTorch       : 2.2.2+cu121
torchvision   : 0.17.2+cu121
CUDA runtime  : 12.1
cuDNN         : 8902 (8.9.2)
GPU           : NVIDIA GeForce RTX 3090
GPU memory    : 24576 MiB
NVIDIA driver : 595.84
```

## 2. 核心依赖版本

```text
torch        2.2.2+cu121
numpy        1.26.4
h5py         3.16.0
scipy        1.17.1
einops       0.8.2
einops-exts  0.0.4
matplotlib   3.11.1
pandas       3.0.3
scikit-learn 1.8.0
PyYAML       6.0.3
filelock     3.29.0
fsspec       2026.4.0
triton       3.8.0
```

说明：该容器还预装了大量与本任务无关的包（transformers / spacy / fastapi / pymilvus 等），
不要把它们当作方案依赖。方案真正需要的只有 `torch + numpy + h5py + scipy`（外加 submission 包内
自带的 `realpde_feature_engineering.py` 与 `rpde_baselines/cno.py`）。

## 3. 存储与挂载

```text
/data         : 真实 HDF5 数据 + 官方 baseline checkpoints（sim_real_cno.pth 等）
/runs         : 训练输出、缓存、checkpoint、submission zip（容器内）
/home/chyfuture/realpde_runs : /runs 的宿主机路径
/repo         : 只读代码挂载（/repo/tools 下是原始辅助脚本）
/third_party  : realpdebench 源码
```

## 4. 复现建议

- 使用同样的镜像 `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime` 可以最小化版本差异。
- 推理只需要 CPU 也能跑，但 Stage 2 / Stage 3 训练需要 GPU。
- `submission.py` 使用 `torch.half()` 加速 CNO backbone；如果没有 GPU，会走 CPU 路径但速度显著变慢。
- 完整 `pip freeze` 见同目录 `environment_freeze.txt`。

## 5. UNKNOWN

```text
- 官方 Codabench 评测机的 CPU/GPU 型号与 time_score 基准未公开；
  同一 submission 的 time_score 在不同提交间有小幅波动（例如 86.54 vs 87.04）。
- 容器内 Python 包环境包含无关组件，未做最小化冻结验证。
```
