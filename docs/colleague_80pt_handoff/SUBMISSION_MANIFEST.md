# SUBMISSION_MANIFEST

## 1. 基本信息

```text
submission artifact : submission_h_h96x8_20260920.zip
path                : /runs/submission_h_h96x8_20260920.zip
local path          : submission_h_h96x8_20260920.zip  (repo root, gitignored)
size                : 35670415 bytes
sha256              : 52347f088cb4f44f922e55fc915dcd9a2286764e6c9fd0f2686dc6775b49aca7
```

## 2. 包内文件（实测 unzip -l）

| file | size | sha256 |
|---|---:|---|
| `model.pth` | 37102398 | `909fdc7f8a6a42335e4ea4ce7471fc9a507135e9a29a4927bc7b7318be9c85b2` |
| `uncertainty_head.pt` | 2024582 | `1dc6b56cf29b62a0e017cb0b434262215f81d94df22b964a04e1898720aa0522` |
| `submission.py` | 11472 | `e1ced6f2a8a6de4ac96a08945f47b92a6ff15f8b1d4b215144187f08e09147a1` |
| `realpde_feature_engineering.py` | 12637 | `975348afb246a46a8fda3f31628c2488b30a839a3d2c866dc7a76c1387d25238` |
| `rpde_baselines/cno.py` | 25230 | `b37d04c3361e9c73cc332fd7852f721444454348f4eea97fa3ee1372996531bb` |
| `rpde_baselines/__init__.py` | 1 | — |
| `residual_submission_meta.json` | 611 | — |

## 3. Entrypoint

```text
entrypoint : submission.predict(input_array, metadata=None)
input      : np.ndarray (N, 20, 32, 64, 3), float32
output     : dict with keys {"prediction", "lower", "upper"}
             each (N, 20, 32, 64, 3), float32
```

## 4. Runtime dependency

```text
torch==2.2.2+cu121
numpy==1.26.4
（包内自带 realpde_feature_engineering.py / rpde_baselines/cno.py，不依赖仓库外源码）
```

## 5. 外部依赖检查

```text
hard-coded path  : NONE inside submission.py（使用 os.path.dirname(__file__)）
包外文件依赖     : NONE
网络依赖         : NONE
checkpoint 依赖  : 包内 model.pth + uncertainty_head.pt
config 依赖      : 常量硬编码在 submission.py 顶部（_HIDDEN/_HEAD_HIDDEN/_UNC_* 等）
```

## 6. 最终有效常量（包内 submission.py 实测）

```text
_HIDDEN              = 96
_BLOCKS              = 2
_MAX_DELTA           = 0.04
_CORRECTION_ALPHA    = 1.0
_HEAD_HIDDEN         = 64
_HEAD_BLOCKS         = 2
_HEAD_INCLUDE_PRESSURE = True
_HEAD_MIN_SIGMA      = 1e-4
_HEAD_MAX_SIGMA      = 1.0
_UNC_FLOOR_U         = 0.0025
_UNC_MULT_U          = 1.25
_UNC_FLOOR_V         = 0.0025
_UNC_MULT_V          = 1.5
_UNC_REL             = 0.005
_BOUND_ABS           = 0.0075
_BOUND_REL           = 0.0075
```

## 7. 官方 bench 结果

在本地同架构容器（`pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`，挂载解压目录到 `/app`）执行：

```text
RESULT per_window_s 0.02379 shape (2, 20, 32, 64, 3) finite True lo<=hi True not_fallback True
```

`not_fallback True` 表示模型正常加载并推理，没有回退 persistence。

## 8. 生成脚本

```text
/tools/colleague_80pt/make_submission_h96x8.py
```

该脚本：解压 `submission_f3_20260919.zip` 模板 -> 替换 `model.pth` / `uncertainty_head.pt`
-> 用正则精确改写常量（`_HIDDEN`、`_HEAD_HIDDEN`、`_UNC_*`、`_MAX_DELTA` 等）-> 重新 zip。

注意：模板 zip 和 checkpoint 都不在 Git 中。复现时需要先从 Stage 2 / 3 生成 checkpoint，
再用打包脚本或 `scripts/colleague_80pt_package.sh` 重新打包。

## 9. 提交历史备注

```text
online final 80.078849 是本方案的最终真实分数
同一模板的前一版 E（79.876809）使用 h64 residual + 早期 head
F3 是 h96 + h64logmae 的中间包，未提交或未记录线上分
```
