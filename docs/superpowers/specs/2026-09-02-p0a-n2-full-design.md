# P0-A + N2 全量比赛训练设计

## 目标

构建一个可提交的 Track 1 P0-A + N2 CNO 单模型。它从官方
`sim_real_ft/sim_real_cno.pth` 重新初始化，用全部 82 条已发布 PIV
trajectory 拟合，并在不超过 6 小时墙钟、12 GB 峰值 GPU 显存的约束内
产出可验证的 checkpoint、提交包和运行记录。

## 冻结模型与输入

- CNO3d：`N_layers=3`，输出三通道；PIV pressure 输出始终设为零。
- 输入是 3 个原始通道加 17 个 P0-A 因果特征，共 20 通道。
- P0-A 只从 20 帧 `input_array` 计算；禁用 P0-B、CFD、Re、AoA、坐标
  metadata、target 和私有测试信息。
- 第一层 lift 卷积从 3 扩展至 20 通道：前三个权重严格复制，新增 17
  通道权重置零，因此训练开始时与 `sim_real_ft` CNO 等价。

## 冻结训练协议

- 训练集合：全部 82 条已发布 real PIV trajectory。全量阶段不保留 dev，
  不执行模型选择或基于标签的 early stopping。
- 损失：`MSE + 0.05*TKE + 0.027514*Rel-L2 + 0.009757*MVPE`，只在 u/v
  上计算，与历史 P0-A + N2 一致。
- 优化：AdamW，`lr=1e-5`，seed `20260901`；micro batch 4，梯度累积 2，
  有效 batch 8；固定 last checkpoint。
- 预算：6,800 optimizer updates；以历史 50-trajectory、4100 update 的
  约 16 epoch 预算，按全部 82 trajectory window 数等比例换算。停止阈值
  为 5 小时 40 分钟，以便验证和打包留出余量；绝不超过 6 小时。
- 显存：`torch.cuda.max_memory_allocated()` 硬 gate 为 12 GiB；超过即保存
  可恢复 checkpoint、停止并报告，绝不自动重启或放宽限制。

## 提交与验证

- 训练后用与训练相同的 P0-A torch 实现生成 submission API 的特征，避免
  numpy/torch 算子漂移；包中包含 checkpoint、submission.py 和官方 vendored
  CNO 依赖。
- 在官方 PyTorch Docker 镜像内进行 smoke test：导入、两次 predict 调用、
  输出形状 `(N,20,32,64,3)`、有限性、lower/upper 顺序和包大小 <256 MB。
- 区间 bounds 不在全量训练后选择。使用既有 P0-A 开发证据中预注册的保守
  常数，并在运行记录中注明它不是针对全量 checkpoint 调出的最优值。
- Codabench 上传不包含在本 Runner 内；训练完成后由用户明确触发结果回收和
  外部提交。

## 运行边界

- 在远程 `gpu` 的官方兼容容器中 detached 运行。
- 日志、checkpoint、submission zip 全部写入远程运行目录；不写入 Git。
- 启动后只核验一次 PID 和日志，不轮询、不 babysit。
