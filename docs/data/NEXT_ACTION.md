# Goal

建立一次与具体模型实验无关的 Track 1 Dataset Profile，作为后续所有实验的固定数据分布参考。

# Tasks

1. 读取 `docs/data/SKILL.md`，按 frozen manifest 对 50 Train / 16 Dev 做 dataset profiling。
2. 生成 per-trajectory input-side descriptors，并比较 Train / Dev 分布、coverage、nearest-neighbor distance / OOD-like 情况。
3. 可计算 Future20 target-side descriptors，但必须明确标记 analysis-only。
4. 将稳定结论写入 `docs/data/DATASET_PROFILE.md`，并保存必要的小型表格 / 图表路径。
5. 记录分析代码 commit、manifest SHA、窗口协议和 exact descriptor definitions。

# Constraints

- 不访问 16 locked-final。
- 不训练模型。
- 不做模型选择。
- 不新增 inference feature。
- 不根据 Dev error 定义 OOD 标签；distribution 标签必须由 Train-only 统计产生。

# Deliverables

- 完整 `docs/data/DATASET_PROFILE.md`
- 可复用分析脚本
- 必要的小型 summary artifacts
- commit SHA

# Stop

完成 profile 并 push `main` 后停止，`NEXT_ACTION = REVIEW_REQUIRED`。
