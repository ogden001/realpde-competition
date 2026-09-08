# NEXT_ACTION

## Goal
补齐 DW-01 Dense-All 的 Future20 逐帧 / By-Horizon 证据；只做 prediction-only replay，不重新训练。

## Tasks
1. 读取 `docs/EXPERIMENT_BY_HORIZON_PROTOCOL.md`，在现有评估管线最小实现通用 by-horizon analyzer，并用 TDD 验证 20-horizon、perfect-prediction、trajectory aggregation 和 probe geometry。
2. 固定原 50 Train / 16 Dev manifest、P0-A、N2、official v9 scorer，重放 DW-01 `7.5k / 20k / 30k` checkpoints；aggregate replay 必须与原 summary 做 parity check。
3. 若能从 artifact manifest 唯一解析同 `sim_pretrain` RW-00 fixed@7.5k checkpoint，则加入 matched 7.5k by-horizon A/B；若资产不存在，只记录缺失，不替代 checkpoint。
4. 输出并解释 early / middle / late horizon：Frame Rel-L2、Frame RMSE、TKE contribution relative error / ratio、MVPE-probe diagnostic，以及 trajectory × horizon 数据。
5. 更新 DW-01 README，commit + push evidence 到 `main`，状态 `REVIEW_REQUIRED`。

## Constraints
- `REQUIRED_BASE_COMMIT = f5043847bdc2b8617c3caf33c8040e18e97f501b`。
- 不训练、不 continuation、不改模型 / Loss / Feature / split / scorer / checkpoint。
- Dev 始终 `start=0,stride=20`；不访问 locked-final / Codabench / full-data。
- TKE / MVPE 的逐帧量只标记为 diagnostic decomposition，不称为 official per-frame score。

## Deliverables
- `by_horizon.csv`、`by_trajectory_horizon.csv`、`summary.json`，可选 `by_horizon.png`。
- replay parity、checkpoint / manifest / scorer SHA、tests / compile 结果。
- DW-01 README 的逐帧结论与 `REVIEW_REQUIRED` handoff。

## Stop
完成 DW-01 逐帧 replay、分析、commit + push 后停止；不要设计或启动下一实验。
