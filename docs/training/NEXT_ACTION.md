# NEXT_ACTION

## Goal
DW-01 逐帧复核已完成。当前不自动执行新的 Training 实验。

## Tasks
1. 保持 DW-01 状态为 `KEEP / STRONG_SIGNAL`。
2. 等待 ChatGPT/Sol 明确授权下一轮实验。
3. 下一候选高信息增益方向是：在 current competition-oriented / SOTA validation family 中，仅将 train window pool 替换为 Dense-All，做 matched A/B。

## Constraints
- 未经新授权，不启动训练、continuation、full-data、locked-final 或 Codabench。
- 不把历史 `sim_real_ft` canonical long run 当作 Dense-All 的严格因果对照。
- h19 late-tail anomaly 记录为机制问题，不自动扩展成新实验。

## Deliverables
- 无。等待下一份 Sol 授权任务。

## Stop
保持 STOP，直到 ChatGPT/Sol 给出新的 `IMPLEMENT_AND_EXECUTE_AUTHORIZED` 或 `READY_FOR_EXECUTION`。
