# Status

`CLOSED`

当前 Data 方向无独立执行任务。

## Frozen Conclusions

- Full 82-trajectory input-side split audit：`SPLIT_OK`。
- 保持当前 `50 Train / 16 Dev / 16 locked-final` manifest，不重划分。
- Cross-split duplicate audit：`DUPLICATE_AUDIT_CLEAN`。
- 唯一 exact duplicate：Train `6300_0.h5` ↔ Final `7575_0.h5`，42 个 Past20 `u/v` windows 完全一致。
- 除上述 pair 外，无 `<0.1` near-duplicate；其余最近距离均 `>=0.573602`。

## Default Rules

1. 日常训练和模型选择继续只使用 Train / Dev。
2. Locked-final 已完成一次 input-side、target-blind audit，但仍不用于模型选择。
3. 若未来经明确授权进行 Final 模型评估，同时报告 `Final-all16` 与 `Final-unique15`（排除 `7575_0`）。
4. 后续 bad-case 分析默认关联 `docs/data/DATASET_PROFILE.md`。
5. 仅当 manifest、窗口协议、输入定义或 descriptor 定义发生实质变化时，重新打开 Dataset Profile。

## References

- `docs/data/DATASET_PROFILE.md`
- `docs/data/DUPLICATE_AUDIT.md`
- `docs/data/data概要.md`

不要自动启动新的 profiling、split redesign 或 duplicate audit。
