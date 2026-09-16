# NEXT_ACTION

## Goal

在已完成 `ADAPTIVE_GO` 的 SOTA-V2 adaptive 结果上，使用修正后的 full-checkpoint SHA guard **干净重建最终 submission ZIP 并重新跑 A/B clean-room smoke**。不重训 backbone，不重训 uncertainty head，不重跑 SPS calibration。

状态：`READY_FOR_EXECUTION / REVIEW_REQUIRED`

## Context

- Adaptive execution/result commit：`ec81c5dd0c6c321950a63248375e027ad6699975`。
- 当前修复 commit：`caef5f78eeaf3b880ab6889df642cb0426c65d6d`。
- 根因：旧 `build_sota_v2_adaptive_package.py` 的 `EXPECTED_FULL_CHECKPOINT_SHA` 多了一个末尾字符；真实 full checkpoint SHA256 为 `f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce`。
- 已新增回归测试，要求常量与记录 digest 完全一致且长度为 64。
- 已完成的科学结果保持冻结：Adaptive best SPS `45.07008160038756`，static SPS `42.12489194711354`，bounds `floor=0.0025, mult=1.0`。

## Tasks

1. 同步 `main`，确认包含 `caef5f78eeaf3b880ab6889df642cb0426c65d6d`，工作区干净。
2. 运行 `tests/test_sota_v2_adaptive_package.py`、`tests/test_sota_v2_adaptive.py` 与相关 `py_compile`。
3. 复用既有 `adaptive_head_1400.pth` 和 `calibration_summary.json`，不要重新训练或重新 calibration。
4. 用修复后的 builder 从零创建新的 package 目录，禁止任何 in-process SHA override/monkeypatch。
5. 对新的 ZIP 独立运行真实 fixture A/B clean-room verifier；两条都必须 PASS，prediction parity `<=1e-6`，deterministic diff `0`。
6. 记录新的 ZIP SHA/bytes/runtime/peak CUDA；更新 adaptive review/handoff/provenance，明确 clean rebuild 已不再使用 override。
7. push `main` 后停止，不提交 Codabench。

## Constraints

- 不修改 backbone、head、bounds、SPS grid 或 runtime 算法。
- 不重训任何模型。
- 不访问 locked-final/private test/Codabench。
- 不提交 checkpoint、ZIP、dataset、大日志到 Git。
- 若新 builder 或 A/B smoke 失败，停止并回报；不要临场修改核心代码。

## Deliverables

- 新 `package_build.json`。
- 新 `package_smoke_a.json` / `package_smoke_b.json`。
- 更新 `SHA256_PROVENANCE.md`、adaptive `README.md`、handoff。
- 最终状态：`READY_TO_UPLOAD / REVIEW_REQUIRED`。

## Stop

clean rebuild + A/B smoke + evidence push 完成后立即停止，等待 ChatGPT/Sol 复核；不要提交 Codabench。
