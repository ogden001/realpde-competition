# NEXT_ACTION

## Goal

为冻结的 SOTA-V2 backbone 完成 **fresh Adaptive Uncertainty → Dev SPS calibration → full@53582 package smoke**。核心代码/验证脚本已由 ChatGPT/Sol 提供；Codex 只执行与记录。

状态：`READY_FOR_EXECUTION / REVIEW_REQUIRED`

## Tasks

1. 同步 `main`、验证写权限，运行 adaptive/package focused tests 与 `py_compile`。
2. 使用 SOTA-V2 50/16 `@32500`：先复现 Dev raw errors `0.0999346 / 0.4692927 / 0.0757780`；不匹配即停止。
3. 冻结 backbone，用 50 Train canonical `2052` windows 训练 fresh v5 Adaptive Uncertainty Head：`1400` updates；随后在 16 Dev `659` windows 跑固定 `4×7=28` SPS grid，并与同 backbone static bounds 比较。
4. 仅 `ADAPTIVE_GO` 时，将同一 head + 冻结 bounds 挂到 full `@53582`（SHA256 `f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce8`），构建 ZIP。
5. 使用真实 fixture 跑 clean-room smoke；package prediction 与裸 full backbone `max_abs_diff <= 1e-6`，pressure prediction/bounds=0，finite/deterministic/shape/dtype 全部通过。
6. 提交轻量 evidence/handoff 并 push `main`。

## Constraints

- 不训练/修改 backbone；不使用 Residual Corrector。
- Head 训练只用 50 Train canonical windows；16 Dev 只做 replay/SPS calibration。
- 固定 head：15→32、2 residual blocks；AdamW `1e-3`, wd `1e-5`, seed `20260905`, 1400 updates。
- 固定 grid：floor `{0,0.0025,0.005,0.0075}` × mult `{0.5,1,1.5,2,2.5,3,4}`；不扩 grid、不 sweep。
- 不访问 locked-final/private test，不提交 Codabench。
- 不提交 checkpoint、ZIP、dataset、大日志到 Git。

## Deliverables

- validation replay / head training / calibration grid+summary。
- 若 GO：package build + clean-room smoke + ZIP path/SHA/size/runtime。
- `docs/coordination/CHATGPT_HANDOFF_SOTA_V2_ADAPTIVE_20260916.md`。
- 状态：`REVIEW_REQUIRED`。

## Stop

完成 evidence + push 后立即停止；不要提交 Codabench。
