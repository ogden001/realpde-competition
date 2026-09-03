# NEXT_ACTION

## Goal
把新的 ChatGPT–Codex 协作规则写入 Codex 长期执行记忆。

## Tasks
1. 阅读 `docs/CHATGPT_CODEX_WORK_PROTOCOL.md`。
2. 更新根目录 `AGENTS.md`，只补充以下规则：
   - Codex 默认按 Luna-medium 的执行能力设计任务；
   - `NEXT_ACTION.md` 必须短、明确、原子化；
   - ChatGPT/Sol 负责实验设计和核心训练/评估/分析脚本；
   - Codex 负责环境适配、执行、smoke test、记录结果；
   - Codex 不做开放式研究，不自行扩大实验范围。
3. 不改动其他执行规则。
4. 提交并 push。

## Constraints
- 不重写 `AGENTS.md`。
- 不新增额外流程。
- 不修改实验代码。

## Deliverables
- `AGENTS.md` diff
- commit SHA

## Stop
`AGENTS.md` 更新、检查并提交后停止。
