# NEXT_ACTION

## Goal
把最新的 ChatGPT–Codex 协作规则增量写入 `AGENTS.md`。

## Tasks
1. 阅读 `docs/CHATGPT_CODEX_WORK_PROTOCOL.md`。
2. 更新根目录 `AGENTS.md`，补充：
   - 默认直接使用 `main`，除非用户明确要求，否则不创建 branch/worktree；
   - 任务开始前检查工作树并同步 `origin/main`；
   - 遇到未知未提交改动先报告，不自行 stash/reset/restore/clean；
   - 任务结束后 commit + push，并保持工作树干净；
   - `NEXT_ACTION.md` 必须短、明确、原子化；
   - ChatGPT/Sol 负责实验设计和核心训练/评估/分析脚本；Codex 负责环境适配与执行。
3. 不改动其他规则。
4. 提交并 push 到 `main`。

## Constraints
- 不重写 `AGENTS.md`。
- 不创建 branch/worktree。
- 不修改实验代码。

## Deliverables
- `AGENTS.md` diff
- commit SHA

## Stop
提交并确认 main 工作树干净后停止。
