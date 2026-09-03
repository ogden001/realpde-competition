# Track 1 执行规则

## 必读上下文

在规划、修改代码或解读实验之前，必须阅读：

1. 工作区根目录的 `MEMORY.md`；
2. 本仓库的 `README.md` 与本任务相关的 `docs/` 文档；
3. 当前交接状态优先阅读 `docs/coordination/STATUS.md`；以及
4. 按任务相关范围阅读 `docs/track1_experiment_registry.md`。

`STATUS.md` 是简短的交接索引，不能替代实验注册表或 `MEMORY.md`。`submission_log.md` 只记录真实的 Codabench 提交。

## ChatGPT–Codex 协作

将用户提供的 `NEXT_ACTION` 视为当前的有界任务。它必须明确，或 Codex 必须依据现有协议确定：目标、允许的数据/资源预算、禁止动作和验收条件。不得扩张任务范围。

每个实质性实验或候选，都要向 `docs/track1_experiment_registry.md` 追加可核验事实：实验 ID、状态（`DONE`、`REVIEW_REQUIRED`、`BLOCKED` 或 `RUNNING`）、commit SHA 或明确的脏工作树说明、split/manifest SHA、精确命令和关键配置、artifact ID 或仓库相对路径、核心指标/观察，以及 `GO` / `STOP` / `REVIEW_REQUIRED` 结论。不得覆盖既有结论。

任务结束时，更新 `docs/coordination/STATUS.md` 中的最近完成项、当前状态、允许范围和 review handoff。只有证据充分时才更新注册表中的稳定结论。不得将数据集、checkpoint、凭证、绝对私有路径或生成的 submission archive 写入 Git。

## 长时任务

启动长时间 CPU/GPU 任务前，先完成实现与 smoke test；说明主机、命令、日志和 artifact 路径，以及监控命令。以 detached 方式启动 Runner，一次性确认其 PID/日志，记录 `RUNNING` 后停止轮询。仅在后续得到明确请求时回收结果。

不要假设 GitHub push/pull 可用，也不要假设外部 ChatGPT 会话能读取这个私有仓库。应在此记录本地事实，并向用户报告尚未推送的协调改动。

## Git Docs 作为长期技术记忆

Git `docs/` 是 ChatGPT/Sol、Codex 与研发人员共享的长期技术记忆。聊天上下文是短期的，不能替代 Git 中沉淀的阶段性结论。

信息流为：实验/分析事实 → 具体实验记录 → 稳定的阶段性结论 → 对应方向概要 → `docs/realpde整体优化概要.md`。具体记录保留完整配置、命令、溯源和指标；概要保持简短，并链接回这些证据。

ChatGPT/Sol 负责技术方向、实验设计和结果复核，包括 `KEEP` / `REVIEW` / `NO-GO` / `STOP` 判断与优先级。形成重要稳定结论后，应指出需要沉淀的结论和需要更新的概要文档。Codex 负责执行事实和 artifact、详细记录，以及在结论确认后增量更新概要。只有一级方向发生实质变化时，才更新整体概要。

当 ChatGPT/Sol 的实验复核明确给出 `Docs impact` 时，Codex 应在当前任务收尾时按其中指定的方向和文档完成增量更新；如果 `Docs impact = NO`，则不要为了形式更新概要。

不要因为进行中的运行、中间 checkpoint 或不稳定观察更新概要。以下情况应更新：证据确认某方法有效、出现值得避免重复的失败/停止路线、形成关键技术认识、优先级变化、一个探索阶段结束，或结果改变了下一步路线。不要把概要写成实验流水账，也不要复制长报告。被新证据推翻的结论应直接更新；重要的 `NO-GO` / `STOP` 结论应保留简短记录，避免重复实验。

开始某个已有方向的新任务前，先阅读其概要，再读取本有界任务需要的详细证据。若拟议实验重复已记录的 `NO-GO` / `STOP`，或与既有结论冲突，应主动指出，而不是静默重跑。
