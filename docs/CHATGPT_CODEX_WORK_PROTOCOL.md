# ChatGPT–Codex 工作协议

## 1. 角色分工

### ChatGPT / Sol
负责需要较强研究判断的工作：
- 技术方向与优先级；
- 实验设计与变量控制；
- 训练、评估、分析脚本的核心逻辑；
- 结果复核与下一步决策。

### Codex / Luna-medium
负责仓库和环境内的工程执行：
- 阅读任务文件和相关代码；
- 做必要的环境适配；
- 实现或落地已明确的改动；
- smoke test；
- 启动训练/评估/分析；
- 记录事实、产物和 commit。

Codex 不承担开放式研究规划，不自行扩大实验范围，不自行设计下一轮实验。

## 2. Git 协作规则

本仓库默认直接使用 `main` 作为 ChatGPT、Codex 和用户之间的共享工作分支。

除非用户明确要求，或任务存在明显隔离需求：
- ChatGPT 不创建新 branch；
- Codex 不创建新 branch 或 worktree；
- 不为普通文档、小型代码修改或单次实验额外建立分支。

开始任务前，Codex 应先确认当前位于 `main`，检查工作树状态并同步 `origin/main`。若存在未知未提交改动，不得擅自 stash、reset、restore、clean 或覆盖，应先报告。

任务完成后，原则上应 commit 并 push 到 `main`，并确认工作树干净；若确有文件不能提交，必须明确列出文件和原因，不得默默留下脏工作区。

ChatGPT 直接写 GitHub 时也默认写入 `main`，从而保持：

**GitHub main = Codex 本地 main = ChatGPT 下一轮读取的项目状态**。

## 3. NEXT_ACTION 规则

`docs/coordination/NEXT_ACTION.md` 是 Codex 当前施工单，必须简单、明确、原子化。

默认只包含：
1. Goal
2. Tasks
3. Constraints
4. Deliverables
5. Stop

避免长背景、重复历史、开放式判断和复杂流程。需要的历史上下文通过已有专项文档引用，不复制到任务单中。

## 4. 实验脚本规则

关键实验尽量在启动前把实验定义固化到 Git 中，包括必要的：
- config；
- train/run 脚本；
- eval/scorer 脚本；
- analysis 脚本。

实验逻辑由 ChatGPT / Sol 设计；Codex 负责根据真实环境做最小适配，例如路径、CUDA、conda、远端主机和 checkpoint 位置。

环境适配不得改变实验定义、数据 split、核心超参数或评价协议。

## 5. 长时 GPU 任务

凡预计消耗 30 分钟以上 GPU 的实验，启动前应尽量满足：
- 实验配置已固定；
- 评估方法已固定；
- 分析输出已定义；
- smoke test 已通过。

Codex 启动长时任务后，只需记录命令、日志、PID、artifact 路径和 `RUNNING` 状态，不持续轮询，除非用户明确要求。

## 6. 决策边界

Codex 可以自主处理：
- 明确的工程错误；
- 路径、依赖、环境适配；
- 不改变实验语义的代码修复。

Codex 不应自主处理：
- 更换模型结构；
- 修改 loss 设计；
- 修改数据 split；
- 增加新特征；
- 改变实验变量；
- 扩展实验矩阵；
- 启动下一轮研究。

遇到这些情况应停止并报告，由 ChatGPT / Sol 决策。

## 7. 最终原则

工作流保持为：

**ChatGPT / Sol = Research Lead + Experiment Designer + Script Author + Reviewer**

**Git = Shared Memory + Experiment Protocol + Task Queue**

**Codex / Luna-medium = Repo-aware Engineer + Runner**

**Remote GPU = Compute Worker**

目标是减少 Codex 的开放式推理负担，让其以中等推理强度稳定执行短、明确、可验收的任务。
