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
- 阅读本方向任务文件和相关代码；
- 做必要的环境适配；
- 落地已明确的改动；
- smoke test；
- 启动训练、评估和分析；
- 记录实验事实、结果和 commit。

Codex 不承担开放式研究规划，不自行扩大实验范围，不自行设计下一轮实验。

## 2. Git 协作规则

远端仓库统一使用 `main`，不为普通优化实验创建长期实验 branch。

ChatGPT 和 Codex 默认都直接与 `main` 同步：
- ChatGPT 的文档、任务和明确代码修改直接写入 `main`；
- Codex 完成任务后 commit 并 push 到 `main`；
- 除非用户明确要求，不创建 branch。

任务开始前，Codex 应：
1. 确认当前工作区没有未知未提交改动；
2. 执行 `git pull --rebase origin main`；
3. 再开始本方向任务。

遇到未知未提交改动，不得擅自 stash、reset、restore、clean 或覆盖，应先报告。

任务提交前再次同步 `origin/main`。如果 rebase 出现实质性冲突，停止并报告，不自行猜测解决。

多个 Codex 可以并行，但不得同时操作同一个本地工作目录。并行 Codex 应使用彼此独立的本地 clone/工作目录，全部跟踪同一个 `origin/main`。

## 3. 优化方向文档

每个优化方向在自己的 `docs/` 目录维护两类核心文件：

- `README.md`：该方向的长期技术记忆；
- `NEXT_ACTION.md`：该方向当前唯一施工单。

例如：
- `docs/loss/README.md` 与 `docs/loss/NEXT_ACTION.md`；
- `docs/feature_engineering/README.md` 与 `docs/feature_engineering/NEXT_ACTION.md`；
- 对于子方向，可使用类似 `docs/model/point_modeling/README.md` 与 `NEXT_ACTION.md` 的结构。

### README.md
必须持续记录该方向做过的实验，包括：
- 实验目的和关键配置；
- 结果指标；
- 有效结果；
- 失败结果、`NO-GO`、`STOP`；
- 当前稳定结论；
- 后续建议。

失败实验也必须记录，避免重复试错。

方向 `README.md` 是该方向实验历史和结论的主要事实源。跨方向总表、整体概要或实验注册表只在需要全局汇总时更新，不要求每个并行实验都即时修改共享文件。

### NEXT_ACTION.md
必须简单、明确、原子化，默认只包含：
1. Goal
2. Tasks
3. Constraints
4. Deliverables
5. Stop

避免长背景、重复历史、开放式判断和复杂流程。需要的上下文通过本方向 `README.md` 或具体证据文档引用。

`docs/coordination/` 只用于跨方向状态、handoff 和协调信息，不放具体优化方向的 `NEXT_ACTION.md`。

## 4. 并行优化管线

多个优化方向可以同时执行，但每个 Codex 只负责一个明确方向。

默认只修改：
- 本方向的 `docs/` 目录；
- 本方向直接相关的代码、配置和脚本。

不要顺手修改其他方向文件，也不要使用 `git add .` 把其他管线的改动带入提交。

以下共享文件默认不由并行 Codex 随意修改，除非任务明确要求：
- 根目录 `AGENTS.md`；
- 根目录 `README.md`；
- `docs/realpde整体优化概要.md`；
- 跨方向协调或总表文档。

## 5. 实验脚本规则

关键实验尽量在启动前把实验定义固化到 Git 中，包括必要的：
- config；
- train/run 脚本；
- eval/scorer 脚本；
- analysis 脚本。

实验逻辑由 ChatGPT / Sol 设计；Codex 负责根据真实环境做最小适配，例如路径、CUDA、conda、远端主机和 checkpoint 位置。

环境适配不得改变实验定义、数据 split、核心超参数或评价协议。

## 6. 长时 GPU 任务

凡预计消耗 30 分钟以上 GPU 的实验，启动前应尽量满足：
- 实验配置已固定；
- 评估方法已固定；
- 分析输出已定义；
- smoke test 已通过。

Codex 启动长时任务后，只需记录命令、日志、PID、artifact 路径和 `RUNNING` 状态，不持续轮询，除非用户明确要求。

## 7. 决策边界

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

## 8. 最终原则

**一个远端 `main`，多个独立本地执行管线。**

**方向目录隔离任务和实验历史，`README.md` 保存长期记忆，`NEXT_ACTION.md` 驱动下一步。**

**ChatGPT / Sol = Research Lead + Experiment Designer + Script Author + Reviewer**

**Codex / Luna-medium = Repo-aware Engineer + Runner**

**Remote GPU = Compute Worker**
