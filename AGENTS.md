# Track 1 执行规则

## 必读上下文

在规划、修改代码或解读实验之前，必须阅读：

1. 工作区根目录的 `MEMORY.md`；
2. `docs/CHATGPT_CODEX_WORK_PROTOCOL.md`；
3. 本任务所属优化方向的 `README.md` 与 `NEXT_ACTION.md`；
4. 本任务明确引用的其他 `docs/` 证据文档。

需要跨方向状态时再读取 `docs/coordination/STATUS.md` 和全局概要/注册表，不要用共享总表替代本方向文档。

## ChatGPT–Codex 协作

ChatGPT / Sol 负责技术方向、实验设计、核心训练/评估/分析逻辑、结果复核和下一步决策。

Codex / Luna-medium 负责明确任务的工程落地、环境适配、smoke test、执行、结果记录和提交。不得自行扩大实验范围，不自行设计下一轮实验。

每个优化方向在自己的 `docs/` 目录维护：
- `README.md`：长期技术记忆，记录全部实验，包括有效、失败、`NO-GO` 和 `STOP`；
- `NEXT_ACTION.md`：当前施工单，必须短、明确、原子化。

`NEXT_ACTION.md` 默认只包含 Goal、Tasks、Constraints、Deliverables、Stop。具体优化方向的 `NEXT_ACTION.md` 不放在 `docs/coordination/`。

## Git 与并行执行

远端仓库统一使用 `main`。除非用户明确要求，不为普通实验创建 branch。

开始任务前：
1. 检查当前工作区是否有未知未提交改动；
2. 执行 `git pull --rebase origin main`；
3. 再开始任务。

未知未提交改动不得擅自 stash、reset、restore、clean 或覆盖，应先报告。

多个 Codex 可以并行，但不得同时操作同一个本地工作目录。并行执行应使用独立的本地 clone/工作目录，全部跟踪同一个 `origin/main`。

每个 Codex 只修改本方向 `docs/` 和直接相关代码/脚本。不要顺手修改其他方向，不要使用 `git add .` 把其他管线文件带入提交。

提交前再次同步 `origin/main`。若 rebase 出现实质性冲突，停止并报告，不自行猜测解决。完成后 commit、push，并确认工作树干净；无法提交的文件必须明确报告。

### GitHub 交付闭环

正式实验的标准交付链路固定为：

`Codex 实现/训练/评估/分析 → commit → push origin/main → ChatGPT/Sol 从 GitHub 读取证据并复核`

因此：
- 本地 commit、临时目录或仅聊天摘要都不算正式实验交付完成；
- Codex 必须在同一任务会话中把可复核的代码、轻量结果、CSV/JSON/Markdown、review evidence commit 并 push 到 `origin/main`；
- ChatGPT / Sol 默认只基于已经进入 GitHub `main` 的证据做最终科研结论，不要求用户代为搬运本地结果；
- 大 checkpoint、原始大日志等仍按 artifact 规则留在远程计算环境，Git 只保存可审计的轻量证据和路径/哈希。

凡预计进行长训练、长评估或较重分析，Codex 在启动前必须额外验证当前执行环境具有 GitHub 写权限。至少执行：

```bash
git fetch origin
git pull --rebase origin main
git push --dry-run origin HEAD:main
```

若 `git push --dry-run` 因认证、权限或 remote 配置失败：
- 不启动新的长训练/长分析；
- 先修复当前 Codex 执行环境的 GitHub authentication / remote 写权限，或停止并报告；
- 不得把“任务做完后由用户手动 push”作为正常交付方案。

已在运行中的历史任务若事后才发现认证失败，应先保护本地产物；但这属于异常恢复，不作为后续标准流程。

## 实验记录

每个实质性实验都应记录到本方向 `README.md`，至少包含：
- 实验目的与关键配置；
- 可复现命令或脚本入口；
- 核心指标/观察；
- `KEEP` / `REVIEW` / `NO-GO` / `STOP` 结论；
- 必要的 artifact 或证据路径。

失败实验同样记录，避免重复试错。

对于 Future-K / Future20 等多步时序预测，凡结果用于正式科研判断，都必须按 `docs/EXPERIMENT_BY_HORIZON_PROTOCOL.md` 补齐逐帧 / By-Horizon 分析。至少保留 overall、trajectory-level 和 `t+1...t+K` 三层证据；不得仅凭 Future20 汇总指标做最终结论。

`docs/track1_experiment_registry.md`、`docs/realpde整体优化概要.md` 等共享文档用于跨方向汇总，不要求每个并行实验都即时修改。只有任务明确要求或形成稳定跨方向结论时再更新。

不得将数据集、checkpoint、凭证、绝对私有路径或生成的 submission archive 写入 Git。

## 长时任务

启动长时间 CPU/GPU 任务前，先完成实现与 smoke test，并明确实验配置、评估方法和分析输出。

在启动 long runner 前，除实验与版本 preflight 外，还必须确认 `git push --dry-run origin HEAD:main` 成功；GitHub 写权限是长任务启动条件之一。

以 detached 方式启动 Runner，记录主机、命令、日志、PID 和 artifact 路径。确认 `RUNNING` 后停止持续轮询，仅在后续得到明确请求时回收结果。

## Git Docs 作为长期技术记忆

Git `docs/` 是 ChatGPT/Sol、Codex 与研发人员共享的长期技术记忆，聊天上下文不能替代 Git 中沉淀的结论。

方向内的信息流为：实验事实 → 本方向 `README.md` → 稳定方向结论。跨方向稳定结论再汇总到整体概要。

开始已有方向的新任务前，先读该方向 `README.md`。如果拟议实验重复已记录的 `NO-GO` / `STOP`，或与既有结论冲突，应主动指出，而不是静默重跑。
