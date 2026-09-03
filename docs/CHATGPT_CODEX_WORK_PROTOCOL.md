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

凡涉及**实验语义**的代码，默认由 ChatGPT / Sol 直接实现并进入 Git，不只提供伪代码让 Codex 自行翻译。包括但不限于：
- loss 与指标计算；
- feature 构造与 fusion；
- 模型结构与 forward 语义；
- checkpoint / optimizer resume 语义；
- LR schedule 与训练阶段切换；
- 数据 split、采样和窗口协议；
- A/B 变量控制、停止条件与 checkpoint 规则；
- official scorer、结果分析与关键实验校验。

Codex 不根据伪代码自行补全或重写这些核心实验逻辑。Codex 主要负责真实环境中的最小工程适配、集成、测试和执行，例如路径、CUDA、conda/container、远端主机、checkpoint 实际位置、日志和 detached runner。

环境适配不得改变实验定义、数据 split、核心超参数或评价协议。若环境问题必须修改实验语义，Codex 应停止并报告，由 ChatGPT / Sol 决策和修改代码。

## 6. Runtime Context 与 Artifact 规则

当任务依赖远程 GPU 机器上的真实环境、数据目录、starting kit、checkpoint 或历史 artifact 时，不再由 ChatGPT / Sol 根据聊天记录猜测这些事实。

仓库提供轻量工具：

`tools/realpde_runtime_context.py`

### Runtime Snapshot

在以下情况之一发生时，应优先刷新 runtime snapshot：
- 新的 SOTA / 长训练任务依赖远程 GPU 环境；
- continuation 依赖某个历史 checkpoint；
- 任务依赖特定数据规模、scorer 或 artifact 是否存在；
- 上一次执行因路径、checkpoint、数据或环境事实不成立而停止。

snapshot 至少应记录：
- GPU、CUDA/PyTorch 等运行环境事实；
- 数据根目录、trajectory 数、window 数、空间 shape；
- starting-kit 路径与 `scoring.py` SHA256；
- 已扫描 checkpoint 的路径、SHA256、iteration、feature config、loss config、是否包含 optimizer state。

示例：

```bash
python tools/realpde_runtime_context.py snapshot \
  --data-root "$DATA_ROOT" \
  --kit-root "$KIT_ROOT" \
  --checkpoint "$RESUME_CHECKPOINT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --output "$OUT_ROOT/runtime_snapshot.json"
```

runtime snapshot 默认保存在远程 artifact 目录，不要求提交绝对机器路径到 Git。Codex 应把影响实验设计的关键事实反馈给 ChatGPT / Sol。若这些事实决定实验是否成立，应先反馈 snapshot，再由 ChatGPT / Sol 最终确定实验任务。

### Artifact Manifest

凡长训练、长评估或会被后续 continuation 复用的 run，结束后应生成：

`artifact_manifest.json`

至少记录：
- run metadata 与 terminal status；
- 实际落盘 checkpoint；
- 每个 checkpoint 的 update、SHA256；
- 是否包含 optimizer state；
- feature/loss config；
- 哪些 checkpoint 可以安全 resume。

可使用：

```bash
python tools/realpde_runtime_context.py manifest --run-dir "$RUN_DIR"
```

后续 handoff / README 必须明确区分：
- **metrics-only evaluation point**；
- **inference-only checkpoint**；
- **resumable checkpoint**。

不得因为某个 update 有 dev 指标，就假设该 update 存在可恢复 checkpoint。

### Continuation 资产选择

continuation 应优先基于 snapshot / artifact manifest 中已经确认存在的资产，不根据目录名或历史文字推断。

必要时可用语义 resolver，例如：

```bash
python tools/realpde_runtime_context.py resolve \
  --snapshot runtime_snapshot.json \
  --iteration 15300 \
  --feature-set P0-A \
  --require-optimizer-state
```

若没有唯一匹配资产，应停止并报告，不得自行换成相邻 update。

对于历史 checkpoint continuation，checkpoint 内已经记录并被历史训练实际使用的 feature config、loss config、optimizer state 等语义优先。当前代码或当前数据重新推导出的值只用于兼容性检查，不得在没有明确实验设计的情况下重定义历史训练语义。

## 7. 长时 GPU 任务

凡预计消耗 30 分钟以上 GPU 的实验，启动前应尽量满足：
- 实验配置已固定；
- 评估方法已固定；
- 分析输出已定义；
- runtime snapshot 已覆盖关键远程事实；
- smoke test 已通过。

Codex 启动长时任务后，只需记录命令、日志、PID、artifact 路径和 `RUNNING` 状态，不持续轮询，除非用户明确要求。

长时任务完成后，应确保 `artifact_manifest.json` 已生成，并在 handoff 中记录 resumable checkpoint 清单。

## 8. 决策边界

Codex 可以自主处理：
- 明确的工程错误；
- 路径、依赖、环境适配；
- 不改变实验语义的代码修复；
- runtime snapshot 和 artifact manifest 的事实盘点。

Codex 不应自主处理：
- 更换模型结构；
- 修改 loss 设计；
- 修改数据 split；
- 增加新特征；
- 改变实验变量；
- 扩展实验矩阵；
- 用其他 checkpoint 替代任务指定 checkpoint；
- 启动下一轮研究。

遇到这些情况应停止并报告，由 ChatGPT / Sol 决策。

## 9. 最终原则

**一个远端 `main`，多个独立本地执行管线。**

**方向目录隔离任务和实验历史，`README.md` 保存长期记忆，`NEXT_ACTION.md` 驱动下一步。**

**远程环境事实先盘点，再设计依赖这些事实的实验；长任务结束后留下可机器读取的 artifact manifest。**

**ChatGPT / Sol = Research Lead + Experiment Designer + Core Experiment Code Author + Reviewer**

**Codex / Luna-medium = Repo-aware Integration Engineer + Runner**

**Remote GPU = Compute Worker**
