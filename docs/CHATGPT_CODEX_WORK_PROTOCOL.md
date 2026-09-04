# ChatGPT–Codex 工作协议

## 1. 角色分工

### ChatGPT / Sol
负责需要较强研究判断的工作：
- 技术方向与优先级；
- 科学假设、实验设计与变量控制；
- 明确定义模型 / Loss / Feature / 数据 / 评估的实验语义；
- 冻结 baseline、唯一变量、预算、Gate、停止条件和禁止项；
- 根据语义风险决定核心代码由 ChatGPT 直接实现，还是交给 Codex 做 bounded 实现；
- 结果复核与下一步决策。

### Codex / Luna-medium
负责仓库和环境内的工程实现与执行：
- 阅读本方向任务文件和相关代码；
- 在实验语义已经明确的前提下，完成 bounded 代码实现；
- 做必要的仓库 / 环境适配；
- unit test / TDD / smoke test；
- 启动训练、评估和分析；
- 记录实验事实、结果和 commit。

Codex 可以写实验代码，但不承担开放式研究规划，不自行改变实验含义，不自行扩大实验范围，不自行设计下一轮实验。

### 代码作者选择原则

不机械规定“实验代码必须由 ChatGPT 写”或“必须由 Codex 写”。默认按 **语义风险** 和 **工程上下文复杂度** 选择：

**优先由 ChatGPT / Sol 直接实现：**
- 改动很小、接口清楚，但数学 / 实验语义高度关键；
- 一处翻译歧义就可能污染实验结论；
- 例如小型 loss / metric 公式、projection / decomposition、关键 scorer helper、简单 feature 定义、明确的 protocol 常量。

**优先由 Codex 实现：**
- 科学语义已经被 ChatGPT 完整冻结；
- 主要难点是理解现有 repo、复用 pipeline、checkpoint / dataloader / runner 接口、测试和真实环境集成；
- 属于单变量、边界清楚的 bounded 实验改动。

**混合模式：**
- ChatGPT 给出精确数学定义、invariant / equivalence test、baseline 与 Gate；
- Codex 在现有代码结构中完成最小集成，并用测试证明实现符合定义。

若实现过程中出现会改变实验语义的歧义，Codex 必须停止并报告，不自行选择新的定义。

核心原则：**谁敲代码不是关键，ChatGPT / Sol 拥有实验语义和研究决策权；Codex 拥有 repo-aware 实现、验证和执行权。**

### 实现分工是软约束，最终结论由 Sol 审核

代码作者按任务性质灵活选择，不以“Sol 写研究代码、Luna 只跑脚本”做机械划分。ChatGPT / Sol 不掌握远程 GPU 机器上的全部真实路径、环境、checkpoint 和 artifact 状态，因此凡实现高度依赖 repo / GPU runtime context 时，优先由 Codex / Luna-medium 在冻结实验语义后完成最小实现、环境适配和执行；凡数学定义、实验语义或一个小实现错误就可能直接污染结论时，优先由 ChatGPT / Sol 给出实现或 reference behavior / invariant。

无论代码由谁实现，**最终科研结论默认由 ChatGPT / Sol 复核后生效**。Codex / Luna-medium 的 `PROMISING / WEAK_SIGNAL / NO_GO / KEEP / STOP` 等标签只视为执行方初步摘要，不直接成为项目决策。

因此重要实验的 Codex task 应由 Sol 预先定义验收与审计标准，并要求 Codex 优先交付可复核的过程证据，而不是只交一段总结。根据实验类型，证据可包括：
- raw metrics、matched delta、trajectory / horizon / spatial 数据；
- branch / component output、gradient、parameter norm / delta、loss term 等机制数据；
- checkpoint、manifest、scorer、commit、runtime snapshot 等 provenance；
- 关键代码 commit / 文件位置，以及必要的 invariant / equivalence / smoke 结果；
- 轻量 CSV / JSON / Markdown / representative figures，便于 Sol 直接从 GitHub 复核。

如果执行方结论与过程数据冲突，以过程数据和 Sol 复核为准。若关键证据不足、实现语义未被证明、branch / loss / feature 是否真正生效无法确认，则结论应先标记为 `REVIEW_REQUIRED / INVALID / NEED_MORE_EVIDENCE`，不得仅依据 overall 指标推进或关闭方向。

### 新一级研究会话的默认入口

新开 Modeling、Loss、Feature Engineering、Data、Sim2Real、OOD、Training 等一级研究会话时，ChatGPT / Sol 默认先读取当前 `main` 的三层信息，而不是只看最近一次实验：

1. **战略地图**：`docs/realpde整体优化概要.md`
2. **当前线上战况**：`docs/sota迭代/README.md`，并按需读取 `docs/submission_log.md` 和当前主线最新 handoff
3. **专项研究记忆**：对应方向的 `README.md` / `*概要.md`，再按需追踪 experiment registry、coordination handoff 和 STATUS

新会话必须同时保留两种视角：

- **Exploration**：检查是否存在新的任务定义、模型结构、表示、训练或物理方向，可以打开新的性能曲线；
- **Competition relevance**：判断新方向未来如何与当前线上主线衔接，而不是在真空中设计方案。

原则：不能因为当前 SOTA 是 CNO 就只围绕 CNO 微调，也不能因为做战略探索就忽略当前 SOTA、已失败实验和已有线上/离线证据。

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

### 执行冻结与版本锁定

GPU / 长时任务启动前，必须同时冻结两件事：

1. **实验契约**：Goal、baseline、唯一变量、数学 / forward 语义、数据与初始化、Loss、预算、checkpoint 规则、评估、数据分析计划、Gate、停止条件、禁止项；
2. **实际执行代码版本**：启动时使用的 Git commit。

根据实现方式分两种模式。

#### 模式 A：ChatGPT 已完成核心实现

当 ChatGPT / Sol 已完成本次任务所需的核心代码、测试 / launcher 和 `NEXT_ACTION.md` 后，应明确给出：

- `READY_FOR_EXECUTION`
- `REQUIRED_COMMIT = <sha>`

从 `READY_FOR_EXECUTION` 到 Codex 完成启动检查期间，ChatGPT / Sol 默认不再修改该任务的核心代码、launcher 或实验定义。若必须修改，应撤销上一版执行授权并给出新的 `REQUIRED_COMMIT`。

#### 模式 B：Codex 负责 bounded 实现并无人值守执行

如果 ChatGPT / Sol 已经完整冻结实验语义，但决定让 Codex 根据现有 repo 完成实现，可明确给出：

- `IMPLEMENT_AND_EXECUTE_AUTHORIZED`
- 必要时给出 `REQUIRED_BASE_COMMIT = <sha>`

Codex 可以在该实验契约内完成代码、TDD / unit test、smoke、commit 和 push；若所有预注册测试与 preflight 通过，可直接启动预授权的 GPU / 长时任务，不要求用户再次在线确认。

正式启动前 Codex 必须记录：

- `EXECUTION_COMMIT = <sha>`
- 实现相对实验契约没有语义漂移；
- 关键 invariant / equivalence / smoke 已通过。

如果实现需要改变模型定义、Loss、Feature、split、初始化、预算、Gate 或其他实验语义，授权立即失效，停止并报告。

### 启动前版本检查

Codex 在正式 tests / smoke / launch 前必须执行：

```bash
git fetch origin
git pull --rebase origin main
git rev-parse HEAD
git rev-parse origin/main
```

模式 A 还必须执行：

```bash
git merge-base --is-ancestor "$REQUIRED_COMMIT" HEAD
```

模式 B 若指定了 `REQUIRED_BASE_COMMIT`，则执行：

```bash
git merge-base --is-ancestor "$REQUIRED_BASE_COMMIT" HEAD
```

启动条件：
- `HEAD == origin/main`；
- required commit / base commit（若有）是当前 `HEAD` 的祖先；
- runtime snapshot / preflight / tests / smoke 满足本任务要求；
- 实验契约没有发生未经授权的变化。

任一条件不满足，停止并报告，不启动 GPU 长任务。

实验 handoff 必须记录**实际执行 commit SHA**。运行中的任务不因 `main` 后续出现与该任务无关的新 commit 而自动停止或切换代码；是否需要重启由 ChatGPT / Sol 根据 diff 决定。

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

### 实验语义必须由 ChatGPT / Sol 冻结

不论最终代码由谁实现，以下内容不得留给 Codex 开放式决定：
- loss 与指标的数学定义；
- feature 构造与 fusion 语义；
- 模型结构 / prediction target / forward 的核心语义；
- checkpoint / optimizer resume 语义；
- LR schedule 与训练阶段切换规则；
- 数据 split、采样和窗口协议；
- A/B 唯一变量、预算、停止条件与 checkpoint 规则；
- official scorer、结果分析口径与关键实验 Gate。

### 实现可以由 ChatGPT 或 Codex 完成

**ChatGPT 直接实现模式**适合小而关键的实验语义代码。ChatGPT 应尽量同时提供 reference behavior / invariant test，Codex 再负责真实 repo 和环境中的集成、测试和执行。

**Codex bounded 实现模式**适合语义已经完全明确、但需要较多 repo 上下文的改动。Codex 可以阅读现有实现、选择最小代码落点、写 unit test / TDD、完成 runner / checkpoint / dataloader / evaluation 集成，只要不改变冻结的实验契约。

对于类似 Mean / Fluctuation projection、loss 公式、metric 这类语义敏感点，如果由 Codex 实现，应优先用以下方式约束：
- 精确数学定义；
- zero-mean / shape / reconstruction 等 invariant test；
- baseline-preserving initialization / numerical-equivalence smoke（若适用）；
- matched control。

Codex 不得因为实现方便而自行换一种数学定义。若当前 repo 结构无法在合理最小改动内满足实验契约，停止并报告，不做“近似实现”。

环境适配不得改变实验定义、数据 split、核心超参数或评价协议。若环境问题必须修改实验语义，Codex 应停止并报告，由 ChatGPT / Sol 决策。

### 实验数据分析框架

实验结果不能只停留在 overall Rel-L2 / TKE / MVPE 等汇总指标。默认研究闭环应为：

`Hypothesis → Controlled Experiment → Error Anatomy → Mechanism Hypothesis → Next Experiment`

其中 `Error Anatomy`（误差解剖）用于回答“为什么好 / 为什么坏”，下一轮实验应尽量由诊断证据驱动，而不是只根据总指标猜测原因。

#### 数据角色与边界

- **50 train trajectories**：用于训练，也作为数据分布参考。当前模型在 train 上的 in-sample error 不能直接作为泛化证据；若需要验证某个 failure mode 是否可泛化，应优先使用 trajectory-level held-out / OOF 预测。
- **16 dev trajectories**：默认研发分析集，可以做逐轨迹、逐预测时刻、逐空间区域、good / bad case 和机制诊断。
- **16 locked-final trajectories**：不用于方法选择和研发 bad-case 分析。只有在模型结构、Loss、Feature、超参和 checkpoint 选择规则完全冻结后，才用于一次性 generalization audit；不得根据 final case 再回头修改方法。
- **Input-side descriptors**：仅由可用输入 Past20 计算，例如 mean/std、TKE proxy、delta、vorticity、strain、spatial gradient、temporal spectrum 等，可用于 Train / Dev 分布、coverage 和 OOD-like 分析。
- **Target-side descriptors**：由 Future20 真值计算，例如 true future TKE、future fluctuation RMS、future spectrum，只用于事后解释，不能混入可部署 Feature 或推理逻辑。

#### Level 0：每个实验的最小分析

每个正式实验至少应尽量保留：
- official raw metrics 及相对 baseline 的 matched delta；
- 关键 checkpoint / update 的训练与验证曲线；
- experiment ID、split、seed、初始化、commit / scorer / manifest 等 provenance；
- 若是 paired A/B，报告 trajectory-level win count，而不只看 macro average；
- 明确结论属于 `GO / SUPPORTIVE / NO_GO / REVIEW_REQUIRED` 中哪一类。

#### Level 1：一般有意义的实验默认分析

当实验会影响下一步研究决策时，优先补充以下分析，按任务相关性选择，不要求机械全做：

1. **Train / Dev Case Distribution**
   - 用 input-side descriptors 比较 50 Train 与 16 Dev 的分布；
   - 查看 quantile、range overlap、tail coverage、nearest-neighbor distance 或低维投影；
   - 标记 Dev case 是 `in-distribution / boundary / OOD-like`，避免把数据分布差异误判成模型机制问题。

2. **By-Trajectory Analysis**
   - 对每条 Dev trajectory 报告 baseline、candidate、delta；
   - 排序找出典型 good case / bad case；
   - 检查收益或退化是普遍现象，还是由少数轨迹驱动。

3. **By-Horizon Analysis**
   - 对 Future20 等时序任务，查看 `t+1 ... t+20` 的误差或关键统计量变化；
   - 检查 long-horizon drift、variance collapse、phase / amplitude degradation 等现象。

4. **By-Spatial Analysis**
   - 对场预测任务，优先查看 target / prediction / error map，以及 candidate - baseline improvement map；
   - 按物理相关区域或高低能量区域总结误差，判断问题是全局幅度、局部区域还是空间结构。

5. **Good / Bad Case Analysis**
   - 选择少量代表性 case 深挖，不按单个 case 定制模型；
   - 每个 bad case 同时回答“模型为什么错”和“这个 case 在 Train 分布中的什么位置”。

#### Level 2：出现指标冲突或重要 NO-GO 时的机制诊断

对于高价值实验，如果出现类似 `Rel/MVPE 改善但 TKE 恶化`、平均指标改善但部分 case 明显退化、或重要假设 `NO_GO`，原则上在进入下一轮建模前先做 prediction-level 深入诊断，优先复用已有 prediction artifact，不重新训练。

诊断目标不是堆更多图，而是区分不同 failure mechanism。例如：
- `field reconstruction` 与 `energy/statistics` 是否出现背离；
- error 是否集中在高波动、高梯度、长预测 horizon 或 OOD-like case；
- 是全局 amplitude calibration 问题，还是 spatial / temporal / spectral structure 问题；
- 是少数 outlier 驱动，还是跨 trajectory 稳定存在。

只有诊断结果会改变下一步实验设计时，才增加对应分析；不为低价值 probe 建立庞大分析流水线。

#### 特殊实验增加专项分析

一般分析框架之外，不同实验类型应增加与其假设直接相关的专项诊断：

- **Modeling / Prediction Target / Decomposition**：component-wise error、结构 invariant、reconstruction equivalence、分支或表示之间的误差归因。Mean / Fluctuation 类实验应额外看 Mean error、波动场误差、Fluctuation RMS、energy ratio、必要时 optimal gain / temporal spectrum。
- **Loss 实验**：除最终指标外，关注各 loss term 的量级、梯度贡献 / 冲突、trajectory-level trade-off，以及“优化了哪个 term、伤害了哪个物理统计量”。
- **Feature Engineering**：关注 feature 的 Train / Dev 分布、tail / OOD coverage、冗余与相关性、增量价值，以及 feature 收益是否只出现在特定 case；避免只看 concat 后总分。
- **Training / Continuation**：关注 metric-vs-update 曲线、checkpoint stability、plateau / overfit、不同 trajectory 的稳定性，区分“继续训练有收益”与“只是在 checkpoint noise 中挑点”。
- **OOD / Sim2Real / Generalization**：重点看 distribution distance、coverage、case cluster、nearest-neighbor / tail behavior，以及 error 与 distribution position 的关系。
- **Calibration / SPS / Bounds**：关注 coverage-width trade-off、per-case coverage、calibration curve 和分布尾部，不把 calibration 问题误判为 backbone 问题。
- **Inference / Runtime 优化**：除速度 / 显存外，必须做 numerical equivalence 或 metric regression 检查，确认工程优化没有改变实验语义。

#### 从分析到下一轮实验

ChatGPT / Sol 在重要实验后应尽量明确区分：
- **已验证事实**：由 overall + case-level / diagnostic evidence 支持；
- **机制假设**：由数据提示但尚未被控制实验验证；
- **下一实验唯一变量**：用于区分最关键的竞争解释。

不因为单个 bad case 设计 trajectory-specific trick，也不因为 16 Dev 的局部规律假设 private test 必然同分布。优先寻找跨 trajectory、跨 case 分布稳定成立的机制。

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
- 实验契约已固定；
- 评估方法已固定；
- 分析输出已定义；
- runtime snapshot 已覆盖关键远程事实；
- unit / invariant / smoke test 已通过。

对于已经明确 `IMPLEMENT_AND_EXECUTE_AUTHORIZED` 的 bounded 无人值守任务，只要实现没有语义漂移、测试 / preflight 通过、版本锁定条件成立，Codex 可以按预授权直接启动，不需要等待用户再次确认。

Codex 启动长时任务后，只需记录命令、日志、PID、artifact 路径和 `RUNNING` 状态，不持续轮询，除非用户明确要求。

长时任务完成后，应确保 `artifact_manifest.json` 已生成，并在 handoff 中记录 resumable checkpoint 清单。

## 8. 决策边界

Codex 可以自主处理：
- 明确的工程错误；
- 路径、依赖、环境适配；
- 实验契约内的 bounded 代码实现；
- 不改变实验语义的代码修复；
- unit test / TDD / invariant / equivalence smoke；
- runtime snapshot 和 artifact manifest 的事实盘点。

Codex 可以**实现** ChatGPT 已明确授权的模型结构、Loss、Feature 或 prediction target 改动，但不得自主**决定或改变**这些研究变量。

Codex 不应自主决定：
- 更换或扩展模型结构；
- 修改 loss 设计或权重；
- 修改数据 split；
- 增加新特征；
- 改变实验变量、预算、Gate；
- 扩展实验矩阵；
- 用其他 checkpoint 替代任务指定 checkpoint；
- 因中途指标自行调参；
- 启动下一轮研究。

遇到这些情况应停止并报告，由 ChatGPT / Sol 决策。

## 9. 最终原则

**一个远端 `main`，多个独立本地执行管线。**

**方向目录隔离任务和实验历史，`README.md` 保存长期记忆，`NEXT_ACTION.md` 驱动下一步。**

**新一级研究会话先读“战略地图 → SOTA 战况 → 专项记忆”，同时保留 Exploration 与 Competition relevance 两种视角。**

**远程环境事实先盘点，再设计依赖这些事实的实验；长任务结束后留下可机器读取的 artifact manifest。**

**实验先冻结语义，再决定由谁写代码；代码作者不是研究决策者。**

**Codex / Luna-medium 优先交付过程证据和可审计实现，不以其总结标签替代科研结论；重要实验由 ChatGPT / Sol 基于 Git 中的证据和关键实现完成最终复核。**

**重要实验不只看总指标；先做与假设相匹配的分层误差分析，再决定下一轮唯一变量。**

**对于 bounded 无人值守实验，ChatGPT 冻结实验契约，Codex 可以实现、测试并按预授权直接执行；任何语义漂移都必须停止。**

**ChatGPT / Sol = Research Lead + Experiment Semantic Owner + Final Reviewer + Selective Core Code Author**

**Codex / Luna-medium = Repo-aware Implementation Engineer + Evidence Producer + Test / Integration Engineer + Runner**

**Remote GPU = Compute Worker**