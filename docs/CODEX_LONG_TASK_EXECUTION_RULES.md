# Codex 长时任务执行规则

本文补充 `docs/CHATGPT_CODEX_WORK_PROTOCOL.md` 的长时 GPU 任务原则，目标是减少无意义的 Agent 轮询、token / tool-call 消耗和人工协调负担。

## 1. 长任务默认无人值守

训练、长评估或数据分析一旦完成启动前检查并确认正常运行，应视为 Compute Worker 自主执行阶段。

Codex 只负责：

1. 启动任务；
2. 做一次启动健康检查；
3. 记录 `EXECUTION_COMMIT`、命令、PID / container ID、日志路径、artifact 路径和 `RUNNING` 状态；
4. 等任务完成后再读取结果、分析、写文档并提交。

不要把 Codex 当作 GPU 监控器。

## 2. 禁止高频轮询

除非用户明确要求，正常运行的长任务不得每几十秒或每几分钟执行：

- `nvidia-smi`；
- `docker ps`；
- `ps`；
- `tail` 日志；
- checkpoint / milestone 文件存在性检查。

默认规则：

- `<10 min` 的任务可以 blocking wait；
- `10–60 min` 的任务启动健康检查后，原则上至少间隔 `10–15 min` 才允许再次检查；
- `>60 min` 的任务优先 detached / resumable runner，Codex 不持续在线监控；
- 能由脚本一次完成 `train → eval → analysis → summary` 的，不在中间 milestone 唤醒 Agent。

如果 shell 自身需要等待，优先让单个 shell 调用内部 `sleep` / `wait`，而不是由 Codex 反复 reasoning → tool call → reasoning。

## 3. 长任务必须与 Codex 会话生命周期解耦

预计超过约 10 分钟的远程任务，优先使用不会因 Codex / SSH 会话结束而退出的执行方式，例如：

- detached container；
- `nohup` / 后台 runner；
- 现有项目的 detached / resumable runner。

启动后应能独立落盘：

- log；
- checkpoint；
- run metadata；
- terminal status / summary。

如果当前任务仍附着在前台 SSH / tool call 上，不应直接结束 Codex 会话。应先确认任务已与客户端生命周期解耦。

## 4. 异常处理

正常任务不需要主动盯守。异常通过以下事实暴露：

- non-zero exit code；
- failure marker；
- terminal status；
- 日志中的明确错误；
- 缺失预期 summary / artifact。

只有出现明确异常，Codex 才重新进入调试流程。

## 5. 人工协调模式

如果用户愿意自己看 GPU 利用率或远程状态，可以在任务已 detached 后结束当前 Codex 执行会话。

任务完成后重新进入 Codex，要求它：

1. 读取已有 run artifacts；
2. 执行预注册评估 / 数据分析；
3. 更新 handoff / README / registry；
4. commit + push；
5. 一次性返回结果。

不需要保持原 Codex 会话持续在线。

## 6. 核心原则

**Remote GPU 是 Compute Worker，Codex 是实现与收尾 Agent，不是持续监控 Agent。**

**启动成功后尽量让计算和 Agent 生命周期解耦；减少轮询，把 token 和人工注意力留给实验设计、结果复核和下一步决策。**
