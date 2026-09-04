# Training Log Review Protocol

本协议用于所有正式训练实验的过程证据留存与 ChatGPT / Sol 复核。

目标不是要求训练必须高频实时打印，也不是要求把远程机器上的完整日志全部提交到 Git。核心要求是：

> **每次正式训练尽量留下可供 Sol 审阅的训练过程证据，避免只剩最终几个统计指标而丢失训练轨迹。**

## 1. 基本原则

- 不规定“>30 分钟训练必须实时 logging”。不同 runner 的合理日志频率可以不同。
- 训练代码应在不显著增加 IO / 日志体积的前提下，记录足够的过程状态。
- `EVAL`、checkpoint、resume、异常、warning/error、关键状态切换等事件应完整保留。
- TRAIN 过程点不要求每 step 记录，也不固定死为每 N steps；runner 可根据总 updates 和实验性质选择合理 cadence。
- Luna / Codex 的任务是产生过程证据，不根据日志自行做最终科研决策。
- ChatGPT / Sol 根据训练日志、metrics、case-level evidence、关键实现共同完成最终复核。

## 2. 原始日志与 Git Review Log

### Raw log

完整原始训练日志保留在执行环境 / GPU 主机，不默认提交 Git。

handoff 至少记录：

- raw log path；
- raw log SHA256；
- raw log line count；
- raw log size；
- execution commit / experiment ID。

若 Sol 在 review log 中发现异常，可再要求 Codex 提交完整 raw log 或指定区间。

### Review log

每次正式训练结束后，尽量生成一个轻量 `review log` 提交到 Git，供 Sol 首轮审阅。

默认规则：

1. 如果 raw log 足够小，例如同时满足：
   - `<= 1000` 行；
   - `<= 200 KiB`；

   则直接把完整 raw log 作为 review log。

2. 如果 raw log 较大，则生成**确定性压缩**版本，不随机抽样。至少保留：
   - 开头 20 行；
   - 结尾 20 行；
   - 所有 `EVAL` / checkpoint / save / resume / warning / error / exception / traceback / OOM / NaN / Inf / abort / stop 等关键事件；
   - TRAIN 行中按时间 / 行号等距抽取约 30–50 个代表点；若没有明确 TRAIN 行，则从其余普通日志中等距抽样。

3. review log 同时生成 metadata JSON，记录：
   - raw/review line count；
   - raw/review size；
   - raw SHA256；
   - selection mode；
   - event line count；
   - sampled line count。

禁止让 Luna 每次临时“随机挑一些看起来重要的行”。选择规则应可重复、可审计。

## 3. Runner 应记录什么

具体训练脚本不要求统一格式，但正式实验尽量包含以下证据：

- run start / experiment ID / commit / seed / checkpoint /主要配置；
- 若有 TRAIN 过程日志：update、主要 loss / loss term、LR，必要时 throughput；
- 所有正式 EVAL：update + raw Rel-L2 / TKE / MVPE 或本实验指定指标；
- checkpoint save / resume / stage transition；
- warning / error / NaN / OOM / early stop；
- 对新 branch / head / feature / loss 等高风险改动，按实验契约记录对应的 activation / invariant evidence，例如 branch output、gradient、parameter delta 或 loss term。

这些字段不是要求每个 step 都输出。重点是让训练结束后能够重建“模型是如何走到最终结果的”。

## 4. Sol Review

Sol 首轮复核默认读取：

1. review log + metadata；
2. update / eval curve；
3. raw metrics 与 case-level evidence；
4. execution commit 和关键实现；
5. 实验专项 invariant / activation evidence。

如果 review log 暴露以下问题，应进一步读取完整 raw log 或局部区间，而不是直接接受最终指标：

- loss / metric 突变；
- checkpoint / resume 语义异常；
- 新 branch 长时间不激活；
- EVAL 与 TRAIN 走势矛盾；
- NaN / OOM / silent recovery；
- 中途配置或阶段发生未经授权的变化；
- 日志不足以证明实验语义实际生效。

## 5. 标准工具

仓库提供：

```bash
python tools/build_training_review_log.py \
  --input /path/to/train.log \
  --output /path/to/train.review.log
```

脚本默认：

- 小日志完整复制；
- 大日志保留头尾、全部关键事件和确定性等距 TRAIN 抽样；
- 在 review log 同目录生成 `<output>.meta.json`。

该工具只做日志取证压缩，不改变训练结果，也不替代实验-specific analysis。
