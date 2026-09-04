# Experiment Analysis Skill

## Purpose

让每个重要实验在训练 / 评估完成后，顺手完成必要的数据分析并一次性提交，减少“看总指标 → 猜原因 → 再补分析”的往返。

默认研究闭环：

`Hypothesis → Controlled Experiment → Error Anatomy → Mechanism Hypothesis → Next Experiment`

后续实验任务若无特殊说明，Codex 默认读取：

- `docs/data/SKILL.md`
- `docs/data/DATASET_PROFILE.md`
- 本文件

并在训练完成后复用 prediction / checkpoint artifact 完成与本实验相关的分析。

## Default analysis

### Level 0: every formal experiment

至少保留：

- official raw metrics
- matched baseline delta
-关键 checkpoint / update 曲线
- trajectory-level win count（paired A/B 时）
- experiment provenance：ID / split / seed / init / commit / scorer / manifest
- `GO / SUPPORTIVE / NO_GO / REVIEW_REQUIRED`

### Level 1: experiments that affect research decisions

按任务相关性尽量完成，不机械堆图：

1. **By-Trajectory**：逐 Dev trajectory 的 baseline / candidate / delta，找 good / bad case。
2. **Distribution Context**：加载 `DATASET_PROFILE.md`，标注关键 good / bad case 位于 Train 分布的 `IN_DISTRIBUTION / BOUNDARY / OOD_LIKE` 哪一类。
3. **By-Horizon**：Future20 任务查看 `t+1 ... t+20` 的误差 / 统计量，检查 drift、variance collapse、phase / amplitude degradation。
4. **By-Spatial**：target / prediction / error map，以及 candidate-baseline improvement map；按高低能量或物理区域总结。
5. **Good / Bad Case**：选择少量代表 case 深挖，不按单条 trajectory 定制模型。

## Level 2: metric conflict / important NO-GO

如果出现明显指标冲突或重要假设失败，进入下一轮建模前优先做 prediction-level mechanism analysis，尽量复用已有 artifact，不重新训练。

目标是区分：

- field reconstruction vs statistics / energy divergence
- amplitude calibration vs spatial structure error
- short-horizon vs long-horizon failure
- low-energy vs high-energy case failure
- in-distribution vs boundary / OOD-like failure
- few-outlier driven vs broad trajectory-level failure
- temporal / spectral structure error（仅在前述分析不足时增加）

## Experiment-specific analysis

一般框架之外，按实验假设增加专项分析。

### Modeling / target decomposition

- component-wise error
- reconstruction / structural invariant
- branch attribution
- Mean / Fluctuation：Mean error、波动场误差、Fluctuation RMS、energy ratio；必要时 optimal gain / temporal spectrum

### Loss

- each loss-term magnitude
- gradient contribution / conflict（若可稳定获得）
- trajectory-level metric trade-off
- 优化了哪个 term、伤害了哪个统计量

### Feature Engineering

- feature Train / Dev distribution and tail coverage
- redundancy / correlation
- incremental value by case
- 不只报告 concat 后 overall score

### Training / continuation

- metric vs update
- checkpoint stability
- plateau / overfit / noise
- trajectory-level stability

### OOD / Sim2Real / generalization

- distribution distance / coverage
- case cluster / nearest-neighbor
- error vs distribution position

### Calibration / SPS / bounds

- coverage-width trade-off
- per-case coverage
- calibration curve / tail behavior

### Runtime / inference

- numerical equivalence or metric regression
- speed / memory / package size

## One-pass execution rule

实验任务在启动前应尽量定义分析输出。训练结束后，Codex 默认按以下顺序一次完成：

1. official evaluation;
2. default analysis;
3. experiment-specific analysis;
4. update handoff / direction README / required registry;
5. commit and push once;
6. stop at `REVIEW_REQUIRED` unless next experiment was explicitly pre-authorized.

若 prediction artifact 已存在，分析优先离线 replay，不重复训练。

不要为了形式完整而做无关分析。只有能够解释结果、判断泛化或改变下一实验决策的分析才应加入。

## Decision output

最终 handoff 应明确区分：

- **Verified facts**：overall + case / diagnostic evidence 支持的事实；
- **Mechanism hypotheses**：数据提示但尚未由控制实验验证；
- **Next discriminating experiment**：下一步最小单变量实验，用于区分最关键的竞争解释。

禁止根据 locked-final、单个 Dev bad case 或事后挑选的局部规律设计 trajectory-specific trick。
