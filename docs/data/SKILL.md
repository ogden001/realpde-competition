# Dataset Analysis Skill

## Purpose

建立一份与单次实验无关、可被所有后续实验复用的 Track 1 数据分布基线。后续 Modeling / Loss / Feature / Training / OOD 等实验在分析 case 时，默认先读取本 Skill 与 `DATASET_PROFILE.md`，避免只凭单次实验的总指标猜测原因。

## Persistent output

- `docs/data/DATASET_PROFILE.md`：当前冻结数据集的长期分布画像。
- 当 split / manifest / window protocol / 数据版本发生变化时，必须刷新该文件。
- locked-final 不进入研发分布画像，不做 bad-case 分析。

## Dataset roles

- 50 Train：训练数据 + 分布参考。
- 16 Dev：研发分析集，可做 case / horizon / spatial / mechanism analysis。
- 16 Locked-final：方法与 checkpoint 完全冻结后的一次性泛化审计，不用于研发决策。

## Dataset profile minimum scope

### 1. Basic inventory

记录 manifest SHA、trajectory 数、window 数、shape、窗口协议、可用通道和数据边界。

### 2. Input-side trajectory descriptors

只使用 Past20 可获得信息，为每条 Train / Dev trajectory 生成可解释描述，优先包括：

- mean / std of u,v
- speed statistics
- temporal delta statistics
- input TKE proxy / fluctuation RMS
- spatial gradient
- vorticity / strain
- high-energy area ratio
- temporal spectrum low/mid/high-band energy ratio（若稳定可算）

这些描述可用于后续 distribution / OOD / case analysis，也可以在未来成为候选 conditioning，但本 Skill 本身不做建模决策。

### 3. Train vs Dev distribution

对关键 descriptor 报告：

- median / p10 / p25 / p75 / p90 / p95 / min / max
- Train / Dev range overlap
- Dev 超出 Train p95 或 min/max 的 case
- trajectory nearest-neighbor distance 或等价 coverage score
- 可选 PCA / 低维 projection，用于整体观察，不作为唯一结论

对每条 Dev trajectory 给出粗粒度标签：

- `IN_DISTRIBUTION`
- `BOUNDARY`
- `OOD_LIKE`

标签规则必须基于 Train-only 统计并记录，不允许人工按模型误差倒推标签。

### 4. Target-side descriptors for analysis only

可额外统计 Future20 真值的：

- true TKE
- future fluctuation RMS
- future mean
- future spectrum

这些只用于事后理解 case，不能进入 inference feature、split 选择或模型输入。

## Generalization rule

后续 bad case 必须同时回答：

1. 模型在哪个 metric / horizon / spatial region 上失败？
2. 该 case 在 50 Train 分布中的位置是什么？

不要因为单条 Dev trajectory 定制模型。优先寻找跨 trajectory、跨分布区间稳定存在的 failure mode。

## Refresh conditions

仅在以下情况重新生成完整 profile：

- manifest / split 改变；
- window protocol 改变；
- 输入数据定义改变；
- 发现当前 profile 缺失会实质影响研究判断的重要 descriptor。

普通模型实验不重复做全量 dataset profiling，只加载已有 `DATASET_PROFILE.md` 并做与本实验相关的增量 case analysis。
