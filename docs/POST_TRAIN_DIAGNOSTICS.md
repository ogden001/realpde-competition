# RealPDE 默认训练后实验诊断协议

Status: `DEFAULT_REQUIRED / REVIEW_GATE`

本协议适用于 **所有正式 GPU 训练实验**，尤其是 Track 1 点预测、残差修正、Loss、Feature、Training / Continuation 等会形成可比较模型输出的实验。

核心原则：

> **训练完成不等于实验完成。训练之后必须默认产生可复核的诊断数据，并提交 Git；Codex / Luna 只负责执行与结构化证据交付，ChatGPT / Sol 负责回收证据、验收完整性和最终科研判断。**

短 smoke、单元测试、implementation parity test 可以显式豁免。只要一个结果会用于 `KEEP / GO / NO_GO / MERGE / LONG_TRAIN / SUBMIT` 等科研决策，就必须执行本协议。

---

## 1. 默认复用仓库已有分析代码

标准分析代码已经在 Git 中，后续任务默认由 Codex / Luna：

1. `git pull --rebase origin main`；
2. 复用现有分析模块；
3. 不重复编写另一套逐帧 / 逐 case / spatial 统计逻辑；
4. 只有现有代码确实无法覆盖新的实验机制时，才由 ChatGPT / Sol 决定是否新增诊断。

当前标准入口包括：

- `tools/post_train_diagnostics.py`
  - 通用 post-train diagnostic bundle；
- `tools/dw01_by_horizon.py`
  - Future20 / by-horizon 机制诊断；
- `tools/colleague_80pt/analyze_checkpoints.py`
  - 历史或已完成 checkpoint 的统一 replay / 对比；
- `tools/colleague_80pt/residual_multi.py`
  - residual 训练结束后已经默认接入标准诊断；
- `tools/build_training_review_log.py`
  - 训练过程 review log 的确定性压缩。

新 runner 应优先调用 `post_train_diagnostics.write_post_train_diagnostics()`，而不是重新复制指标实现。

---

## 2. 每个正式 GPU 训练实验的默认诊断

训练结束后，在冻结的 Dev / evaluation protocol 上至少产出以下几层证据。

### 2.1 Overall / 比赛主指标

至少保留：

- Rel-L2；
- TKE；
- MVPE；
- 若实验涉及 SPS，则包含 SPS / coverage / width；
- 若实验涉及 runtime，则包含 time / throughput；
- 相对 matched baseline 的 delta。

目的：

> 判断实验总体有没有收益，是第一层 gate，但不能单独承担科研结论。

---

### 2.2 Future1-Future20 逐帧 / By-Horizon

至少保留：

- `frame_rel_l2`；
- `u_rmse`；
- `v_rmse`；
- `velocity_rmse`；
- `tke_contrib_rel_l2`；
- `tke_contrib_ratio`；
- 与 baseline 的逐 horizon delta。

完整保留 Future1..20，不允许只保存 1-5 / 6-10 / 11-15 / 16-20 分段均值。

目的：

> 判断误差随预测 horizon 如何演化，识别 long-horizon drift、Future18/19/20 cliff、variance / energy collapse，以及某个优化到底改善早期、中期还是尾部。

逐帧 TKE 是机制分解量，不是官方 single-frame TKE score。具体定义见 `docs/EXPERIMENT_BY_HORIZON_PROTOCOL.md`。

---

### 2.3 By-Trajectory / 逐 Case

每条 Dev trajectory 至少保留：

- Rel-L2；
- TKE diagnostic / official aggregate where applicable；
- MVPE；
- u/v RMSE；
- 相对 baseline delta；
- mean-field / fluctuation / energy 相关统计。

同时保留 win count，例如：

- Rel 改善 trajectory 数 / 总数；
- TKE 改善 trajectory 数 / 总数。

目的：

> 判断平均收益是跨 case 稳定存在，还是被少数 trajectory 拉动；区分“普遍收益”和“局部偶然收益”。

---

### 2.4 Trajectory × Horizon

至少保留：

- trajectory；
- window_start；
- horizon；
- frame Rel-L2；
- velocity RMSE；
- 与实验核心假设相关的 TKE contribution / 其他逐帧诊断。

目的：

> 把“时间问题”和“样本问题”解耦。比如 Future20 凸起到底是所有 case 都存在，还是由少数异常 trajectory 造成。

---

### 2.5 Mean / Fluctuation / Energy 分解

对速度场至少分析：

`u(t,x,y) = mean(u)(x,y) + u'(t,x,y)`

保留：

- `mean_field_rel_l2`；
- `fluctuation_rel_l2`；
- `tke_field_rel_l2`；
- `pred_tke_mean`；
- `target_tke_mean`；
- `tke_energy_ratio`。

目的：

> 判断模型提升来自平均流场、动态波动，还是单纯 amplitude calibration；识别“Rel-L2 好但波动被抹平”“TKE 能量系统性偏低”等机制。

---

### 2.6 Spatial Error / 空间误差

至少保留或汇总：

- u RMSE map；
- v RMSE map；
- velocity RMSE map；
- mean-field RMSE map；
- fluctuation RMSE map；
- TKE abs-error map；
- target TKE mean map。

轻量结果应进入 Git，至少形成 `spatial_summary.csv`；小体积 `spatial_maps.npz` 可以直接提交。

目的：

> 判断错误是 global 还是 local，是否集中在 wake、剪切层、翼型附近或其他高能区域，为 region-aware loss / local corrector 等后续方向提供证据。

---

### 2.7 Training Progress / 训练过程

从已有 eval / checkpoint / log 中尽量整理：

- step / update；
- Rel-L2；
- TKE；
- MVPE；
- LR；
- 实验核心 loss term；
- best checkpoint / early-stop 信息。

不要求为每个中间 checkpoint 重新 replay 全套逐帧诊断。

目的：

> 区分“训练不够”“已经 plateau”“过拟合”“指标沿稳定 Pareto trade-off 移动”，避免把方向性冲突误判成训练时长不足。

---

### 2.8 实验特有诊断

默认七层之外，根据实验机制增加必要证据。

例如：

- Residual：
  - base vs corrected；
  - correction_help_fraction / hurt_fraction；
  - delta RMS；
  - delta mean / fluctuation RMS；
  - base / corrected mean-field、fluctuation、TKE energy；
- SPS / uncertainty：
  - coverage-width trade-off；
  - per-case coverage；
  - calibration curve；
  - sigma-error ranking；
- Loss：
  - loss term magnitude；
  - gradient contribution / conflict；
- Feature：
  - Train / Dev feature distribution；
  - OOD / tail coverage；
- Runtime：
  - numerical equivalence / metric regression。

原则：

> 只增加会改变研发判断的专项分析，不为每个小实验无限堆图。

---

## 3. 标准 Git 交付结构

正式训练结果应尽量形成：

```text
<experiment>/
├── final_primary_metrics.json
├── training_progress.csv
├── diagnostic_manifest.json
└── diagnostics/
    ├── summary.json
    ├── by_horizon.csv
    ├── by_trajectory.csv
    ├── by_trajectory_horizon.csv
    ├── spatial_summary.csv
    └── spatial_maps.npz          # 小体积时提交
```

Paired / multi-arm 实验再增加：

```text
comparison_summary.csv
comparison_by_horizon.csv
trajectory_comparison_summary.csv
```

大 checkpoint、H5、raw prediction cache、超大日志保留在 GPU / artifact 存储，不提交 Git；Git 必须记录其路径、SHA256、iteration 和 provenance。

`diagnostic_manifest.json` 至少记录：

- experiment ID；
- execution commit；
- diagnostic code commit；
- checkpoint path / SHA256 / iteration；
- split manifest / SHA256；
- scorer / protocol；
- windows / trajectories；
- terminal status。

---

## 4. Codex / Luna 的职责边界

Codex / Luna 是 **Evidence Producer + Runner**，不是实验结果最终 reviewer。

训练结束后，Codex 默认负责：

1. 拉取当前 Git 最新诊断代码；
2. 执行标准 post-train diagnostics；
3. 把原始轻量 CSV / JSON / Markdown / 必要 figures 结构化落盘；
4. 做基本 shape / row-count / finite / parity 检查；
5. commit + push 到 `origin/main`；
6. 返回 Git commit、目录、文件清单和缺失项。

Codex 可以报告事实，例如：

- 100 horizon rows；
- 16 trajectory rows；
- 某 checkpoint SHA；
- 某 metric 数值。

但 Codex 的：

- `PROMISING`
- `NO_GO`
- `KEEP`
- “已经分析完成”
- “下一步应该……”

均不直接成为项目科研结论。

**Codex 不负责替 ChatGPT / Sol 决定实验是否值得继续。**

---

## 5. ChatGPT / Sol 的 Review Gate

这是硬规则。

当 Codex 返回 GPU 实验结果后，ChatGPT / Sol **第一步不是分析指标，而是验收证据完整性**。

### 5.1 必须先从 GitHub 检查

至少确认：

- 结果 commit 已进入远端 `main`；
- required CSV / JSON / manifest 存在；
- row count / horizon / trajectory 覆盖符合协议；
- 关键数字有可复核的原始数据源；
- checkpoint / split / scorer / code provenance 足够；
- 没有只留本地路径或聊天摘要。

### 5.2 缺失时的行为

任一关键证据缺失：

`REVIEW_BLOCKED`

此时 ChatGPT / Sol 必须：

1. 停止科研 review；
2. 不依据 Codex 的文字摘要做机制解释；
3. 不脑补缺失的 Future20 / trajectory / spatial 数据；
4. 明确列出缺失项；
5. 给 Codex 一个只补证据的 bounded task；
6. 等 Git 证据完整后再开始 review。

### 5.3 完整后才进入 Review

只有 evidence checklist 通过后，ChatGPT / Sol 才：

- 读取 overall；
- 看 by-horizon；
- 看 by-trajectory；
- 看 mean / fluctuation / energy；
- 看 spatial；
- 看 training progress；
- 结合实验特有机制数据；
- 给出 `GO / NO_GO / LONG_TRAIN / STOP / NEXT` 等科研判断。

核心规则：

> **Codex 的“任务完成”不等于“可 Review”；Git 中的可复核证据完整，才等于“可 Review”。**

---

## 6. Review 的分析初衷

默认诊断不是为了做报告，而是回答五个问题：

1. **总体**：分数到底变没变？
2. **时间**：哪些 Future step 变了？是否有 long-horizon cliff？
3. **样本**：哪些 trajectory 变了？收益是否稳定？
4. **物理**：变的是 mean、fluctuation、energy，还是某个空间区域？
5. **训练**：这是没训够，还是方向本身存在 trade-off？

只有这些问题回答完整，才进入下一轮实验设计。

---

## 7. 最终原则

**任意正式 GPU 训练实验，默认执行必要诊断；不把 post-train analysis 当成可选项。**

**已有诊断代码默认从 Git 拉取和复用，除非新的实验机制确实需要新增分析。**

**Codex / Luna 负责跑、整理、提交结构化证据，不负责最终科研判断。**

**ChatGPT / Sol 先做 Evidence Acceptance，再做 Scientific Review。**

**证据缺失 = REVIEW_BLOCKED，不开始 review。**
