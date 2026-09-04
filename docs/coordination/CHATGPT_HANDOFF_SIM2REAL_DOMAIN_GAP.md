# RealPDE Track 1 — Phase-Free CFD / PIV Domain-Gap Audit

状态：`COMPLETED`。只分析 82 个共同 Re/AoA condition；不假设 trajectory pairing 或 frame alignment。

## Protocol

- 输入：正式 `train_sim.tar.gz` / `train_real.tar.gz`；按 condition key 分组，不把同 key 当成唯一 physical realization。
- 统一：`u/v only`、raw `64×128 → 32×64`、无 pressure、无 locked-final Future20。
- descriptor：完整 trajectory 的 mean-flow norm、fluctuation RMS/TKE proxy、8-bin radial spatial spectrum、temporal autocorrelation（lag 1/5/10/20）和 FFT dominant frequency。
- reference：由于没有 exact repeated PIV condition，使用同 AoA、最近 Re 的 PIV↔PIV 邻近-condition variation；它是保守参考，不是 exact natural replicate。

## 结果（82 conditions 的中位数）

| descriptor | CFD↔PIV gap | PIV↔PIV neighbor variation | gap / reference |
|---|---:|---:|---:|
| mean-flow norm relative gap | 4.716 | 0.087 | 54.0× |
| TKE-norm relative gap | 189.841 | 0.173 | 1,095× |
| spatial spectrum L1 | 0.109 | 0.036 | 3.0× |
| ACF lag-1 absolute gap | 0.0054 | 0.0023 | 2.4× |
| ACF lag-5 absolute gap | 0.0498 | 0.0225 | 2.2× |
| ACF lag-10 absolute gap | 0.0922 | 0.0644 | 1.4× |
| ACF lag-20 absolute gap | 0.1916 | 0.1340 | 1.4× |
| dominant-frequency absolute gap | 0.1843 | 0.0576 | 3.2× |

“mean-flow norm gap”与“TKE-norm gap”是 phase-free scalar scale diagnostics，不是官方 scorer 的 Rel-L2/TKE 分数。它们首先暴露 CFD/PIV raw amplitude/normalization 不一致；不能据此直接声称 CFD 物理场错误。但即使只比较 normalized spectrum/temporal shape，CFD↔PIV gap 仍高于邻近 PIV variation，且 temporal gap 没有表现出可直接迁移的稳定关系。

## 四类判断

1. **Mean flow — 可描述，但 raw transfer 不成立。** CFD 与 PIV 的 condition coverage/large-scale structure 有价值，但 raw norm 差异约 4.7 倍；必须先解决单位、尺度、观测算子和 geometry/mask 口径。
2. **Fluctuation RMS / TKE proxy — 高风险。** CFD TKE proxy 的跨域能量差远超 PIV neighbor reference；PIV 测量噪声、滤波和 mask 会强烈影响高频能量。未经校准的 CFD TKE auxiliary loss 存在负迁移风险。
3. **Spatial spectrum — 只有弱表征信号。** spectrum L1 gap 约为邻近 PIV variation 的 3 倍；可以支持 frozen representation/OOD 探索，但不足以支持直接 spectrum-matching 或新模型结构。
4. **Temporal autocorrelation / PSD — 不支持 raw temporal transfer。** dominant frequency gap 约为邻近 PIV variation 的 3.2 倍，长 lag ACF 也更差；没有稳定 global phase lag 的证据。逐帧 residual、phase-aligned CFD teacher 和直接 raw rollout transfer 不应继续。

## 结论

主结论：`REPRESENTATION_ONLY_MORE_PLAUSIBLE`。

子结论：`CFD_TEMPORAL_TRANSFER_NOT_SUPPORTED`（针对未经观测退化、尺度校准和 domain-specific calibration 的 raw temporal transfer）。CFD 仍可能在 representation/OOD 或经过明确 calibrated pseudo-PIV 的窄路线中有价值，但本轮没有证据支持把 CFD 当作可直接迁移的 PIV dynamics teacher。

### CFD 对 PIV TKE 的判断

更可能是**存在负迁移风险**，不是“无明显价值”。原因是 TKE energy gap 极大、空间 spectrum gap 也高于 PIV 邻近 variation，而且 PIV 的噪声/滤波会改变 TKE。下一轮若保留 TKE 方向，必须把 calibrated pseudo-PIV 和 PIV-only control 放在同一 clean train/dev protocol 下；本报告不授权训练。

## 证据路径

- `artifacts/sim2real_round1/summary.json`
- `artifacts/sim2real_round1/pairs.csv`
- `artifacts/sim2real_round1/piv_neighbor_variation.csv`
- `artifacts/sim2real_round1/inventory.csv`

没有使用 pressure、locked-final、private-test、Codabench 或逐帧 CFD/PIV residual。
