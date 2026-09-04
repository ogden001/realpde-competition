# Round 2 REP-01 — CFD representation probe

## Verdict

`CFD_REPRESENTATION_NOT_SUPPORTED`

本实验只使用 manifest 的 train/dev（50/16 trajectories），没有读取 locked-final，未访问 Codabench。

## Protocol

- 同一官方 CNO3d：3 layers，输入/输出 3 channels，32×64；20→20，stride=20，u/v only，pressure channel 置零。
- A1：官方 `sim_pretrain` checkpoint，backbone frozen。
- A0：同 architecture、同 seed protocol 的 random initialization，backbone frozen。
- 两组使用相同 2,440-parameter per-pixel linear probe、MSE、batch=16、100 updates；未训练 backbone。
- PIV dev：659 windows；目标为 full Future20 u/v。

## PIV dev result

| arm | Rel-L2 ↓ | TKE rel-L2 ↓ | MVPE ↓ |
|---|---:|---:|---:|
| A1 official CFD representation | 0.9026 | 32.0180 | 1.1509 |
| A0 matched random representation | **0.7680** | **13.0809** | **0.7800** |

A1 没有优于 A0，且三个指标均明显更差。

## Frozen representation gap

对 train/dev 的 66 条 PIV trajectory，各取前 20 帧构造 CFD/PIV pair；同一官方 frozen CNO 输出后，对时空维做 mean/variance pooling：pooled-mean relative L2 median `0.1213`，cosine distance median `0.00048`，pooled-variance relative L2 median `0.2044`。这说明 pooled mean 的方向相似，但不足以证明 representation 具有 PIV forecast 增量；variance 仍有约 20% 相对差异。

Round 1 raw-field phase-free gap 参考：mean relative gap median `4.7159`（PIV↔PIV same-AoA nearest-Re 为 `0.0874`）。latent pooled mean 比 raw field 更紧，但与 A1/A0 probe 结果合并后，证据不足以支持 official CFD representation 更适合 PIV。

## Reproducibility

- experiment id：`T1-ID-SIM2REAL-REP01-S20260904`
- manifest SHA256：`42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`
- sim_pretrain checkpoint SHA256：`af85374bfd06c0e386ec803d777396c21484978392213025697c5a7470106b6b`
- runners：`code/tools/realpde_sim2real_rep01.py`、`code/tools/realpde_sim2real_rep01_latent_gap.py`
- device：remote RTX 3090；A1 184.6 s，A0 163.7 s。

结论：停止把 official CFD representation 作为当前 PIV forecast 的主增量来源。
