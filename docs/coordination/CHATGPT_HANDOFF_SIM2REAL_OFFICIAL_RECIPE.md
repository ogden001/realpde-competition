# RealPDE Track 1 — Official Sim2Real Recipe / Provenance Audit

状态：`COMPLETED`。结论：`INSUFFICIENT_PROVENANCE`。

## 来源层级

1. 官方 Track 1 Starting Kit v9：`tmp/realpde_t1_starting_kit_v9 (1).zip`，SHA-256 `c14b24d0e385d6761be5a29d721a05d2152ccda317ab3c700caad6993febf61c`。它包含 inference loader、CNO/FNO/Transolver vendored model、scorer 和 submission template，没有训练脚本、训练日志或 sim_pretrain 配置。
2. 本地 checkpoint：`data/baseline_checkpoints/`。checkpoint 记录 state、train/val metric history、iteration/best_iteration，但没有保存完整 CLI/config provenance。
3. `third_party/RealPDEBench/`：只能参考。其 Foil dataset/config 代码提供“代码支持的 recipe”，不能证明 competition checkpoint 实际使用了所有默认值。

## 能确认的内容

| 项目 | 结论 |
|---|---|
| architecture | checkpoint state 及 starting kit loader 确认：CNO 为 `CNO3d, N_layers=3`；FNO 为 `(modes1,modes2,modes3)=(4,12,16), width=64, 4 layers`；Transolver 为 3 blocks、hidden 256、8 heads、16 slices。 |
| task | Starting Kit 明确是 `20→20`，输入/输出 competition shape 为 `(20,32,64,3)`；PIV pressure 为 zero placeholder。 |
| checkpoint budget | sim_pretrain 与 sim_real_ft 的可用训练 checkpoint 都记录 `iteration=5000`；sim CNO `best_iteration=2700`，real-ft CNO `best_iteration=5000`。 |
| checkpoint provenance | `sim_pretrain/sim_cno.pth` SHA `af85374bfd06c0e386ec803d777396c21484978392213025697c5a7470106b6b`；`sim_real_ft/sim_real_cno.pth` SHA `82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61`。sim_real_ft 明确是官方 real-PIV-finetuned 目录，但训练 trajectory 覆盖与本地 clean manifest 的关系 UNKNOWN。 |

## 代码支持 ≠ competition 实际确认

`third_party/RealPDEBench/realpdebench/data/fluid_dataset.py` 的 Foil 默认值**只代表代码支持/默认配置**：`in_step=20`、`out_step=20`、`interval=20`、`sub_s_real=2`、`sub_s_numerical=2`、Gaussian normalizer、`mask_prob=0.1`、`noise_scale=0.1`；Foil CNO yaml 为 3 layers、`num_update=5000`、batch 16、AdamW-like `lr=3e-4`、cosine scheduler。dataset code 还支持 Gaussian/Poisson/optical noise 以及 numerical pressure masking。

但是以下项目对 `sim_pretrain` competition checkpoint 实际使用值均为 `UNKNOWN`：

- exact sampling/trajectory split/window realization；
- exact loss implementation and weighting；
- whether `noise_scale=0.1` 实际启用、noise type 及其作用于 input/output 的方式；
- whether `mask_prob=0.1` 实际启用、pressure mask 的随机 seed；
- exact normalizer fit set、统计量和 train/real-ft domain-specific handling；
- optimizer/scheduler/batch/worker 实际 CLI；
- whether the checkpoint was trained from the shown third-party Foil yaml or another private/competition script。

Starting Kit README 只确认提交接口、model shapes、scorer、real `p=0` handling 和 default SPS band；它不补齐上述训练 provenance。

## Recipe audit verdict

- architecture / 20→20 / 5000 iterations：`CONFIRMED`。
- sampling / loss / noise / mask / normalization / exact budget semantics：`PARTIAL or UNKNOWN`。
- sim_real_ft 的 real fine-tuning：目录命名和项目文档确认其用途，但覆盖 trajectory 与 clean split 污染关系 `UNKNOWN`；因此只作 competition-oriented warm start，不作为 clean causal evidence。

最终结论：`INSUFFICIENT_PROVENANCE`。不能把“代码支持的 augmentation”写成“official sim_pretrain 已实际完成的 Sim2Real”。在 provenance 补齐前，不重复开发复杂 baseline，也不据此声称 official pretrain 已经 strong。
