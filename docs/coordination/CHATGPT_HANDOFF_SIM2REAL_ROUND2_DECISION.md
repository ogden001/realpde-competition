# RealPDE Track 1 Sim2Real Round 2 Decision

## Representation

`STOP`。REP-01 中 official sim_pretrain CNO + tiny linear probe 在 Rel-L2、TKE、MVPE 均劣于 matched random frozen CNO。

## CFD OOD coverage

`KEEP`。18 个 CFD-only condition 有限补充 Re/AoA space 的 edge/extrapolation，normalized temporal descriptor 仍接近 PIV space；定位为 OOD/analysis asset，不是 forecast transfer 证据。

## Overall

`WEAK_SIGNAL / PARKED`

当前 CFD / Sim2Real 主线停止继续投入，回到其它一级优化方向。

停止：CFD representation→PIV forecast、temporal raw transfer、mixed curriculum、long CFD pretraining、pseudo-PIV sweep、teacher/student、adversarial alignment。 不进入 Round 3，不做 submission/private-test 验证。
