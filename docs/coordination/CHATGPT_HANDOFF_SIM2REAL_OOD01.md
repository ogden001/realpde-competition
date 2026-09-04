# Round 2 OOD-01 — CFD-only condition coverage

## Verdict

`CFD_OOD_COVERAGE_MODERATE`

18 个 CFD-only condition：`15225_{0,5,10,15,20}`、`17775_0`、`22875_{5,20}`、`24150_5`、`25425_5`、`26700_{5,20}`、`27975_{0,5,10,15,20}`、`3750_15`。

按相同 AoA 的 PIV Re 范围与最近 Re 判断（不假设 trajectory pairing）：interpolation `8/18`，edge `7/18`，extrapolation `3/18`。主要沿已有 Re 网格补齐少量缺口和高 Re tail，并非大面积新 regime。

沿用 phase-free、u/v-only、32×64 descriptor；不使用 pressure，不作逐帧配对。raw amplitude（mean/RMS/TKE）会把 CFD-only 与 PIV 空间明显拉开，但这主要反映 CFD/PIV calibration gap。只看 normalized temporal/shape descriptor（ACF lag 1/5/10/20 + dominant frequency），18 条 CFD-only 到 PIV descriptor space 的 nearest-neighbor distance median 为 `0.168`（IQR-scaled units，range `0.113–0.593`）。

因此 CFD-only 有限补充边缘和少量外推点，值得保留为 OOD/analysis asset，但不足以单独支持 temporal Sim2Real 或长 CFD pretraining。
