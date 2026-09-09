# Structured Temporal Dynamics R2 Handoff

Status: `REVIEW_REQUIRED`

## 1. Execution status

- R1 CPU summary consumed `/home/chyfuture/realpde_runs/structured_temporal_dynamics_round1_20260907_lowmem/`; no GPU inference was rerun.
- R2 C0, V1, A1 and A2 all completed continuation from Direct@1500 and evaluated absolute updates 2000/2500/3000.
- Fixed protocol: 50 Train / 16 Dev, P0-A, N2, stride 20, seed `20260901`, AdamW `1e-5`, physical batch 8, no gradient accumulation.
- Locked-final, full-data, SPS and Codabench were not accessed.
- C0 aggregate/runtime were recovered from completed eval CSVs after a post-training metadata write bug; the three eval directories and checkpoints are intact. C0 runtime fields not recoverable from that run are recorded as null.

## 2. C0/V1/A1/A2 aggregate table

Lower raw error is better. Values are official-v9 raw metric replay/diagnostic aggregates.

| Arm | Update | Rel-L2 | TKE | MVPE |
|---|---:|---:|---:|---:|
| C0 | 2000 | 0.180337 | 0.592580 | 0.159620 |
| C0 | 2500 | 0.161814 | 0.572662 | 0.132887 |
| C0 | 3000 | 0.161343 | 0.557971 | 0.131496 |
| V1 | 2000 | 0.175822 | 0.593703 | 0.150516 |
| V1 | 2500 | 0.158527 | 0.570234 | 0.126417 |
| V1 | 3000 | 0.156023 | 0.555572 | 0.121585 |
| A1 | 2000 | 0.180842 | 0.593417 | 0.162271 |
| A1 | 2500 | 0.161198 | 0.576478 | 0.132822 |
| A1 | 3000 | 0.159602 | 0.556817 | 0.127943 |
| A2 | 2000 | 0.180870 | 0.592161 | 0.161177 |
| A2 | 2500 | 0.161446 | 0.574025 | 0.132812 |
| A2 | 3000 | 0.161400 | 0.559184 | 0.131574 |

## 3. R1 horizon-band summary

Percentages are relative to C0; blank TKE means the R1 single-frame horizon diagnostic had zero TKE denominator.

| Arm | Update | Early Rel/TKE/MVPE % | Mid Rel/TKE/MVPE % | Late Rel/TKE/MVPE % | Far Rel/TKE/MVPE % |
|---|---:|---|---|---|---|
| T1 | 2000 | 4.351/–/2.253 | 1.229/–/-0.837 | 0.556/–/-1.451 | 3.712/–/3.307 |
| T1 | 2500 | 4.843/–/4.706 | 0.288/–/-3.297 | 0.530/–/-1.974 | 3.330/–/3.149 |
| T1 | 3000 | 11.220/–/4.146 | 4.833/–/-2.244 | 2.306/–/-3.535 | 4.424/–/0.482 |
| T2 | 2000 | 2.442/–/0.544 | 3.576/–/2.440 | 2.888/–/0.429 | 0.209/–/-1.126 |
| T2 | 2500 | 2.560/–/1.587 | 2.839/–/0.527 | 2.899/–/0.004 | 2.955/–/3.412 |
| T2 | 3000 | -0.336/–/1.149 | 0.837/–/1.592 | -0.497/–/0.648 | -2.695/–/-0.118 |
| T3 | 2000 | -0.577/–/-0.322 | -0.354/–/0.023 | -0.401/–/-0.120 | -0.878/–/-1.276 |
| T3 | 2500 | -1.441/–/-1.299 | -0.331/–/0.129 | -0.460/–/-1.284 | 0.082/–/0.669 |
| T3 | 3000 | -0.151/–/1.737 | 0.196/–/1.791 | -0.047/–/2.123 | -0.751/–/-0.227 |

## 4. R2 horizon-band and trajectory summary

| Arm | Update | Early Rel/TKE/MVPE % | Mid Rel/TKE/MVPE % | Late Rel/TKE/MVPE % | Far Rel/TKE/MVPE % |
|---|---:|---|---|---|---|
| V1 | 2000 | 3.874/–/4.513 | 3.278/–/3.738 | 2.740/–/2.458 | 1.182/–/0.053 |
| V1 | 2500 | 3.050/–/3.604 | 3.038/–/2.764 | 3.557/–/3.794 | 0.391/–/-1.249 |
| V1 | 3000 | 3.441/–/3.515 | 4.336/–/5.434 | 4.160/–/5.559 | 2.337/–/2.596 |
| A1 | 2000 | -0.448/–/-1.330 | -0.626/–/-1.671 | -0.640/–/-1.445 | 0.223/–/-0.137 |
| A1 | 2500 | 0.367/–/-0.105 | 0.173/–/-0.408 | 0.178/–/-0.016 | 0.501/–/0.880 |
| A1 | 3000 | 0.671/–/1.390 | 0.669/–/1.168 | 0.476/–/1.043 | 1.626/–/2.447 |
| A2 | 2000 | -0.267/–/-0.522 | -0.268/–/-0.433 | -0.274/–/-0.471 | -0.366/–/-0.513 |
| A2 | 2500 | 0.145/–/-0.171 | 0.095/–/-0.167 | 0.077/–/0.004 | 0.346/–/0.834 |
| A2 | 3000 | 0.102/–/0.078 | -0.032/–/-0.050 | -0.113/–/-0.280 | -0.095/–/-0.102 |

| Arm | Update | Rel wins/16 | TKE wins/16 | MVPE wins/16 | All-three wins | Rel+MVPE, TKE degradation ≤2% |
|---|---:|---:|---:|---:|---:|---:|
| V1 | 2000 | 15 | 15 | 15 | 14 | 15 |
| V1 | 2500 | 15 | 11 | 14 | 10 | 12 |
| V1 | 3000 | 16 | 1 | 16 | 1 | 7 |
| A1 | 2000 | 7 | 3 | 0 | 0 | 0 |
| A1 | 2500 | 13 | 4 | 10 | 1 | 4 |
| A1 | 3000 | 16 | 1 | 16 | 1 | 6 |
| A2 | 2000 | 1 | 3 | 1 | 0 | 1 |
| A2 | 2500 | 13 | 2 | 7 | 0 | 5 |
| A2 | 3000 | 5 | 15 | 7 | 4 | 4 |

## 5. Architecture / optimizer preflight

- All arms: initial prediction max absolute difference from C0 `0.0`; pressure max absolute value `0.0`; finite outputs; latent input to original `project` has 32 channels and time length 20; project preserves time length.
- Backbone optimizer state: 132 state entries after restoring Direct@1500; A1/A2 temporal parameters were added afterward in a fresh param group.
- A1: temporal Conv3d kernels `(3,1,1)`, hidden width 32, 6208 added parameters, initial adapter gradient max `0.0358178616`.
- A2: one temporal-only MHA layer, 4 heads over 32 channels, zero-init output projection, 5280 added parameters, initial adapter gradient max `0.0358178653`.
- V1: fixed deterministic calibration `lambda_vort=15.5385751724`; no other auxiliary loss.

## 6. Runtime / params

| Arm | Parameters | Added | Train wall s | Inference s/window | Peak allocated GiB | Adapter output norm | Adapter parameter norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 7,989,843* | 0 | n/a* | n/a* | n/a* | 0 | 0 |
| V1 | 7,989,843 | 0 | 1298.94 | 0.024240 | 18.31 | 0 | 0 |
| A1 | 7,996,051 | 6,208 | 1539.50 | 0.034695 | 18.43 | 0.629677 | 3.653208 |
| A2 | 7,995,123 | 5,280 | 1337.27 | 0.025280 | 19.41 | 0.644312 | 10.162996 |

`*` C0 parameter count is the same direct backbone as V1; C0 runtime fields were not emitted before the post-training write error and are not reconstructed as timing claims.

## 7. Artifact paths

- Remote root: `/home/chyfuture/realpde_runs/structured_temporal_dynamics_round2_20260908/`
- R1 summary: `r1_summary/r1_horizon_delta.csv`, `r1_summary/r1_trajectory_summary.csv`, `r1_summary/r1_horizon_band_summary.csv`
- R2 summary: `summary/r2_horizon_delta.csv`, `summary/r2_trajectory_summary.csv`, `summary/r2_horizon_band_summary.csv`
- Per-arm evidence: `runs/{C0,V1,A1,A2}/`
- Final A1/A2/C0/V1 `eval_03000/predictions.npz` and all requested metric CSVs are under their respective arm directories.

## 8. Tests

- `PYTHONPATH=code/tools python3 -m pytest code/tests/test_structured_temporal_dynamics.py -q -k 'latent_temporal or r2_rejects'` → 2 passed.
- `python3 -m py_compile code/tools/realpde_structured_temporal_dynamics_r2.py code/tools/realpde_structured_temporal_summary.py` → passed.
- Remote container py-compile of R2 runner and summary → passed.

## 9. Commit SHA

`9deb41edd07259cc3c290e1ce640517a7b437088` (implementation commit; handoff update follows in the next commit)
