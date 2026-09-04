# RealPDE Track 1 — MF Energy-Aware Campaign 02 Handoff

## Status

`REVIEW_REQUIRED`. All preregistered arms completed at +500/+1000/+1500:
C0, E4, E5, E6, E7 and E8. Locked-final and Codabench were not accessed.

## Provenance

- Required base and synchronized `main`: `47c4e5bdada669dd02877d47bdb27e04ddbf0dc3` (ancestor check passed).
- Campaign IDs: `T1-ID-MF-C02-CONT-S20260901`, `...-TKEREL-...`, `...-RMSREL-...`, `...-HIFLUC-...`, `...-CONDGAIN-FROZEN-...`, `...-SPATIALGAIN-FROZEN-...`.
- Split: CLEAN 50 train / 16 Dev; locked-final `NO`; manifest SHA `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`.
- Seed `20260901`, P0-A, AdamW `1e-5`, batch 8, workers 2, 1500 additional updates, official v9 scorer SHA `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`.
- All arms initialized from MF-01@1500 checkpoint SHA `488a8118f489789d385ec90e02856ef6a8482d6fa75c252e2e5d2d1f50e72226`.
- Remote run root: `/home/chyfuture/realpde_runs/mf_energy_campaign02/`; local small-artifact mirror: `artifacts/mf_energy_campaign02_20260904/`.

## Primary result (+1500, lower is better)

| Arm | Rel-L2 | Δ vs MF-01 | TKE | Δ vs MF-01 | MVPE | Δ vs MF-01 | Δ vs C0 TKE | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Original Control@1500 | 0.193675 | +2.79% | 0.633786 | -1.76% | 0.165178 | +2.88% | — | reference |
| Original MF-01@1500 | 0.188409 | — | 0.645156 | — | 0.160552 | — | — | reference |
| C0 continuation | 0.164327 | -12.78% | 0.582928 | -9.64% | 0.130374 | -18.80% | — | matched baseline |
| E4 scorer-aligned TKE | 0.165204 | -12.32% | 0.578876 | -10.27% | 0.131071 | -18.36% | -0.004052 (-0.70%) | `NO_GO` |
| E5 RMS/amplitude | 0.165113 | -12.39% | 0.580655 | -10.00% | 0.133421 | -16.90% | -0.002273 (-0.39%) | `NO_GO` |
| E6 high-energy weighted | 0.153459 | -18.57% | 0.580184 | -10.07% | 0.144576 | -9.95% | -0.002744 (-0.47%) | `NO_GO` |
| E7 frozen conditional gain | 0.191385 | +1.58% | 0.655003 | +1.53% | 0.163086 | +1.58% | +0.072075 (+12.36%) | `NO_GO` |
| E8 frozen spatial gain | 0.191383 | +1.58% | 0.654982 | +1.52% | 0.163086 | +1.58% | +0.072054 (+12.36%) | `NO_GO` |

Trajectory wins are against C0 / original MF-01 respectively:

| Arm | Rel wins | TKE wins | MVPE wins | Rel wins vs MF | TKE wins vs MF | MVPE wins vs MF |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 16/16 | 10/16 | 16/16 | 16/16 | 10/16 | 16/16 |
| E4 | 4/16 | 3/16 | 7/16 | 16/16 | 9/16 | 16/16 |
| E5 | 4/16 | 15/16 | 4/16 | 16/16 | 15/16 | 16/16 |
| E6 | 12/16 | 6/16 | 1/16 | 16/16 | 4/16 | 12/16 |
| E7 | 0/16 | 7/16 | 1/16 | 2/16 | 12/16 | 4/16 |
| E8 | 0/16 | 7/16 | 1/16 | 2/16 | 12/16 | 4/16 |

## Analysis conclusions

- **E4:** scorer-aligned normalized TKE-map loss improves aggregate TKE only 0.70% versus C0, with 3/16 wins, and does not meet the supportive gate. It is not a stable answer to the scorer-alignment hypothesis.
- **E5:** RMS/amplitude loss is the most stable TKE mechanism in this screen: 15/16 TKE wins versus C0 and a 0.39% aggregate TKE improvement, but it costs 2.34% MVPE versus C0 and fails the protected multi-metric gate. It is `REVIEW_REQUIRED`, not KEEP.
- **E6:** high-energy weighting does not repair the high-TKE pattern. It improves Rel but worsens MVPE by 10.89% versus C0; target top-20% TKE-region error worsened for `26700_0` (-7.24e-5), `24150_10` (-1.57e-4), and `20325_20` (-6.77e-6). `NO_GO`.
- **E7:** alpha remained effectively identity: min 0.997376, p25 0.997424, median 0.997442, p75 0.997451, max 0.997461, std 2.15e-5. `GAIN_REMAINS_EFFECTIVELY_IDENTITY`.
- **E8:** alpha also remained effectively identity: min 0.997082, p25 0.997433, median 0.997456, p75 0.997462, max 0.997468, std 4.22e-5. `GAIN_REMAINS_EFFECTIVELY_IDENTITY`.
- The three existing MF bad cases remain mixed rather than fixed: `26700_0` and `24150_10` are high-energy regressions for E6, while `20325_20` is also worse in the top-energy region. Good cases do not show one universal mechanism.

## Mechanism conclusion

The strongest bounded conclusion is **optimization/objective competition**, not a deployable gain mechanism: E5 can protect TKE broadly but trades away MVPE, while E4 is weaker. Explicit target-derived high-energy weighting is insufficient and harms spatial/mean-flow behavior. Frozen runtime-safe conditional/spatial inputs do not produce a usable correction signal under this architecture; both gains remain identity. The remaining bottleneck is therefore consistent with deeper fluctuation dynamics / spatial energy representation, with runtime-safe information insufficiency not ruled out but unsupported by E7/E8.

Recommended mechanisms: **KEEP** original N2/MF-01 continuation as the control; **DROP** E4, E6, E7 and E8; keep E5 only as a review candidate, not an automatic next model. No spectral, transformer, independent fluctuation backbone or MF-02 was started.

## Artifact manifest

Each arm contains `summary.json`, `run_metadata.json`, `eval_00000/`, `eval_00500/`, `eval_01000/`, `eval_01500/`, and update checkpoints in the remote run root. The local mirror contains all final trajectory/horizon CSVs, six representative TKE map figures per arm, and final summaries under `artifacts/mf_energy_campaign02_20260904/analysis2_{e4,e5,e6,e7,e8}/`. No checkpoint, prediction NPZ, dataset or submission archive is committed.

`NEXT_ACTION = REVIEW_REQUIRED`
