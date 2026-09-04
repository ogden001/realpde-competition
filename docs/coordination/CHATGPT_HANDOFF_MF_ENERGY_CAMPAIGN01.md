# RealPDE Track 1 — MF Energy Campaign 01 Handoff

## Status

`REVIEW_REQUIRED`. E1, E2 and E3 each completed the locked 1500-update screen
with evaluations/checkpoints at 500/1000/1500. No MF-02 or other training was
started after completion.

## Provenance and invariants

- Execution commit: `f4dc26d` (implementation commit `b151cb1` plus device fix).
- Experiments: `T1-ID-MF-C01-TKE2X-S20260901`,
  `T1-ID-MF-C01-CONDGAIN-S20260901`, `T1-ID-MF-C01-SPATIALGAIN-S20260901`.
- Shared initialization: `sim_pretrain/sim_cno.pth`, SHA-256
  `af85374bfd06c0e386ec803d777396c21484978392213025697c5a7470106b6b`.
- Manifest SHA-256:
  `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`.
- Official scorer SHA-256:
  `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`.
- Fixed split: 50 train / 16 dev / 16 locked-final; P0-A; seed `20260901`;
  AdamW `1e-5`; batch 8; same windows and effective batch as MF-01.
- Control and all campaign arms start from the same CLEAN `sim_pretrain`
  checkpoint. No current full@43260 SOTA checkpoint was used.
- Unique variables: E1 only changes N2 TKE weight `0.05 -> 0.10`; E2 only
  adds a zero-initialized 5-scalar conditional gain; E3 only adds a
  zero-initialized spatial `1x1 Conv2d` gain. No new loss, feature family,
  branch capacity, LR, optimizer, batch, or unrelated structure change.
- Initialization smoke for E2/E3: zero gain gave `alpha=1`, prediction and
  pressure max error `0`, fluctuation temporal-mean max error
  `3.1590463e-7`.
- `locked-final accessed: NO`; `Codabench accessed: NO`.

## Matched 1500-update dev result

Lower error is better. Deltas are relative to MF-01.

| Arm | Rel-L2 | ΔRel | TKE | ΔTKE | MVPE | ΔMVPE | Rel wins | TKE wins | MVPE wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Control | 0.193675 | — | 0.633786 | — | 0.165178 | — | — | — | — |
| MF-01 | 0.188409 | — | 0.645156 | — | 0.160552 | — | — | — | — |
| E1 TKE2X | 0.190596 | +1.161% | 0.641192 | -0.614% | 0.158759 | -1.117% | 2/16 | 5/16 | 8/16 |
| E2 CondGain | 0.183465 | -2.624% | 0.634980 | -1.577% | 0.155231 | -3.314% | 16/16 | 4/16 | 14/16 |
| E3 SpatialGain | 0.183428 | -2.644% | 0.636492 | -1.343% | 0.154953 | -3.488% | 16/16 | 3/16 | 14/16 |

The official v9 subscores for E1/E2/E3 respectively were
`91.2994/75.7234/92.6458`, `91.5975/75.9019/92.7975`, and
`91.5991/75.8584/92.8094` for Rel/TKE/MVPE. Remote run metadata records all
three milestone results and exit code 0 under
`/home/chyfuture/realpde_runs/mf_energy_campaign01/`.

## Stage 0 offline oracle

This was replay-only on existing MF-01@1500 dev predictions; no target oracle
is available at test time and no oracle factor was deployed. Raw MF-01 was
Rel-L2/TKE `0.188409/0.645156`. Global, per-trajectory, per-window and fixed
4x8 spatial target-oracle factors gave respectively:

| Oracle | Rel-L2 | TKE | alpha range |
|---|---:|---:|---:|
| Global | 0.185545 | 0.619890 | 0.932 |
| Per-trajectory | 0.186437 | 0.598181 | 0.747–1.085 |
| Per-window | 0.186483 | 0.574088 | 0.466–1.438 |
| Fixed 4x8 spatial | 0.185613 | 0.481970 | 0–7.416 |

The large unconstrained spatial factor is an overfitting/capacity warning, not
evidence for a deployable method. Summary:
`/home/chyfuture/realpde_runs/mf_energy_campaign01/stage0/oracle_summary.json`.

## Mean / fluctuation / TKE diagnostics

The campaign replay confirms the MF-01 decomposition result: the improvement
is not Mean-only. MF-01 Mean Rel-L2 was `0.144862 -> 0.141107` (`-2.593%`,
10/16 trajectory wins), while Fluctuation Rel-L2 was `1.519750 -> 1.479491`
(`-2.649%`, 16/16 wins). Official TKE was `0.633786 -> 0.645156`
(`+1.794%` error, 7/16 wins).

For the campaign arms, high-energy-region absolute velocity error versus
MF-01 was `0.031881` (E1 `0.032579`, E2 `0.031491`, E3 `0.031458`); low-energy
error was `0.008729` (E1 `0.009509`, E2 `0.008347`, E3 `0.008422`). E2/E3
improved both regions; E1 worsened both. Final learned gain alpha statistics
over 659 dev windows were E2 `0.997955–0.998003` (median `0.997992`) and E3
`0.997948–0.998007` (median `0.997998`), so the added gain was effectively
near-identity.

Therefore: fluctuation-field reconstruction improved under the diagnostic,
but this did not produce stable TKE improvement. The explicit phenomenon is
`FLUCTUATION_REL_L2_IMPROVED_BUT_TKE_DEGRADED`; the fluctuation field itself
was not diagnosed as worsened. E2/E3 improve aggregate Rel-L2/MVPE, while TKE
wins fall to 4/16 and 3/16. E1 modestly improves TKE but loses Rel-L2 and has
only 5/16 TKE wins.

## Decision

E1, E2 and E3 are `NO_GO` for automatic continuation under the preregistered
stability requirement. E2/E3 provide the strongest aggregate Rel-L2/MVPE
evidence, and E3 is marginally best on those metrics and high-energy error,
but neither provides stable TKE evidence. This handoff records evidence only;
it does not propose or infer an MF-02 structure.

`NEXT_ACTION = REVIEW_REQUIRED`
