# SPS-A2 Backbone / Uncertainty-Head Mismatch Audit — 2026-09-17

Status: **`COMPLETED / REVIEW_REQUIRED`**

## Scope and frozen protocol

- Execution commit: `dffa4f9772d4373fcd2329abbdfffdf92696622f`.
- Fixed manifest: 50 Train / 16 Dev, SHA-256 `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`.
- Backbone A: same validation run `@30000`, selected by the exact-30000 rule.
- Backbone B: current SOTA-V2 validation `@32500`.
- Both heads use canonical fixed stride-20 windows, `in_channels=15`, `hidden=32`, `blocks=2`, Gaussian NLL, seed `20260905`, AdamW schedule, and `1400` updates.
- Head-B was reused from the existing frozen SOTA-V2 artifact; it was not retrained.
- All four combinations use the same 16 Dev trajectories and the same official v9 scorer / 28-row `floor × mult` grid.
- No locked-final/private Future20/Codabench/full head/package/smoke access.

## Provenance

| asset | iteration / scope | SHA-256 |
|---|---|---|
| Backbone A | validation `@30000` | `d57539d97e94ae193997a59a04d02ca61f68f1e8837599a24df2e152ed10059f` |
| Backbone B | validation `@32500` | `6926722895611c38dee79cf16ce43b87c7de3d4a1305e9db431dadd4c1a7bf47` |
| Head-B | frozen validation head, reused | `01c1fc06f806ca00a370cf971df0f093d363ab7f3a4c4e473f1727441f3d4d17` |
| Head-A | fresh A-matched head, 1400 updates | `5126e24d12db70b1ba2e4af7a98041c70809d6e0647168682c2fa25e0d1206ea` |

Head-B metadata confirms B@32500, 50-train canonical windows (`2052` windows), h32/b2, and the same manifest. Backbone B prediction parity after swapping Head-B for Head-A was exactly `0.0`.

## Four combinations

| combination | SPS | coverage | mean width UV | best floor | best mult | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| A + Head-A + A calibration | 43.21291033455746 | 0.842655475616277 | 0.024530485272407532 | 0.0025 | 1.0 | 0.6844905288113411 | 0.6436330639370169 |
| B + Head-B + B calibration | 45.07008160038756 | 0.8558713772975425 | 0.02358330972492695 | 0.0025 | 1.0 | 0.673554148900692 | 0.6351773921881435 |
| B + Head-A + A calibration | 45.11577471806446 | 0.8630128898979591 | 0.023998944088816643 | 0.0025 | 1.0 | 0.6760302511241982 | 0.6290179264521235 |
| B + Head-A + B recalibration | 45.11577471806446 | 0.8630128898979591 | 0.023998944088816643 | 0.0025 | 1.0 | 0.6760302511241982 | 0.6290179264521235 |

Both A and B calibration selected the same grid point `(floor=0.0025, mult=1.0)`. Therefore the A-cal and B-cal cross-backbone rows are numerically identical in this audit.

## Mismatch interpretation

The required definitions are applied exactly:

```text
mismatch_total = SPS(B + Head-B + Bcal) - SPS(B + Head-A + Acal)
                 = 45.07008160038756 - 45.11577471806446
                 = -0.04569311767689754

mismatch_after_recalibration = SPS(B + Head-B + Bcal) - SPS(B + Head-A + Bcal)
                              = -0.04569311767689754
```

Both values are effectively zero relative to the SPS scale and do not show a material penalty from moving Head-A onto Backbone B. Recalibration does not change the result because both calibration scans choose the same `(0.0025, 1.0)` point. The observed difference is a small favorable fluctuation for the cross-backbone Head-A row, not evidence of a mismatch failure.

## Verification and evidence

- `pytest -q tests/test_sps_backbone_head_mismatch.py tests/test_sps_stride1_replica.py tests/test_sota_v2_adaptive.py tests/test_sota_v2_adaptive_package.py`: `21 passed`.
- `python -m py_compile tools/realpde_sps_backbone_head_mismatch.py tests/test_sps_backbone_head_mismatch.py`: passed.
- Independent evidence checks: `PASS`; `calibration_grids.csv` contains `84` rows (`3 × 28`).
- Remote artifact root: `/home/chyfuture/realpde_runs/sps_backbone_head_mismatch_20260917/run`.

Evidence files in this directory:

- `summary.json`
- `combinations.csv`
- `calibration_grids.csv`
- `head_provenance.json`

No next experiment is authorized by this audit.

## Final conclusion

BACKBONE_HEAD_MISMATCH_NOT_SUPPORTED
