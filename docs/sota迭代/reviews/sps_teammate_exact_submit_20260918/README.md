# SPS-TEAMMATE-EXACT-SUBMIT-01

Status: `REVIEW_REQUIRED`

This is an execution-only replication of the frozen teammate SPS recipe on the frozen full SOTA-V2 point predictor. No point-model training or source redesign was performed.

## Result

- Execution commit: `58b9ec63baee83e81adf5c983afe117d9e1ef030`
- Full point checkpoint: `@53582`
- Full checkpoint SHA256: `f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce`
- Manifest SHA256: `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`
- Uncertainty head: teammate exact 35 channels, `h32 / blocks2 / dropout0 / include_pressure`
- Train: frozen 50 trajectories, 2052 canonical windows
- Dev: frozen 16 trajectories, 659 canonical windows
- Selected update: `1800`
- Selected floor / multiplier: `0.0025 / 1.0`
- Dev SPS: `44.48550611699792`
- Dev coverage: `0.8642850171482572`
- Dev mean UV width: `0.025083480402827263`
- Point prediction parity max abs diff: `0.0`
- Head SHA256: `61bc8714cf6722370685f327f1be572aaea2485d02d63d65ea48e8bbb980e261`

The recipe used seed `41`, AdamW `lr=1e-3`, `weight_decay=1e-5`, batch `8`, `2000` updates, evaluation every `200` updates, masked Gaussian NLL on non-zero target `u/v`, and the fixed 28-row floor × multiplier grid. The point predictor remained frozen.

## Package

- Remote ZIP: `/home/chyfuture/realpde_runs/sps_teammate_exact_submit_20260918/package/submission.zip`
- ZIP bytes: `30260939`
- ZIP SHA256: `bb1a8804276767362e835ad4ae15e4c558153c2510efdea7f1d75ec8063a8e71`
- Package build status: `PACKAGED`, `submission_recommended=true`

## Verification

- Required tests: `PYTHONPATH=tools pytest -q tests/test_sps_teammate_uncertainty.py tests/test_sps_teammate_exact_submit.py tests/test_sps_teammate_package.py tests/test_sota_v2_adaptive_package.py`
- Required tests result: `21 passed` (one non-fatal Torch/NumPy initialization warning)
- Clean-room smoke: `PASS`
- Smoke prediction parity: `0.0`
- Smoke deterministic max abs diff: `0.0`
- Smoke ZIP bytes / SHA256: `30260939 / bb1a8804276767362e835ad4ae15e4c558153c2510efdea7f1d75ec8063a8e71`
- Safety checks: finite outputs, `lower <= prediction <= upper`, zero pressure interval width, no fallback
- Fixture: released Train trajectory `16500_0.h5` only, via the previously verified `32x64` fixture

## Runtime notes and boundaries

- Bounded runtime/path fix: the frozen manifest files are under the verified remote released-data directory `/home/chyfuture/RealPDE_data/p0ab_real_h5_20260830` (container path `/data/p0ab_real_h5_20260830`), not directly under `/data`.
- Bounded runtime/path fix: execution used an isolated source archive at the required execution commit because the remote host's pre-existing checkout was dirty and its SSH alias was unavailable. No algorithmic/source semantics were changed.
- `locked-final/private/Codabench NOT accessed`.
- Dev16 is held out only for the uncertainty head; the full `@53582` point predictor was trained on all released trajectories, so this is not point-model OOF validation.

Large checkpoints, raw logs, and the submission ZIP remain on the remote machine and are not committed here.


## Online Codabench result

Manual submission completed on 2026-09-18 using the verified package above.

| Metric | Score |
|---|---:|
| Rel-L2 | `93.816645` |
| TKE | `79.164203` |
| MVPE | `93.411176` |
| Time | `86.828909` |
| SPS | `31.899537` |
| Final | `77.728857` |

Point-model scores are exactly unchanged from the SOTA-V2 backbone submission, which is consistent with offline point-prediction parity `0.0`. The online change is therefore attributable to the interval/SPS path plus normal runtime variation, not to a changed point predictor.

Relative to the prior 2026-09-17 full-specific teammate35 package (`SPS=31.961724`, `Final=77.732796`), this exact-recipe package changed SPS by `-0.062187` and Final by `-0.003939`. The added masked NLL, seed 41, 2000-update budget, and checkpoint/calibration selection did **not** produce an online SPS gain.

## Sol review / stable conclusion

Status after online submission: `COMPLETED / ONLINE_NO_GAIN / CLOSE_SPS_ONLY_RECIPE_TUNING`.

What this experiment supports:

- The teammate uncertainty-head training recipe is now closely replicated on our frozen full SOTA point predictor: 35-channel features, h32/b2/drop0, masked Gaussian NLL, seed 41, 2000 updates, eval every 200, and the exact 28-row SPS calibration grid.
- These recipe-level changes do not explain the teammate's online SPS `38.442870`; our online SPS remains `31.899537`.
- Do not spend another submission on SPS-only changes such as loss choice, seed, 1600/1800/2000 steps, h32/h64, 50/82 head scope, or floor/mult micro-tuning without new evidence.
- This is **not** a full end-to-end replica of the teammate system. Their uncertainty head observes the pre-correction base while the interval is centered on a residual-corrected final prediction. Our SOTA lacks the same `base -> residual corrector -> final` structure, so the remaining gap should be investigated jointly with the residual-correction/main-prediction structure rather than by further uncertainty-head-only tuning.

The submission itself did not access locked-final/private data during training or packaging. Codabench was accessed only later by the user for the manual online submission recorded above.
