# Sol Review — Overnight Integrated Adaptive Probe

Status: **`REVIEW_REQUIRED / BASELINE_PARITY_AUDIT`**

## 1. Repaired implementation review

The material semantic bugs from the first run have been repaired in the repository:

- `residual_mse` now targets `target - frozen_backbone_prediction`;
- base and corrected uncertainty heads are separated;
- fresh uncertainty head starts at `sigma=0.02`;
- `full=True` no longer refits uncertainty;
- repository Gate evaluator uses the official scorer path;
- corrected-head-only validation path exists.

The v4 evidence is internally consistent:

- Gate: `PASS`;
- baseline raw errors: Rel-L2 `0.12998666`, TKE `0.66506779`, MVPE `0.16063145`;
- corrected raw errors: Rel-L2 `0.09917600`, TKE `0.54960150`, MVPE `0.09375764`;
- relative improvement: Rel-L2 `23.7029%`, TKE `17.3616%`, MVPE `41.6318%`;
- TKE degradation `>15%`: `0/16` trajectories;
- fixed 56-row calibration grid complete;
- best base SPS: `37.644685` at `floor=0.0025, mult=1`;
- best corrected SPS: `44.145264` at `floor=0.0025, mult=1`.

These are strong signals, but they are **not yet sufficient to authorize full refit**.

## 2. Baseline parity blocker

The registered SOTA documentation records the same validation family at update `30900` as:

- Rel-L2 `0.11284460`
- TKE `0.50010282`
- MVPE `0.08728255`

The v4 Gate baseline is substantially worse:

- Rel-L2 `0.12998666` — about `15.19%` worse;
- TKE `0.66506779` — about `32.99%` worse;
- MVPE `0.16063145` — about `84.04%` worse.

This discrepancy is too large to treat as noise or aggregation variation. The Gate evaluator uses the same official raw-error formulas as the repository scoring helper, so the difference must be explained before the `+23.7% / +17.4% / +41.6%` corrector gains can be attributed relative to the canonical strong validation backbone.

Possible causes to audit, not assume:

1. wrong or stale checkpoint despite the path name `...10300_to30900/.../model_last.pth`;
2. checkpoint content / iteration / feature config mismatch;
3. manifest or dataset-window mismatch;
4. P0-A builder/runtime config mismatch;
5. historical SOTA validation evidence was produced by a different prediction path.

Current v4 metadata records checkpoint SHA256:

`e3d5faaf1a71e121b09077dd7dd7d0456a617e2916b8a671986f412fb54f6388`

This SHA must be reconciled with the checkpoint that produced the registered SOTA validation raw errors.

## 3. Provenance cleanup needed

The review README still names execution commit `1aa0b50...` and remote `OUT_ROOT ..._v3`, while the repaired evidence is explicitly under `..._v4` and depends on later repair commits. This does not invalidate the numerical files by itself, but the final handoff must record the actual code commit / runner SHA used by v4.

## 4. Sol decision

**Do not start all-82 refit or package yet. Do not retrain anything yet.**

Run one bounded baseline-parity audit on the existing artifacts/checkpoint:

- verify checkpoint iteration / feature config / manifest provenance;
- evaluate the exact same base checkpoint on the same 659 Dev windows through both the canonical SOTA validation path and the current adaptive Gate path;
- compare base predictions numerically, not only scalar metrics;
- if parity holds, add per-trajectory Rel-L2/MVPE alongside TKE and return to Sol;
- if parity fails because the v4 corrector was trained against a wrong backbone/config, mark v4 validation invalid and stop.

If baseline parity is established and the large corrected gains remain, the expected next Sol decision is `GO_FULL_REFIT`.

`NEXT_ACTION = BASELINE_PARITY_AUDIT_ONLY`
