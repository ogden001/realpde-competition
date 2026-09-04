# Track 1 Cross-Split Duplicate Audit

Conclusion: `DUPLICATE_AUDIT_CLEAN`

The audit covers the frozen 50 Train / 16 Dev / 16 Final manifest using the
formal `T_in=20`, `T_future=20`, `stride=20` window protocol. It checks only
Past20 `u/v` input windows. No Final Future20 slice was read, no model or
metric was run, and the manifest was not modified.

## Exact duplicate

One cross-split pair has identical Past20 inputs for every valid window:

| Split A | Trajectory A | Split B | Trajectory B | Descriptor distance | Past20 max abs diff | Past20 mean abs diff |
|---|---|---|---|---:|---:|---:|
| Train | `6300_0.h5` | Final | `7575_0.h5` | 0.0 | 0.0 | 0.0 |

Both trajectories have 42 valid windows. The equality is exact at the stored
`u/v` array level, not a floating-point tolerance artifact.

## Near-duplicate candidates

Candidate screening uses Euclidean distance between the 17 existing input
descriptors after one all-82 standardization, with threshold `distance < 0.1`.
The only candidate is the exact pair above. The next closest cross-split
distances are:

- Train ↔ Dev: `0.573602` (`12675_10.h5` ↔ `11400_10.h5`)
- Train ↔ Final: `0.653983` (`12675_15.h5` ↔ `13950_15.h5`)
- Dev ↔ Final: `0.574815` (`8850_10.h5` ↔ `7575_10.h5`)

Thus there is no second near-duplicate group requiring follow-up input-side
comparison.

## Decision

`DUPLICATE_AUDIT_CLEAN`: apart from the known repeated Past20 pair, no
additional cross-split duplicate or near-duplicate was found. This does not
alter the accepted 50/16/16 split and does not trigger re-splitting.

Machine-readable outputs are in
`artifacts/duplicate_audit_20260904/duplicate_pairs.csv` and
`artifacts/duplicate_audit_20260904/duplicate_audit_summary.json`.
